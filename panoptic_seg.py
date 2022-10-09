# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import vis_utils 
from PIL import Image
from panopticapi.utils import rgb2id
import sys
import os 
import time
from typing import Callable, Dict, List, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.layers import Conv2d, DepthwiseSeparableConv2d, ShapeSpec, get_norm
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    SEM_SEG_HEADS_REGISTRY,
    build_backbone,
    build_sem_seg_head,
)
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.projects.deeplab import DeepLabV3PlusHead
from detectron2.projects.deeplab.loss import DeepLabCE
from detectron2.structures import BitMasks, ImageList, Instances
from detectron2.utils.registry import Registry

from detectron2.projects.panoptic_deeplab.post_processing import get_panoptic_segmentation


__all__ = ["PanopticDeepLab", "INS_EMBED_BRANCHES_REGISTRY", "build_ins_embed_branch", "PREV_FRAME_INS_HEAD_REGISTRY", "build_prev_frame_ins_head"]


INS_EMBED_BRANCHES_REGISTRY = Registry("INS_EMBED_BRANCHES")
INS_EMBED_BRANCHES_REGISTRY.__doc__ = """
Registry for instance embedding branches, which make instance embedding
predictions from feature maps.
"""
PREV_FRAME_INS_HEAD_REGISTRY = Registry("PREV_FRAME_INS_HEAD")
PREV_FRAME_INS_HEAD_REGISTRY.__doc__ = """
Registry for previous frame instance embedding branches, which make instance embedding
predictions from feature maps.
"""


_IMAGE_FORMAT = '%s_image'
_PREV_IMAGE_FORMAT = '%s_prev_image'
_SEMANTIC_LABEL_FORMAT = '%s_semantic_label'
_CENTER_LABEL_FORMAT = '%s_center_label'
_PREV_CENTER_LABEL_FORMAT = '%s_prev_center_label'
_OFFSET_LABEL_FORMAT = '%s_offset_label'
_FRAME_OFFSET_LABEL_FORMAT = '%s_frame_offset_label'
_PANOPTIC_LABEL_FORMAT = '%s_panoptic_label'
_PANOPTIC_PREV_LABEL_FORMAT = '%s_panoptic_prev_label'


_SEMANTIC_PREDICTION_FORMAT = '%s_semantic_prediction'
_CENTER_HEATMAP_PREDICTION_FORMAT = '%s_center_prediction'
_OFFSET_PREDICTION_RGB_FORMAT = '%s_offset_prediction_rgb'
_FRAME_OFFSET_PREDICTION_RGB_FORMAT = '%s_frame_offset_prediction_rgb'



def _get_fg_mask(label_map: np.ndarray, thing_list: List[int]) -> np.ndarray:
  fg_mask = np.zeros_like(label_map, np.bool)
  for class_id in np.unique(label_map):
    if class_id in thing_list:
      fg_mask = np.logical_or(fg_mask, np.equal(label_map, class_id))
  fg_mask = np.expand_dims(fg_mask, axis=2)
  return fg_mask.astype(np.int)



@META_ARCH_REGISTRY.register()
class PanopticDeepLabKitti(nn.Module):
    """
    Main class for panoptic segmentation architectures.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_output_shape = self.backbone.output_shape()
        concat_output_shape = {k: ShapeSpec(channels = v.channels * 2 , stride = v.stride) for k,v in backbone_output_shape.items()}
        self.sem_seg_head = build_sem_seg_head(cfg, backbone_output_shape)
        self.ins_embed_head = build_ins_embed_branch(cfg, concat_output_shape)
        self.prev_ins_embed_head =  build_prev_frame_ins_head(cfg, backbone_output_shape)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)
        self.meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.stuff_area = cfg.MODEL.PANOPTIC_DEEPLAB.STUFF_AREA
        self.threshold = cfg.MODEL.PANOPTIC_DEEPLAB.CENTER_THRESHOLD
        self.nms_kernel = cfg.MODEL.PANOPTIC_DEEPLAB.NMS_KERNEL
        self.top_k = cfg.MODEL.PANOPTIC_DEEPLAB.TOP_K_INSTANCE
        self.predict_instances = cfg.MODEL.PANOPTIC_DEEPLAB.PREDICT_INSTANCES
        self.use_depthwise_separable_conv = cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV
        assert (
            cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV
            == cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV
        )
        self.size_divisibility = cfg.MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY
        self.benchmark_network_speed = cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, store_images=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * "center": center points heatmap ground truth
                   * "offset": pixel offsets to center points ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:
                * "panoptic_seg", "sem_seg": see documentation
                    :doc:`/tutorials/models` for the standard output format
                * "instances": available if ``predict_instances is True``. see documentation
                    :doc:`/tutorials/models` for the standard output format
        """
        
        image_id = batched_inputs[0]["image_id"]
        thing_list = self.meta.thing_dataset_id_to_contiguous_id.values()
        
        '''
        if self.training:
            
            prev_img = batched_inputs[0]["prev_image"]
            prev_img = prev_img.cpu().numpy().astype(dtype=np.uint8)
            prev_img = np.transpose(prev_img, (1,2,0))

            prev_pan_gt = np.array(Image.open(batched_inputs[0]["prev_pan_seg_file_name"]), dtype=np.uint32)
            prev_pan_gt_vis = prev_pan_gt[:,:,0] * self.meta.label_divisor + prev_pan_gt[:,:,1] * 256 + prev_pan_gt[:,:,2] 
            image_number = os.path.basename(image_id)
            prev_image_number = '%06d' % (int(image_number) - 1)
            prev_image_id = os.path.join(os.path.dirname(image_id), prev_image_number)

            vis_utils.save_parsing_result(
            parsing_result=prev_pan_gt_vis,
            label_divisor=self.meta.label_divisor,
            thing_list=[11,13],
            save_dir="./output",
            filename=_PANOPTIC_PREV_LABEL_FORMAT % image_id,
            colormap_name="cityscapes")

        
            sem_seg_img = batched_inputs[0]["sem_seg"]
            sem_seg_img = torch.squeeze(sem_seg_img, axis=0).cpu().numpy()

            center_img = batched_inputs[0]["center"]
            center_img = torch.squeeze(center_img, axis=0).cpu().numpy()

            prev_center_img = batched_inputs[0]["prev_center"]
            prev_center_img = torch.squeeze(prev_center_img, axis=0).cpu().numpy()


            vis_utils.save_annotation(
                sem_seg_img,
                "./output",
                _SEMANTIC_LABEL_FORMAT % image_id,
                add_colormap = True
                )

            offset_img = batched_inputs[0]["offset"]
            offset_img = torch.squeeze(offset_img, axis=0).cpu().numpy()
            offset_img = np.transpose(offset_img, (1,2,0))

            frame_offset_img = batched_inputs[0]["frame_offset"]
            frame_offset_img = torch.squeeze(frame_offset_img, axis=0).cpu().numpy()
            frame_offset_img = np.transpose(frame_offset_img, (1,2,0))

            

            center_offset_label_rgb = vis_utils.flow_to_color(offset_img)
            
            gt_fg_mask = _get_fg_mask(sem_seg_img, thing_list)
            center_offset_label_rgb = center_offset_label_rgb * gt_fg_mask

            frame_center_offset_label_rgb = vis_utils.flow_to_color(frame_offset_img)
            frame_center_offset_label_rgb = frame_center_offset_label_rgb * gt_fg_mask
            
            #offset 
            vis_utils.save_annotation(
                center_offset_label_rgb,
                "./output",
                _OFFSET_LABEL_FORMAT % image_id,
                add_colormap=False)
            
            #frame_offset
            vis_utils.save_annotation(
                frame_center_offset_label_rgb,
                "./output",
                _FRAME_OFFSET_LABEL_FORMAT % image_id,
                add_colormap=False
            )
        '''

        images = [x["image"].to(self.device) for x in batched_inputs]
        img = images[0].cpu().numpy().astype(dtype=np.uint8)
        img = np.transpose(img, (1,2,0))
        
        '''
        if self.training:
        #input image
            vis_utils.save_annotation(
                img,
                "./output",
                _IMAGE_FORMAT % image_id,
                add_colormap = False
                )

            
            vis_utils.save_annotation(
                prev_img,
                "./output",
                _PREV_IMAGE_FORMAT % image_id,
                add_colormap = False
                )

        #center heatmap
            vis_utils.save_annotation(
                vis_utils.overlay_heatmap_on_image(
                    center_img, 
                    img
                ),
                "./output",
                _CENTER_LABEL_FORMAT % image_id,
                add_colormap = False
            )

            vis_utils.save_annotation(
                vis_utils.overlay_heatmap_on_image(
                    prev_center_img, 
                    prev_img
                ),
                "./output",
                _PREV_CENTER_LABEL_FORMAT % image_id,
                add_colormap = False
            )
        '''
        
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        
      
        
        # To avoid error in ASPP layer when input has different size.
        size_divisibility = (
            self.size_divisibility
            if self.size_divisibility > 0
            else self.backbone.size_divisibility
        )
        images = ImageList.from_tensors(images, size_divisibility)
        
        prev_images = [x["prev_image"].to(self.device) for x in batched_inputs]
        prev_images = [(x - self.pixel_mean) / self.pixel_std for x in prev_images]
        prev_images  = ImageList.from_tensors(prev_images, size_divisibility)

        batch_size = images.tensor.shape[0]
        images_to_backbone = torch.hstack((torch.flatten(images.tensor, start_dim=1), torch.flatten(prev_images.tensor, start_dim=1))).view(batch_size * 2, images.tensor.shape[1], images.tensor.shape[2], images.tensor.shape[3])
        

        
        features = self.backbone(images_to_backbone)
        
        current_features = {key: value[::2] for key,value in features.items()}
        concat_features = {key: torch.cat((value[::2], value[1::2]), dim=1) for key,value in features.items()}
        
        

        losses = {}
        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
            if "sem_seg_weights" in batched_inputs[0]:
                # The default D2 DatasetMapper may not contain "sem_seg_weights"
                # Avoid error in testing when default DatasetMapper is used.
                weights = [x["sem_seg_weights"].to(self.device) for x in batched_inputs]
                weights = ImageList.from_tensors(weights, size_divisibility).tensor
            else:
                weights = None
        else:
            targets = None
            weights = None
        sem_seg_results, sem_seg_losses = self.sem_seg_head(current_features, targets, weights)
        losses.update(sem_seg_losses)

        if "center" in batched_inputs[0] and "offset" in batched_inputs[0]:
            center_targets = [x["center"].to(self.device) for x in batched_inputs]
            center_targets = ImageList.from_tensors(
                center_targets, size_divisibility
            ).tensor.unsqueeze(1)
            center_weights = [x["center_weights"].to(self.device) for x in batched_inputs]
            center_weights = ImageList.from_tensors(center_weights, size_divisibility).tensor

            offset_targets = [x["offset"].to(self.device) for x in batched_inputs]
            offset_targets = ImageList.from_tensors(offset_targets, size_divisibility).tensor
            offset_weights = [x["offset_weights"].to(self.device) for x in batched_inputs]
            offset_weights = ImageList.from_tensors(offset_weights, size_divisibility).tensor
            if "frame_offset" in batched_inputs[0]:
                frame_offset_targets = [x["frame_offset"].to(self.device) for x in batched_inputs]
                frame_offset_targets = ImageList.from_tensors(frame_offset_targets, size_divisibility).tensor
                frame_offset_weights = [x["frame_offset_weights"].to(self.device) for x in batched_inputs]
                frame_offset_weights = ImageList.from_tensors(frame_offset_weights, size_divisibility).tensor
        else:
            center_targets = None
            center_weights = None

            offset_targets = None
            offset_weights = None

            frame_offset_targets = None
            frame_offset_weights = None

        
        center_results, offset_results, center_losses, offset_losses = self.prev_ins_embed_head(
            current_features, center_targets, center_weights, offset_targets, offset_weights
        )
        frame_offset_results, frame_offset_losses = self.ins_embed_head(
            concat_features, frame_offset_targets, frame_offset_weights
        )
        
        losses.update(center_losses)
        losses.update(offset_losses)
        losses.update(frame_offset_losses)
        

        if self.training:
            return losses

        if self.benchmark_network_speed:
            return []

        processed_results = []
        for sem_seg_result, center_result, offset_result, frame_offset_result, input_per_image, image_size in zip(
            sem_seg_results, center_results, offset_results, frame_offset_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
            c = sem_seg_postprocess(center_result, image_size, height, width)
            o = sem_seg_postprocess(offset_result, image_size, height, width)
            fo = sem_seg_postprocess(frame_offset_result, image_size, height, width)
            # Post-processing to get panoptic segmentation.
            panoptic_image, _ = get_panoptic_segmentation(
                r.argmax(dim=0, keepdim=True),
                c,
                o,
                thing_ids=self.meta.thing_dataset_id_to_contiguous_id.values(),
                label_divisor=self.meta.label_divisor,
                stuff_area=self.stuff_area,
                void_label=-1,
                threshold=self.threshold,
                nms_kernel=self.nms_kernel,
                top_k=self.top_k,
            )
            # For semantic segmentation evaluation.
            processed_results.append({"sem_seg": r})
            panoptic_image = panoptic_image.squeeze(0)
            semantic_prob = F.softmax(r, dim=0)
            
            '''
            r_img = r.argmax(dim=0)
            r_img = r_img.cpu().numpy()
            vis_utils.save_annotation(
                r_img,
                "./output",
                _SEMANTIC_PREDICTION_FORMAT % image_id,
                add_colormap = True
                )
            c_img = torch.squeeze(c, axis=0).cpu().numpy()

            vis_utils.save_annotation(
                    vis_utils.overlay_heatmap_on_image(
                    c_img,
                    img),
                "./output",
                _CENTER_HEATMAP_PREDICTION_FORMAT % image_id,
                add_colormap=False)
            
            o_img = o.cpu().numpy()
            o_img = np.transpose(o_img, (1,2,0))
            
            center_offset_prediction_rgb = vis_utils.flow_to_color(o_img)
            pred_fg_mask = _get_fg_mask(r_img, thing_list)
            center_offset_prediction_rgb = (center_offset_prediction_rgb * pred_fg_mask)
            vis_utils.save_annotation(
                center_offset_prediction_rgb,
                "./output",
                _OFFSET_PREDICTION_RGB_FORMAT % image_id,
                add_colormap=False)
            
            of_img = fo.cpu().numpy()
            of_img = np.transpose(of_img, (1,2,0))

            center_f_offset_prediction_rgb = vis_utils.flow_to_color(of_img)
            pred_fg_mask = _get_fg_mask(r_img, thing_list)
            center_f_offset_prediction_rgb = (center_f_offset_prediction_rgb * pred_fg_mask)
            vis_utils.save_annotation(
                center_f_offset_prediction_rgb,
                "./output",
                _FRAME_OFFSET_PREDICTION_RGB_FORMAT % image_id,
                add_colormap=False)
            '''

            # For panoptic segmentation evaluation.
            processed_results[-1]["panoptic_seg"] = (panoptic_image, None)
            # For instance segmentation evaluation.
            if self.predict_instances:
                instances = []
                panoptic_image_cpu = panoptic_image.cpu().numpy()
                for panoptic_label in np.unique(panoptic_image_cpu):
                    if panoptic_label == -1:
                        continue
                    pred_class = panoptic_label // self.meta.label_divisor
                    isthing = pred_class in list(
                        self.meta.thing_dataset_id_to_contiguous_id.values()
                    )
                    # Get instance segmentation results.
                    if isthing:
                        instance = Instances((height, width))
                        # Evaluation code takes continuous id starting from 0
                        instance.pred_classes = torch.tensor(
                            [pred_class], device=panoptic_image.device
                        )
                        mask = panoptic_image == panoptic_label
                        instance.pred_masks = mask.unsqueeze(0)
                        # Average semantic probability
                        sem_scores = semantic_prob[pred_class, ...]
                        sem_scores = torch.mean(sem_scores[mask])
                        # Center point probability
                        mask_indices = torch.nonzero(mask).float()
                        center_y, center_x = (
                            torch.mean(mask_indices[:, 0]),
                            torch.mean(mask_indices[:, 1]),
                        )
                        center_scores = c[0, int(center_y.item()), int(center_x.item())]
                        # Confidence score is semantic prob * center prob.
                        instance.scores = torch.tensor(
                            [sem_scores * center_scores], device=panoptic_image.device
                        )
                        # Get bounding boxes
                        instance.pred_boxes = BitMasks(instance.pred_masks).get_bounding_boxes()
                        instances.append(instance)
                if len(instances) > 0:
                    processed_results[-1]["instances"] = Instances.cat(instances)

        return processed_results


    


@SEM_SEG_HEADS_REGISTRY.register()
class PanopticDeepLabSemSegHeadKitti(DeepLabV3PlusHead):
    """
    A semantic segmentation head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        loss_weight: float,
        loss_type: str,
        loss_top_k: float,
        ignore_value: int,
        num_classes: int,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            loss_weight (float): loss weight.
            loss_top_k: (float): setting the top k% hardest pixels for
                "hard_pixel_mining" loss.
            loss_type, ignore_value, num_classes: the same as the base class.
        """
        super().__init__(
            input_shape,
            decoder_channels=decoder_channels,
            norm=norm,
            ignore_value=ignore_value,
            **kwargs,
        )
        assert self.decoder_only

        self.loss_weight = loss_weight
        use_bias = norm == ""
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.head = DepthwiseSeparableConv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.head = nn.Sequential(
                Conv2d(
                    decoder_channels[0],
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.head[0])
            weight_init.c2_xavier_fill(self.head[1])
        self.predictor = Conv2d(head_channels, num_classes, kernel_size=1)
        nn.init.normal_(self.predictor.weight, 0, 0.001)
        nn.init.constant_(self.predictor.bias, 0)

        if loss_type == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_value)
        elif loss_type == "hard_pixel_mining":
            self.loss = DeepLabCE(ignore_label=ignore_value, top_k_percent_pixels=loss_top_k)
        else:
            raise ValueError("Unexpected loss type: %s" % loss_type)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["head_channels"] = cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS
        ret["loss_top_k"] = cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K
        return ret

    def forward(self, features, targets=None, weights=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y = self.layers(features)
        if self.training:
            return None, self.losses(y, targets, weights)
        else:
            y = F.interpolate(
                y, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return y, {}

    def layers(self, features):
        assert self.decoder_only
        y = super().layers(features)
        y = self.head(y)
        y = self.predictor(y)
        return y

    def losses(self, predictions, targets, weights=None):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.loss(predictions, targets, weights)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


def build_ins_embed_branch(cfg, input_shape):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.INS_EMBED_HEAD.NAME
    return INS_EMBED_BRANCHES_REGISTRY.get(name)(cfg, input_shape)


def build_prev_frame_ins_head(cfg, input_shape):
    """
    Build a instance embedding branch from `cfg.MODEL.PREV_INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.PREV_INS_EMBED_HEAD.NAME
    return PREV_FRAME_INS_HEAD_REGISTRY.get(name)(cfg, input_shape)


@INS_EMBED_BRANCHES_REGISTRY.register()
class PanopticDeepLabInsEmbedHeadKitti(DeepLabV3PlusHead):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        offset_loss_weight: float,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            center_loss_weight (float): loss weight for center point prediction.
            offset_loss_weight (float): loss weight for center offset prediction.
        """
        super().__init__(input_shape, decoder_channels=decoder_channels, norm=norm, **kwargs)
        assert self.decoder_only

        self.offset_loss_weight = offset_loss_weight
        use_bias = norm == ""
        # offset prediction
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.offset_head = DepthwiseSeparableConv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.offset_head = nn.Sequential(
                Conv2d(
                    decoder_channels[0],
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.offset_head[0])
            weight_init.c2_xavier_fill(self.offset_head[1])
        self.offset_predictor = Conv2d(head_channels, 2, kernel_size=1)
        nn.init.normal_(self.offset_predictor.weight, 0, 0.001)
        nn.init.constant_(self.offset_predictor.bias, 0)

        self.offset_loss = nn.L1Loss(reduction="none")
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.PREV_INS_EMBED_HEAD.CONVS_DIM] * (
            len(cfg.MODEL.PREV_INS_EMBED_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.PREV_INS_EMBED_HEAD.ASPP_CHANNELS]
        ret = dict(
            input_shape={
                k: v for k, v in input_shape.items() if k in cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES
            },
            project_channels=cfg.MODEL.PREV_INS_EMBED_HEAD.PROJECT_CHANNELS,
            aspp_dilations=cfg.MODEL.PREV_INS_EMBED_HEAD.ASPP_DILATIONS,
            aspp_dropout=cfg.MODEL.PREV_INS_EMBED_HEAD.ASPP_DROPOUT,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.PREV_INS_EMBED_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.PREV_INS_EMBED_HEAD.NORM,
            train_size=train_size,
            head_channels=cfg.MODEL.PREV_INS_EMBED_HEAD.HEAD_CHANNELS,
            offset_loss_weight=cfg.MODEL.PREV_INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
        )
        return ret
    
    def forward(
        self,
        features,
        offset_targets=None,
        offset_weights=None,
    ):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        offset = self.layers(features)
        if self.training:
            return (
                None,
                self.offset_losses(offset, offset_targets, offset_weights),
            )
        else:
            offset = (
                F.interpolate(
                    offset, scale_factor=self.common_stride, mode="bilinear", align_corners=False
                )
                * self.common_stride
            )
            return offset, {}

    def layers(self, features):
        assert self.decoder_only
        y = super().layers(features)
        # offset
        offset = self.offset_head(y)
        offset = self.offset_predictor(offset)
        return offset

    def offset_losses(self, predictions, targets, weights):
        predictions = (
            F.interpolate(
                predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            * self.common_stride
        )
        loss = self.offset_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_frame_offset": loss * self.offset_loss_weight}
        return losses


@PREV_FRAME_INS_HEAD_REGISTRY.register()
class PanopticDeepLabPrevInsEmbedHeadKitti(DeepLabV3PlusHead):
    """
    A instance embedding head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        center_loss_weight: float,
        offset_loss_weight: float,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            center_loss_weight (float): loss weight for center point prediction.
            offset_loss_weight (float): loss weight for center offset prediction.
        """
        super().__init__(input_shape, decoder_channels=decoder_channels, norm=norm, **kwargs)
        assert self.decoder_only

        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight
        use_bias = norm == ""
        # center prediction
        # `head` is additional transform before predictor
        self.center_head = nn.Sequential(
            Conv2d(
                decoder_channels[0],
                decoder_channels[0],
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, decoder_channels[0]),
                activation=F.relu,
            ),
            Conv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, head_channels),
                activation=F.relu,
            ),
        )
        weight_init.c2_xavier_fill(self.center_head[0])
        weight_init.c2_xavier_fill(self.center_head[1])
        self.center_predictor = Conv2d(head_channels, 1, kernel_size=1)
        nn.init.normal_(self.center_predictor.weight, 0, 0.001)
        nn.init.constant_(self.center_predictor.bias, 0)

        # offset prediction
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.offset_head = DepthwiseSeparableConv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.offset_head = nn.Sequential(
                Conv2d(
                    decoder_channels[0],
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.offset_head[0])
            weight_init.c2_xavier_fill(self.offset_head[1])
        self.offset_predictor = Conv2d(head_channels, 2, kernel_size=1)
        nn.init.normal_(self.offset_predictor.weight, 0, 0.001)
        nn.init.constant_(self.offset_predictor.bias, 0)

        self.center_loss = nn.MSELoss(reduction="none")
        self.offset_loss = nn.L1Loss(reduction="none")

    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.INS_EMBED_HEAD.CONVS_DIM] * (
            len(cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.INS_EMBED_HEAD.ASPP_CHANNELS]
        ret = dict(
            input_shape={
                k: v for k, v in input_shape.items() if k in cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES
            },
            project_channels=cfg.MODEL.INS_EMBED_HEAD.PROJECT_CHANNELS,
            aspp_dilations=cfg.MODEL.INS_EMBED_HEAD.ASPP_DILATIONS,
            aspp_dropout=cfg.MODEL.INS_EMBED_HEAD.ASPP_DROPOUT,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.INS_EMBED_HEAD.NORM,
            train_size=train_size,
            head_channels=cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS,
            center_loss_weight=cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT,
            offset_loss_weight=cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
        )
        return ret

    def forward(
        self,
        features,
        center_targets=None,
        center_weights=None,
        offset_targets=None,
        offset_weights=None,
    ):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        center, offset = self.layers(features)
        if self.training:
            return (
                None,
                None,
                self.center_losses(center, center_targets, center_weights),
                self.offset_losses(offset, offset_targets, offset_weights),
            )
        else:
            center = F.interpolate(
                center, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            
            offset = (
                F.interpolate(
                    offset, scale_factor=self.common_stride, mode="bilinear", align_corners=False
                )
                * self.common_stride
            )
            return center, offset, {}, {}

    def layers(self, features):
        assert self.decoder_only
        y = super().layers(features)
        # center
        center = self.center_head(y)
        center = self.center_predictor(center)
        #offset
        offset = self.offset_head(y)
        offset = self.offset_predictor(offset)
        return center, offset

    def center_losses(self, predictions, targets, weights):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.center_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_center": loss * self.center_loss_weight}
        return losses

    def offset_losses(self, predictions, targets, weights):
        predictions = (
            F.interpolate(
                predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            * self.common_stride
        )
        loss = self.offset_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_offset": loss * self.offset_loss_weight}
        return losses

'''
@PREV_FRAME_INS_HEAD_REGISTRY.register()
class PanopticDeepLabPrevInsEmbedHeadKitti(DeepLabV3PlusHead):
    """
    A instance embedding head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        center_loss_weight: float,
        offset_loss_weight: float,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            center_loss_weight (float): loss weight for center point prediction.
            offset_loss_weight (float): loss weight for center offset prediction.
        """
        super().__init__(input_shape, decoder_channels=decoder_channels, norm=norm, **kwargs)
        assert self.decoder_only

        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight
        use_bias = norm == ""
        # center prediction
        # `head` is additional transform before predictor
        self.center_head = nn.Sequential(
            Conv2d(
                decoder_channels[0],
                decoder_channels[0],
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, decoder_channels[0]),
                activation=F.relu,
            ),
            Conv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, head_channels),
                activation=F.relu,
            ),
        )
        weight_init.c2_xavier_fill(self.center_head[0])
        weight_init.c2_xavier_fill(self.center_head[1])
        self.center_predictor = Conv2d(head_channels, 1, kernel_size=1)
        nn.init.normal_(self.center_predictor.weight, 0, 0.001)
        nn.init.constant_(self.center_predictor.bias, 0)

        # offset prediction
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.offset_head = DepthwiseSeparableConv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.offset_head = nn.Sequential(
                Conv2d(
                    decoder_channels[0],
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.offset_head[0])
            weight_init.c2_xavier_fill(self.offset_head[1])
        self.offset_predictor = Conv2d(head_channels, 2, kernel_size=1)
        nn.init.normal_(self.offset_predictor.weight, 0, 0.001)
        nn.init.constant_(self.offset_predictor.bias, 0)

        self.center_loss = nn.MSELoss(reduction="none")
        self.offset_loss = nn.L1Loss(reduction="none")

    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.INS_EMBED_HEAD.CONVS_DIM] * (
            len(cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.INS_EMBED_HEAD.ASPP_CHANNELS]
        ret = dict(
            input_shape={
                k: v for k, v in input_shape.items() if k in cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES
            },
            project_channels=cfg.MODEL.INS_EMBED_HEAD.PROJECT_CHANNELS,
            aspp_dilations=cfg.MODEL.INS_EMBED_HEAD.ASPP_DILATIONS,
            aspp_dropout=cfg.MODEL.INS_EMBED_HEAD.ASPP_DROPOUT,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.INS_EMBED_HEAD.NORM,
            train_size=train_size,
            head_channels=cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS,
            center_loss_weight=cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT,
            offset_loss_weight=cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
        )
        return ret

    def forward(
        self,
        features,
        center_targets=None,
        center_weights=None,
        offset_targets=None,
        offset_weights=None,
    ):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        center, offset = self.layers(features)
        if self.training:
            return (
                None,
                None,
                self.center_losses(center,center_targets, center_weights),
                self.offset_losses(offset, offset_targets, offset_weights),
            )
        else:
            
            center = F.interpolate(
                center, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            
            offset = (
                F.interpolate(
                    offset, scale_factor=self.common_stride, mode="bilinear", align_corners=False
                )
                * self.common_stride
            )
            return center, offset, {}, {}

    def layers(self, features):
        assert self.decoder_only
        y = super().layers(features)
        # center
        center = self.center_head(y)
        center = self.center_predictor(center)
        #offset
        offset = self.offset_head(y)
        offset = self.offset_predictor(offset)
        return center, offset

    def center_losses(self, predictions, targets, weights):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.center_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_center": loss * self.center_loss_weight}
        return losses

    def offset_losses(self, predictions, targets, weights):
        predictions = (
            F.interpolate(
                predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            * self.common_stride
        )
        loss = self.offset_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_offset": loss * self.offset_loss_weight}
        return losses
'''