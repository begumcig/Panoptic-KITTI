Created JSON files for training and validation data using the panoptic images.

Transformed the dataset into Detectron2’s dataset dictionary format.

Modified panoptic validation implementation of Detectron2 to work with the KITTI dataset.

Full training and validation done for Panoptic Deeplab with KITTI dataset using Detectron2.

The data loader is modified to include previous frames.

A new instance segmentation added to learn previous frame offsets.

New ground truths are created from previous frame panoptic segmentation to train the new instance head. 
