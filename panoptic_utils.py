import json
import os.path
import numpy as np
from detectron2.data import MetadataCatalog, DatasetCatalog
from PIL import Image
import cv2
from os import listdir
from os import walk

train_root = "images/train"
train_panoptic_root = "panoptic_maps/train"
val_root = "images/val"
val_panoptic_root = "panoptic_maps/val"
panoptic_root = "panoptic_maps"
image_root = "images"
panoptic_train_json = "panoptic_train.json"
panoptic_val_json = "panoptic_val.json"
panoptic_overfit_train_json = "panoptic_overfit_train.json"
panoptic_overfit_val_json = "panoptic_overfit_val.json"
VOID = 255
categories = [{"id":0, "name":"road", "isthing":0},
              {"id":1, "name":"sidewalk", "isthing":0},
              {"id":2, "name":"building", "isthing":0},
              {"id":3, "name":"wall", "isthing":0},
              {"id":4, "name":"fence", "isthing":0},
              {"id":5, "name":"pole", "isthing":0},
              {"id":6, "name":"traffic light", "isthing":0},
              {"id":7, "name":"traffic sign", "isthing":0},
              {"id":8, "name":"vegetation", "isthing":0},
              {"id":9, "name":"terrain", "isthing":0},
              {"id":10, "name":"sky", "isthing":0},
              {"id":11, "name":"person", "isthing":1},
              {"id":12, "name":"rider", "isthing":0},
              {"id":13, "name":"car", "isthing":1},
              {"id":14, "name":"truck", "isthing":0},
              {"id":15, "name":"bus", "isthing":0},
              {"id":16, "name":"train", "isthing":0},
              {"id":17, "name":"motorcycle", "isthing":0},
              {"id":18, "name":"bicycle", "isthing":0}
              ]

def get_files(root):
  files = []
  for (dirpath, dirnames, filenames) in walk(root):
    for f in filenames:
      name = os.path.join(dirpath, f)
      sequence = dirpath[-4:]
      frame = f[:-4]
      key = "/".join([sequence, frame])
      files.append(key)
  return files

def get_files_overfit(root):
  files = []
  for (dirpath, dirnames, filenames) in walk(root):
    for f in filenames:
    
      key = f[:-4]
      files.append(key)
  return files



def get_kitti_dicts(img_dir, pan_dir, keys):
  dataset_dicts = []
  for k in keys:
    record = {}
    k_ext = k + ".png"
    filename = os.path.join(img_dir, k_ext)
    pan_filename = os.path.join(pan_dir, k_ext)
    height, width = cv2.imread(filename).shape[:2]

    record["file_name"] = filename
    record["image_id"] = k
    record["height"] = height
    record["width"] = width
    record["pan_seg_file_name"] = pan_filename

    #read the panotpic map pixel by pixel to extract semantic id and category id
   
    panoptic_img = Image.open(pan_filename)
    panoptic_img = np.asarray(panoptic_img)
    ids = np.zeros_like(panoptic_img)
    if isinstance(panoptic_img, np.ndarray) and len(panoptic_img.shape) == 3:
      if panoptic_img.dtype == np.uint8:
        panoptic_img = panoptic_img.astype(np.int32)
      ids = panoptic_img[:, :, 0] + 256 * panoptic_img[:, :, 1] + 256 * 256 * panoptic_img[:, :, 2]
    else:
      ids = int(panoptic_img[0] + 256 * panoptic_img[1] + 256 * 256 * panoptic_img[2])

    
    segm_ids = panoptic_img[:,:,0]
    unique_ids, indices = np.unique(ids, True)
    record["segments_info"] = []
    
    for id,idx in zip(unique_ids, indices):  
      row = idx // width
      col = idx % width     
      segm_id = segm_ids[row][col]
      if segm_id == 255:
        continue

      isInstance = segm_id == 11 or segm_id == 13
      iscrowd = 0
      if isInstance:
        if panoptic_img[:,:,1][row][col] == 0 and panoptic_img[:,:,2][row][col] == 0:
          iscrowd = 1

      '''
      if isInstance:
        if segm_id == 11:
          segm_id = 0
        elif segm_id == 13:
          segm_id = 1
      elif segm_id == 255:
        continue
      elif segm_id ==12:
        segm_id = 11
      elif segm_id >= 14:
        segm_id = segm_id - 2
      '''
  
      pan = {
          "id" : id,
          "category_id": segm_id,
          "isthing": isInstance,
          "iscrowd": iscrowd
      }
      record["segments_info"].append(pan)
    dataset_dicts.append(record)
  return dataset_dicts

def get_kitti_dicts_json(json_file, image_root, panoptic_root):
  with open(json_file, "r") as f:
    dicts = json.load(f)
  anns = dicts["annotations"]
  dataset_dicts = []
  for ann in anns:
    image_id = ann["image_id"]
    image_number = os.path.basename(image_id)
    prev_img_name = None
    prev_panoptic_name=None
    file_name = os.path.join(image_root,ann["file_name"])
    panoptic_file = os.path.join(panoptic_root,  ann["file_name"])
    if(image_number == "000000"):
      prev_img_name = file_name
      prev_panoptic_name = panoptic_file
    else:
      prev_img_number = '%06d' % (int(image_number) - 1)
      prev_img_name = os.path.join(os.path.dirname(file_name), prev_img_number + ".png")
      prev_panoptic_name = os.path.join(os.path.dirname(panoptic_file), prev_img_number + ".png")


    height, width = cv2.imread(file_name).shape[:2]
    
    for segment in ann["segments_info"]:
      segment.pop("area", None)
      cat_id = segment["category_id"]
      if cat_id == 255:
        continue
      isInstance = cat_id == 13 or cat_id == 11
      """
      if isInstance:
        if cat_id == 11:
          cat_id =0
        elif cat_id == 13:
          cat_id = 1
      elif cat_id == 255:
        continue
      elif cat_id == 12:
        cat_id = 11
      elif cat_id >= 14:
        cat_id = cat_id - 2
      """
      segment["category_id"] = cat_id
      segment["isthing"] = isInstance
    dataset_dicts.append({
      "file_name": file_name,
      "prev_file_name": prev_img_name,
      "image_id":image_id,
      "height":height,
      "width":width,
      "pan_seg_file_name":panoptic_file,
      "prev_pan_seg_file_name":prev_panoptic_name,
      "segments_info": ann["segments_info"]
    })
  return dataset_dicts

def get_kitti_dicts_json_q(json_file, image_root, panoptic_root):
  with open(json_file, "r") as f:
    dicts = json.load(f)
  anns = dicts["annotations"]
  dataset_dicts = []
  for ann in anns:
    image_id = ann["image_id"]
    file_name = os.path.join(image_root,ann["file_name"])
    height, width = cv2.imread(file_name).shape[:2]
    panoptic_file = os.path.join(panoptic_root,  ann["file_name"])
    for segment in ann["segments_info"]:
      segment.pop("area", None)
      cat_id = segment["category_id"]
      if cat_id == 255:
        continue
      isInstance = cat_id == 13 or cat_id == 11
      segment["category_id"] = cat_id
      segment["isthing"] = isInstance
    dataset_dicts.append({
      "file_name": file_name,
      "image_id":image_id,
      "height":height,
      "width":width,
      "pan_seg_file_name":panoptic_file,
      "segments_info": ann["segments_info"]
    })
    break

  return dataset_dicts


def load_kitti(overfit=False):
    if  overfit:
      #DatasetCatalog.register("d_val", lambda: get_kitti_dicts_json_q("panoptic_val_q.json", image_root, panoptic_root))
      #MetadataCatalog.get("d_val").set(panoptic_json = "panoptic_val_q.json")
      if os.path.isfile(panoptic_overfit_train_json):
        overfit_root = "overfit_data"
        panoptic_of_root = "overfit_data/panoptic_maps"
        DatasetCatalog.register("d_train", lambda: get_kitti_dicts_json(panoptic_overfit_train_json, overfit_root , panoptic_of_root ))
        DatasetCatalog.register("d_val", lambda: get_kitti_dicts_json(panoptic_overfit_val_json, overfit_root, panoptic_of_root))
      else:
        train_files = get_files_overfit("overfit_data/train")
        val_files = get_files_overfit("overfit_data/val")
        DatasetCatalog.register("d_val", lambda: get_kitti_dicts("overfit_data/val", "overfit_data/panoptic_maps/val", val_files))
        DatasetCatalog.register("d_train", lambda: get_kitti_dicts("overfit_data/train", "overfit_data/panoptic_maps/train", train_files))
            
    else:
      if os.path.isfile(panoptic_val_json):
            #load from the json file if it exists. 
          DatasetCatalog.register("d_val", lambda: get_kitti_dicts_json(panoptic_val_json, image_root, panoptic_root))
      else:
          val_files = get_files(val_root)
          DatasetCatalog.register("d_val", lambda: get_kitti_dicts(val_root, val_panoptic_root, val_files))
            
      
      if os.path.isfile(panoptic_train_json):
        DatasetCatalog.register("d_train", lambda: get_kitti_dicts_json(panoptic_train_json, image_root, panoptic_root))
      else:
        train_files = get_files(train_root)
        DatasetCatalog.register("d_train", lambda: get_kitti_dicts(train_root, train_panoptic_root, train_files))
          
    
    MetadataCatalog.get("d_val").set(stuff_classes=["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky","person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"])
    MetadataCatalog.get("d_val").set(thing_classes=["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky","person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"])
    MetadataCatalog.get("d_val").set(ignore_label=255)
    MetadataCatalog.get("d_val").set(thing_dataset_id_to_contiguous_id={11:11, 13:13})
    MetadataCatalog.get("d_val").set(stuff_dataset_id_to_contiguous_id={0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 12:12, 14:14, 15:15, 16:16, 17:17, 18:18})
    MetadataCatalog.get("d_val").set(label_divisor = 1000)
    

    MetadataCatalog.get("d_train").set(stuff_classes=["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky","person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"])
    MetadataCatalog.get("d_train").set(thing_classes=["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky","person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"])
    MetadataCatalog.get("d_train").set(ignore_label=255)
    MetadataCatalog.get("d_train").set(thing_dataset_id_to_contiguous_id={11:11, 13:13})
    MetadataCatalog.get("d_train").set(label_divisor = 1000)
    

    if overfit:
      MetadataCatalog.get("d_val").set(image_root = "overfit_data")
      MetadataCatalog.get("d_val").set(panoptic_root = "overfit_data/panoptic_maps")
      MetadataCatalog.get("d_train").set(image_root = "overfit_data")
      MetadataCatalog.get("d_train").set(panoptic_root = "overfit_data/panoptic_maps")
      convert_to_panoptic_json("d_val", "overfit_data",  panoptic_overfit_val_json)
      convert_to_panoptic_json("d_train", "overfit_data",  panoptic_overfit_train_json)
      MetadataCatalog.get("d_val").set(panoptic_json = "panoptic_overfit_val.json")
      MetadataCatalog.get("d_train").set(panoptic_json = "panoptic_overfit_train.json")
    else:
      MetadataCatalog.get("d_val").set(image_root = image_root)
      MetadataCatalog.get("d_val").set(panoptic_root = panoptic_root)
      convert_to_panoptic_json("d_val", image_root,  panoptic_val_json)
      MetadataCatalog.get("d_train").set(image_root = image_root)
      MetadataCatalog.get("d_train").set(panoptic_root = panoptic_root)
      MetadataCatalog.get("d_val").set(panoptic_json = "panoptic_val.json")
      convert_to_panoptic_json("d_train", image_root, panoptic_train_json)
      MetadataCatalog.get("d_train").set(panoptic_json = "panoptic_train.json")







    




def convert_to_panoptic_json(dataset_name, image_root, output_file):
    if os.path.isfile(output_file):
        #file already exists!
        return
    else:
        dataset_dicts = DatasetCatalog.get(dataset_name)
        out_dicts = {}
        out_dicts["annotations"] = []
        out_dicts["categories"] = categories
        for record in dataset_dicts:
            ret = {}
            ret["image_id"] = record["image_id"]
            file_name = record["file_name"]
            ret["file_name"] = file_name[len(image_root)+1:]
            

            panoptic_file = record["pan_seg_file_name"]
            panoptic_img = Image.open(panoptic_file)
            panoptic_img = np.asarray(panoptic_img)
            height, width = panoptic_img.shape[:-1]
            ids = np.zeros_like(panoptic_img)

            if isinstance(panoptic_img, np.ndarray) and len(panoptic_img.shape) == 3:
                if panoptic_img.dtype == np.uint8:
                    panoptic_img = panoptic_img.astype(np.int32)
                ids = panoptic_img[:, :, 0] + 256 * panoptic_img[:, :, 1] + 256 * 256 * panoptic_img[:, :, 2]
            else:
                ids = int(panoptic_img[0] + 256 * panoptic_img[1] + 256 * 256 * panoptic_img[2])


            segm_ids = panoptic_img[:,:,0]
            for dicts in record["segments_info"]:
                dicts.pop("isthing",None)
            record["segments_info"] = [dict([key, int(value)] if type(value) == np.int32 else [key,value] for key, value in dicts.items() ) for dicts in record["segments_info"]]
            segm_dicts = {el['id']: el for el in record['segments_info']}
            unique_ids, indices, counts = np.unique(ids, return_index=True, return_counts=True)

            for id, idx, count in zip(unique_ids, indices, counts):
                row = idx // width
                col = idx % width     
                segm_id = segm_ids[row][col]
                if segm_id == VOID:
                  continue
                segm_dicts[id]["category_id"]= int(segm_id)
                #segm_dicts[id]["id"] = int(id)

                isInstance = segm_id == 11 or segm_id == 13

                if id not in segm_dicts:
                    raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in data dict.'.format(record['image_id'], id))
                segm_dicts[id]["area"] = int(count)
                
                ret["segments_info"] = list(segm_dicts.values())
            out_dicts["annotations"].append(ret)
        json_object = json.dumps(out_dicts, indent=4)
        with open(output_file, "w") as out:
          out.write(json_object)












