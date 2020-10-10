import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from numpy import save
import cv2, glob, tqdm

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch, DefaultPredictor
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

from model.detectron.centermask.evaluation import COCOEvaluator
from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.data.dataset_mapper import DatasetMapper
from model.detectron.centermask.config import get_cfg
from model.detectron.centermask.checkpoint import AdetCheckpointer

from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances

def preprocess_cfg(cfg):
    model_weight_ckpt = cfg.MODEL.WEIGHTS
    
    tmps = model_weight_ckpt.split('/')
    ckpt_name = tmps[-1]
    ckpt_dir = '/'.join(tmps[:-1])
    
    if ckpt_name not in os.listdir(ckpt_dir):
        model_weight_ckpt = model_weight_ckpt.replace(ckpt_name, 'model_final.pth')

    cfg.MODEL.WEIGHTS = model_weight_ckpt
    return cfg

def setup(args, inference=False):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg = preprocess_cfg(cfg)
    print(cfg.MODEL.WEIGHTS)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.FCOS.NUM_CLASSES = 5
    if inference:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.3]

    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def get_top_score(outputs, score_thres=0.5):
    tmp = outputs._fields['scores'] >= score_thres    
    top_k = torch.sum(tmp).item()
    
    for k in outputs._fields.keys():
        if torch.is_tensor(outputs._fields[k]):
            outputs._fields[k] = outputs._fields[k].narrow(0, 0, top_k)
        else:
            outputs._fields[k].tensor = outputs._fields[k].tensor.narrow(0, 0, top_k)
    
    return outputs

def get_top_output(outputs, top_k=3):
    '''
    outputs must be in "cpu" mode
    '''
    for k in outputs._fields.keys():
        if torch.is_tensor(outputs._fields[k]):
            outputs._fields[k] = outputs._fields[k].narrow(0, 0, top_k)
        else:
            outputs._fields[k].tensor = outputs._fields[k].tensor.narrow(0, 0, top_k)
    
    return outputs

def evaluate_and_save_fig(predictor, thres_type="topK", val_set="firevysor_val", thres=3, vis_res_dir=None):
    if vis_res_dir == None:
        vis_res_dir = os.path.join(cfg.OUTPUT_DIR, f"visualize_{thres_type}_{thres}")
    try:
        os.mkdir(vis_res_dir)
        print("Directory " , vis_res_dir ,  " Created ") 
    except FileExistsError:
        print("Directory " , vis_res_dir ,  " already exists")

    
    firevysor_metadata = MetadataCatalog.get(val_set)

    val_imgs_dir = firevysor_metadata.get('image_root')
    list_img_dir = glob.glob(val_imgs_dir+'/*')
    
    for img_dir in tqdm.tqdm(list_img_dir):
        img = cv2.imread(img_dir)
        outputs = predictor(img)
        
        v = Visualizer(img[:, :, ::-1],
                   metadata=firevysor_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        outputs_cpu = outputs["instances"].to("cpu")
        
        if thres_type == "topK":
            outputs_topk = get_top_output(outputs_cpu, thres)
        else:
            outputs_topk = get_top_score(outputs_cpu, thres)

        out = v.draw_instance_predictions(outputs_topk)
        image_name = img_dir.split('/')[-1]
        cv2.imwrite(os.path.join(vis_res_dir, image_name), out.get_image()[:, :, ::-1])

if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    # register_coco_instances("firevysor_train", {}, "datasets/coco/annotations/train_annot_newcat.json", "datasets/coco/train_newcat")
    # register_coco_instances("firevysor_val", {}, "data/Split_CleanedImage/val_annot.json", "data/Split_CleanedImage/val")
    
    register_coco_instances("firevysor_val", {}, "data/Split_CleanedImage/val_annot.json", "data/Split_CleanedImage/val")
    register_coco_instances("hardcases_val", {}, "data/annotations/hard_cases.json", "data/hard_cases")

    MetadataCatalog.get('firevysor_val').thing_classes = ['linear', 'spider', 'shadow', 'unsolder', 'aal']
    MetadataCatalog.get('hardcases_val').thing_classes = ['linear', 'spider', 'shadow', 'unsolder', 'aal']

    cfg = setup(args, inference=True)
    predictor = DefaultPredictor(cfg)
    # INFERENCE AND VISUALIZE
    save_dir = os.path.join(cfg.OUTPUT_DIR, "hardcases_topScore_0.5")
    evaluate_and_save_fig(predictor, thres_type="topScore", thres=0.5, vis_res_dir=save_dir, val_set="hardcases_val")

    save_dir = os.path.join(cfg.OUTPUT_DIR, "val_topScore_0.5")
    evaluate_and_save_fig(predictor, thres_type="topScore", thres=0.5, vis_res_dir=save_dir, val_set="firevysor_val")

