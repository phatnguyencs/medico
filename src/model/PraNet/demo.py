import warnings
warnings.filterwarnings("ignore")

import torch
import argparse
import os
import os.path as osp
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import glob
from model.PraNet.utils import (
    StructureLoss, BCEDiceLoss, clip_gradient, adjust_lr, AvgMeter, get_default_config, 
    free_gpu_memory, get_parser, MyWriter, TSA_StructureLoss
)
from model.PraNet.dataset import ImageDataset
from model.PraNet.network import MedicoNet
from model.PraNet.utils import metrics
from model.PraNet.utils.visualize import visualize_validation, visualize_prediction, thresholding_mask 

from model.postprocess.apply_crf import get_dcrf_model, apply_dcrf_model

def setup():
    args = get_parser()
    cfg = get_default_config()
    cfg.merge_from_file(args.config)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return args, cfg

def prepare_model(cfg, checkpoint_dir: str):
    model = MedicoNet(cfg)
    resume = checkpoint_dir
    model.load_checkpoint(resume)
    print(f"LOADED MODEL SUCCESSFULLY")

    model.to_device()
    model.eval()
    return model


if __name__ == "__main__":
    args, cfg = setup()

    img_dir = cfg.DATA.TEST_IMAGES
    save_dir = cfg.OUTPUT_DIR
    test_name = 'ckcbpwbcz27y50y5p99aud939.jpg'
    img_path = osp.join(cfg.DATA.ROOT_DIR, img_dir, test_name)
    print(f"inference on image {img_path}")

    checkpoint_dir = cfg.CHECKPOINT_PATH
    model = prepare_model(cfg, checkpoint_dir)
    model.to_device()
    model.set_eval()

    img_input, raw_shape = ImageDataset.prepare_image(img_path, cfg, high_boost=True)
    img_input = img_input.cuda().unsqueeze(0)
    if cfg.INFERENCE.TTA:
        output = model.predict_mask_tta(img_input, raw_shape)
    else:
        output = model.predict_mask(img_input, raw_shape)
            
    output = thresholding_mask(output, cfg.INFERENCE.MASK_THRES)
    
    raw_img = cv2.imread(img_path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    save_path = osp.join(save_dir, test_name)
    mask = output.cpu().numpy()
    visualize_prediction(
        images = np.expand_dims(raw_img, axis=0),
        masks = np.expand_dims(mask, axis=0),
        savepaths=[save_path],
        cfg=cfg,
        raw_shape=raw_shape
    )
