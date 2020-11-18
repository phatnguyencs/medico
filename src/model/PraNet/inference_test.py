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

def refine_result(images, masks, crf_model, cfg):
    # Convert to numpy:
    np_masks = masks.cpu().numpy().transpose(0, 2, 3, 1) # [batch, 1, H, W] --> [batch, H, W, 1]
    np_imgs = images.cpu().numpy().transpose(0, 2, 3, 1) # [batch, 1, H, W] --> [batch, H, W, 1]
    batch_size = np_masks.shape[0]
    
    for i in range(batch_size):
        refined_mask = apply_dcrf_model(np_imgs[i], np_masks[i], crf_model, n_steps=cfg.INFERENCE.CRF_STEP) 
        np_masks[i] = np.expand_dims(refined_mask, axis=2) # [H, W] --> [H, W, 1]
    
    return torch.Tensor(np_masks.transpose(0, 3, 1, 2)) # covnert back to [batch, 1, H, W]


def setup():
    args = get_parser()
    cfg = get_default_config()
    cfg.merge_from_file(args.config)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return args, cfg

def prepare_model(cfg, checkpoint_dir: str):
    model = MedicoNet(cfg)
    resume = checkpoint_dir
    ckpt = model.load_checkpoint(resume)
    print(F"loaded model checkpoint at epoch: {ckpt['epoch']}, best score: {ckpt['best_score']}")
    print(f"LOADED MODEL SUCCESSFULLY")

    model.to_device()
    model.eval()
    return model


def visualize_on_specific_folder(save_folder, img_folder, model, cfg):
    # Setup
    batch_size = cfg.INFERENCE.BATCH_SIZE

    # test_img_dir = osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.TEST_IMAGES)
    test_img_dir = osp.join(cfg.DATA.ROOT_DIR, img_folder)
    list_imgs = os.listdir(test_img_dir)
    
    vis_savedir = osp.join(cfg.INFERENCE.SAVE_DIR, save_folder)
    os.makedirs(vis_savedir, exist_ok=True)

    crf_model, crf_val_tracker = None, None
    if cfg.INFERENCE.CRF:
        crf_model = get_dcrf_model(cfg.MODEL.IMAGE_SIZE)
        crf_val_tracker = metrics.ValidationTracker()

    with torch.no_grad():
        for img_path in tqdm(list_imgs):
            if img_path[0] == '.':
                continue
            img_path = osp.join(test_img_dir, img_path)
            img_input, raw_shape = ImageDataset.prepare_image(img_path, cfg)
            img_input = img_input.cuda().unsqueeze(0)
            
            if cfg.INFERENCE.TTA:
                output = model.predict_mask_tta(img_input, raw_shape)
            else:
                output = model.predict_mask(img_input, raw_shape)
            
            # print(f"output shape: {output.shape}")
            output = thresholding_mask(output, cfg.INFERENCE.MASK_THRES)
            
            # print(output.shape)
            # print(img_input.shape)
            # break

            img_name = img_path.split('/')[-1]
            raw_img = cv2.imread(img_path)
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            save_path = osp.join(vis_savedir, img_name)
            mask = output.cpu().numpy()
            visualize_prediction(
                images = np.expand_dims(raw_img, axis=0),
                masks = np.expand_dims(mask, axis=0),
                savepaths=[save_path],
                cfg=cfg,
                raw_shape=raw_shape
            )

def main():
    args, cfg = setup()
    checkpoint_dir = cfg.CHECKPOINT_PATH
    model = prepare_model(cfg, checkpoint_dir)

    folders_to_test = ['test_images']
    folders_to_save = [f"visualize_{s}_{cfg.INFERENCE.MASK_THRES:.01f}" for s in folders_to_test]

    if cfg.INFERENCE.TTA:
        folders_to_save = [name + '_tta' for name in folders_to_save]

    for i in range(len(folders_to_save)):
        print(f"visualizing folder {folders_to_test[i]} ...")
        visualize_on_specific_folder(
            save_folder=folders_to_save[i], img_folder=folders_to_test[i],
            model=model,cfg=cfg
        )

if __name__ == "__main__":
    main()

