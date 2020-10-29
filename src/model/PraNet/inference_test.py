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


def setup():
    args = get_parser()
    cfg = get_default_config()
    cfg.merge_from_file(args.config)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return args, cfg

def prepare_model(cfg, checkpoint_dir: str):
    model = MedicoNet(cfg)
    if 'best_model.pt' not in checkpoint_dir:
        resume = osp.join(checkpoint_dir, 'best_model.pt')
    else:
        resume = checkpoint_dir

    model.load_checkpoint(resume)
    model.to_device()
    print(f"LOADED MODEL SUCCESSFULLY")

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
    with torch.no_grad():
        for img_path in tqdm(list_imgs):
            if img_path[0] == '.':
                continue
            img_path = osp.join(test_img_dir, img_path)
            img_input, raw_shape = ImageDataset.prepare_image(img_path, cfg)
            img_input = img_input.cuda().unsqueeze(0)
            
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

    folders_to_test = ['test_easy', 'test_hard', 'test_images']
    folders_to_save = [f"visualize_{s}_{cfg.INFERENCE.MASK_THRES:.01f}" for s in folders_to_test]

    for i in range(len(folders_to_save)):
        print(f"visualize folder {folders_to_test[i]}")
        visualize_on_specific_folder(
            save_folder=folders_to_save[i], img_folder=folders_to_test[i],
            model=model,cfg=cfg
        )

if __name__ == "__main__":
    main()

