import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import glob
import dataset
from model.ResUnet.utils import metrics
from model.ResUnet.core.res_unet import ResUnet
from model.ResUnet.core.res_unet_plus import ResUnetPlusPlus
import torch
import argparse
import os
import os.path as osp
import cv2
import numpy as np
from model.ResUnet.utils import (
    get_parser,get_default_config,BCEDiceLoss,
    MetricTracker,jaccard_index,dice_coeff,
    MyWriter,
)
from model.ResUnet.init_config import setup
from model.ResUnet.utils.visualize import visualize_validation, visualize_prediction
from model.ResUnet.dataset import ImageDataset

def prepare_model(checkpoint_dir: str):
    model = ResUnetPlusPlus(3).cuda()

    resume = osp.join(checkpoint_dir, 'best_model.pt')
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
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
            
            output = model(img_input)

            img_name = img_path.split('/')[-1]
            raw_img = cv2.imread(img_path)
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            save_path = osp.join(vis_savedir, img_name)
            visualize_prediction(
                images = np.expand_dims(raw_img, axis=0),
                masks = output.cpu().permute(0, 2, 3, 1).numpy(),
                savepaths=[save_path],
                cfg=cfg,
                raw_shape=raw_shape
            )

def main():
    args, cfg = setup()
    checkpoint_dir = cfg.CHECKPOINT_PATH
    model = prepare_model(checkpoint_dir)

    folders_to_test = ['test_easy', 'test_hard', 'test_images']
    folders_to_save = [f"visualize_{s}_{cfg.INFERENCE.MASK_THRES}" for s in folders_to_test]

    for i in range(len(folders_to_save)):
        print(f"visualize folder {folders_to_test[i]}")
        visualize_on_specific_folder(
            save_folder=folders_to_save[i],
            img_folder=folders_to_test[i],
            model=model,
            cfg=cfg
        )


if __name__ == "__main__":
    main()