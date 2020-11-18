import warnings
warnings.filterwarnings("ignore")

import torch
import argparse
import os
import os.path as osp
import cv2
import numpy as np
import time 
from tqdm import tqdm
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# Import model
from model.PraNet.utils import (
    StructureLoss, BCEDiceLoss, clip_gradient, adjust_lr, AvgMeter, get_default_config, 
    free_gpu_memory, get_parser, MyWriter, TSA_StructureLoss
)
from model.PraNet.dataset import ImageDataset
from model.PraNet.network import MedicoNet
from model.PraNet.utils.visualize import visualize_validation, visualize_prediction, thresholding_mask 

# Set path to test dataset

TEST_DATASET_PATH = "data/test_images"
CHECKPOINT_PATH = "result/PraNet/submission/best_model.pt"
MASK_PATH = "result/PraNet/submission/masks"
IMAGE_SIZE = (352, 352)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BACKBONE = "resnet50"

MASK_THRES = 0.5
os.makedirs(MASK_PATH, exist_ok=True)

# ---------------------- HELPER FUNCTIONS -----------------------
def prepare_model(cfg, checkpoint_dir: str):
    model = MedicoNet(cfg)
    resume = checkpoint_dir
    ckpt = model.load_checkpoint(resume)
    print(F"loaded model checkpoint at epoch: {ckpt['epoch']}, best score: {ckpt['best_score']}")
    print(f"LOADED MODEL SUCCESSFULLY")

    model.to_device()
    model.set_eval()
    return model

# ---------------------- INFERENCE -----------------------
# Load Keras model
cfg = get_default_config()
cfg.MODEL.BACKBONE = BACKBONE
model = prepare_model(cfg, CHECKPOINT_PATH)

image_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, Image.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

time_taken = []
for image_name in tqdm(os.listdir(TEST_DATASET_PATH)):
    # Load the test image
    image_path = os.path.join(TEST_DATASET_PATH, image_name)
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    outH, outW = IMAGE_SIZE

    image = image_transforms(image)
    image = image.cuda().unsqueeze(0)
    
    # Start time
    start_time = time.time()

    mask = model.predict_mask(
        image = image,
        raw_shape = {'height': outH, 'width': outW},
        is_numpy = False
    )

    # End timer
    end_time = time.time() - start_time
    
    
    time_taken.append(end_time)
    # print("{} - {:.10f}".format(image_name, end_time))

    # mask = nn.Upsample(size=(H, W), mode='bilinear', align_corners=False)(mask)
    # mask = mask.sigmoid().squeeze()
    # mask = mask.data.cpu().numpy()
    # mask = (mask - mask.min())/(mask.max()-mask.min()+1e-8)
    
    mask = mask.cpu().numpy()
    mask = cv2.resize(mask, dsize=(W, H))

    cv_img = cv2.imread(image_path)
    # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    
    thres = MASK_THRES*mask.max()
    mask[mask >= thres] = 1.0
    mask[mask < thres] = 0.0

    alpha=0.5
    # print(cv_img.shape)
    # print(mask.shape)
    cv_img[np.nonzero(mask)] = cv_img[np.nonzero(mask)]*alpha + np.array([224, 0, 0], dtype=np.float)*(1-alpha)
    
    mask = mask.astype(np.float32)
    mask = mask * 255.0

    mask_path = os.path.join(MASK_PATH, image_name)
    # cv2.imwrite(mask_path, mask)
    cv2.imwrite(mask_path, cv_img)

mean_time_taken = np.mean(time_taken)
mean_fps = 1/mean_time_taken
print("Mean FPS: ", mean_fps)
