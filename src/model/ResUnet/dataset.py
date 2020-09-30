"""" Modified version of https://github.com/jeffwen/road_building_extraction/blob/master/src/utils/data_utils.py """
from __future__ import print_function, division
from torch.utils.data import Dataset
from skimage import io
import glob
import os
import os.path as osp
import torch
from torchvision import transforms
import cv2
import pandas as pd 
import numpy as np
import random
from PIL import Image

class ImageDataset(Dataset):
    """Massachusetts Road and Building dataset"""

    def __init__(self, cfg, train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image paths
            train_valid_test (string): 'train', 'valid', or 'test'
            root_dir (string): 'mass_roads', 'mass_roads_crop', or 'mass_buildings'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.train = train
        self.train_img_path = osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.TRAIN_IMAGES)
        self.train_mask_path = osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.TRAIN_MASKS)
        
        self._load_csv_data(cfg)
        self.transform = transform
        self.image_size = cfg.MODEL.IMAGE_SIZE

    def _load_csv_data(self, cfg):
        if self.train:
            df = pd.read_csv(osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.TRAIN))
        else:
            df = pd.read_csv(osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.VAL))
        
        self.list_img = df['image'].tolist()
        self.mask_list = [osp.join(self.train_mask_path, f'{s}.jpg') for s in self.list_img]
        self.img_list = [osp.join(self.train_img_path, f'{s}.jpg') for s in self.list_img]

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        maskpath = self.mask_list[idx]
        imagepath = self.img_list[idx]
        # image_name = maskpath.split('/')[-1].split('.')[0]
        
        image = io.imread(imagepath) #[H, W, C]
        mask = io.imread(maskpath) #[H, W, 1]
        image = cv2.resize(image, dsize=self.image_size)#, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, dsize=self.image_size)#, interpolation=cv2.INTER_CUBIC)
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        sample = {
            "sat_img": Image.fromarray(image).convert("RGB"), 
            "map_img": Image.fromarray(gray_mask).convert('L'),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensorTarget(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sat_img, map_img = sample["sat_img"], sample["map_img"]
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return {
            "sat_img": transforms.functional.to_tensor(sat_img), # [C, H, W]
            "map_img": transforms.functional.to_tensor(map_img), #[1, H, W]
            # torch.from_numpy(map_img).unsqueeze(0).float().div(255),#[1, H, W]
        }  # unsqueeze for the channel dimension

class NormalizeTarget(transforms.Normalize):
    """Normalize a tensor and also return the target"""

    def __call__(self, sample):
        return {
            "sat_img": transforms.functional.normalize(
                sample["sat_img"], self.mean, self.std
            ),
            "map_img": sample["map_img"],
        }


# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

class Rotate(object):
    def __init__(self, degree=[-30, -15, 0, 15, 30]):
        self.degree = random.choice(degree)
    
    def __call__(self, sample):
        """
        Args:
            sample: Tensor of image and mask
        Returns:
            dictionary of result
        """
        return {
            'sat_img': transforms.functional.rotate(sample['sat_img'], self.degree),
            'map_img': transforms.functional.rotate(sample['map_img'], self.degree),
        }

class AdjustBrightness(object):
    def __init__(self, brightness_factor=[0.5, 0.75, 1., 1.25, 1.5]):
        self.brightness_factor = random.choice(brightness_factor)
    
    def __call__(self, sample):
        """
        Args:
            sample: Tensor of image and mask
        Returns:
            dictionary of result
        """
        return {
            'sat_img': transforms.functional.adjust_brightness(sample['sat_img'], self.brightness_factor),
            'map_img': sample['map_img']
        }

class AdjustGamma(object):
    def __init__(self, gamma=[0.5, 0.75, 1., 1.25, 1.5]):
        self.gamma = random.choice(gamma)
    
    def __call__(self, sample):
        """
        Args:
            sample: Tensor of image and mask
        Returns:
            dictionary of result
        """
        return {
            'sat_img': transforms.functional.adjust_gamma(sample['sat_img'], self.gamma),
            'map_img': sample['map_img'],
        }

class AdjustContrast(object):
    def __init__(self, contrast_factor=[0.5, 0.75, 1., 1.25, 1.5]):
        self.contrast_factor = random.choice(contrast_factor)
    
    def __call__(self, sample):
        """
        Args:
            sample: Tensor of image and mask
        Returns:
            dictionary of result
        """
        return {
            'sat_img': transforms.functional.adjust_contrast(sample['sat_img'], self.contrast_factor),
            'map_img': sample['map_img'],
        }
