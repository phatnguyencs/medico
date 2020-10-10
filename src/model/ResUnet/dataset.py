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

    def __init__(self, cfg, img_path, mask_path, train=True, image_transform=None, label_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image paths
            train_valid_test (string): 'train', 'valid', or 'test'
            root_dir (string): 'mass_roads', 'mass_roads_crop', or 'mass_buildings'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.train = train
        self.img_path = img_path
        self.mask_path = mask_path
        
        self._load_csv_data(cfg)
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.image_size = cfg.MODEL.IMAGE_SIZE

    def _load_csv_data(self, cfg):
        if self.train:
            df = pd.read_csv(osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.TRAIN))
        else:
            df = pd.read_csv(osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.VAL))
        
        self.list_img = df['image'].tolist()
        self.mask_list = [osp.join(self.mask_path, f'{s}.jpg') for s in self.list_img]
        self.img_list = [osp.join(self.img_path, f'{s}.jpg') for s in self.list_img]

        print(f"Created dataset with {len(self.img_list)} images")

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        maskpath = self.mask_list[idx]
        imagepath = self.img_list[idx]
        
        image = Image.open(imagepath).convert("RGB") #[W, H, C]
        mask = Image.open(maskpath)# .convert('L')
        original_width, original_height = image.size

        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            mask = self.label_transform(mask)[0,:,:]

        sample = {
            "sat_img": image,
            "map_img": mask,
        }

        sample['image_path'] = imagepath
        sample['raw_shape'] = {
            'width': original_width,
            'height': original_height,
        }

        return sample

    @staticmethod
    def prepare_image(img_path, cfg):
        '''
            Prepare an iamge ready to feed into ResUnet++ model
        '''
        image = io.imread(img_path)
        raw_shape = {'height': image.shape[0], 'width': image.shape[1]}
        image = cv2.resize(image, dsize=cfg.MODEL.IMAGE_SIZE)
        image = Image.fromarray(image).convert("RGB")
        return transforms.functional.to_tensor(image), raw_shape

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
            "sat_img": transforms.functional.normalize(sample["sat_img"], self.mean, self.std),
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
    def __init__(self, degree=list(range(0, 90, 10))):
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

def create_dataset(cfg, mode='train', transforms=None):
    pass