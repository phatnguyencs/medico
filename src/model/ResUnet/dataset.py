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
import albumentations as A


class ImageDataset(Dataset):
    """Massachusetts Road and Building dataset"""

    def __init__(self, cfg, img_path, mask_path, train=True):
        """
        Args:
            csv_file (string): Path to the csv file with image paths
            train_valid_test (string): 'train', 'valid', or 'test'
            root_dir (string): 'mass_roads', 'mass_roads_crop', or 'mass_buildings'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.is_train = train
        self.img_path = img_path
        self.mask_path = mask_path
        
        self.is_aug = cfg.TRAIN.AUGMENT
        self._load_csv_data(cfg)
        self.image_size = cfg.MODEL.IMAGE_SIZE

        self._setup_transform(cfg)

    def _load_csv_data(self, cfg):
        if self.is_train:
            df = pd.read_csv(osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.TRAIN))
        else:
            df = pd.read_csv(osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.VAL))
        
        self.list_img = df['image'].tolist()
        self.mask_list = [osp.join(self.mask_path, f'{s}.jpg') for s in self.list_img]
        self.img_list = [osp.join(self.img_path, f'{s}.jpg') for s in self.list_img]

        print(f"Created dataset with {len(self.img_list)} images")

    # Hard code augmentation for training step
    def _setup_transform(self, cfg):
        self.resize_transform = transforms.Resize(cfg.MODEL.IMAGE_SIZE, Image.NEAREST)
        # image_mask_trans = [
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomVerticalFlip(p=0.5),
        #     transforms.RandomAffine(degrees=45, scale=(0.8, 1.2), shear=(-2,2))
        # ]
        # self.img_mask_transform = transforms.Compose(image_mask_trans) 
        self.img_mask_transform = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30),
            A.Flip(),
            A.Transpose(),
            A.ElasticTransform(),
            A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0,p=0.5),
    #         A.OneOf([
    #                 A.RandomCrop(height=size_crop,width=size_crop,p=0.5),  
    #                 A.CenterCrop(height=size_crop,width=size_crop,p=0.5)
    #             ]),
            ],p=0.9)
        self.img_pixel_transform = A.Compose([
            A.GaussNoise(),
            A.Blur(blur_limit=3),
            A.RandomBrightnessContrast(),
            A.HueSaturationValue(hue_shift_limit=3,sat_shift_limit=20,val_shift_limit=3 ,p=0.5),
        ],p=0.5)
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(mean=cfg.TRAIN.NORMALIZE_MEAN, std=cfg.TRAIN.NORMALIZE_STD)

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        maskpath = self.mask_list[idx]
        imagepath = self.img_list[idx]
        
        image = Image.open(imagepath).convert("RGB") #[W, H, C]
        mask = Image.open(maskpath)# .convert('L')
        original_width, original_height = image.size

        # augment when training only
        # if self.is_aug and self.is_train and idx % 2 == 0:
        if self.is_aug and self.is_train:
            transformed = self.img_mask_transform(image=np.array(image), mask=np.array(mask))
            image = transformed['image']
            mask = transformed['mask']
            image = self.img_pixel_transform(image=image)['image']

            image, mask = Image.fromarray(image), Image.fromarray(mask)

        image, mask = self.resize_transform(image), self.resize_transform(mask)
        image, mask = self.to_tensor_transform(image), self.to_tensor_transform(mask)[0,:,:]
        image = self.normalize_transform(image)

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
        image = Image.open(img_path).convert("RGB")
        raw_shape = {'height': image.size[1], 'width': image.size[0]}
        image = transforms.Resize(cfg.MODEL.IMAGE_SIZE, Image.NEAREST)(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=cfg.TRAIN.NORMALIZE_MEAN, std=cfg.TRAIN.NORMALIZE_STD)(image)

        raw_image = io.imread(img_path)

        return image, raw_shape

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