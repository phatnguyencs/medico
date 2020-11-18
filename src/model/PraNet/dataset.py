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

    def __init__(self, cfg, img_path, mask_path, train=True, csv_file=None):
        """
        Args:
            csv_file (string): Path to the csv file with image paths
            train_valid_test (string): 'train', 'valid', or 'test'
        """
        self.is_train = train
        self.img_path = img_path
        self.mask_path = mask_path
        
        self.is_aug = cfg.TRAIN.AUGMENT
        self.load_csv_data(cfg, csv_file)
        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.size_crop = 448
        self._setup_transform(cfg)

    def load_csv_data(self, cfg, csv_file: str = None):
        if csv_file is None:
            self.img_list = glob.glob(self.img_path+'/*')
            self.mask_list = glob.glob(self.mask_path+'/*')
        else:
            df = pd.read_csv(osp.join(cfg.DATA.ROOT_DIR, csv_file))
            self.list_img = df['image'].tolist()
            self.mask_list = [osp.join(self.mask_path, f'{s}.jpg') for s in self.list_img]
            self.img_list = [osp.join(self.img_path, f'{s}.jpg') for s in self.list_img]

        print(f"Created dataset with {len(self.img_list)} images")

    # Hard code augmentation for training step
    def _setup_transform(self, cfg):
        # Albumentation example: https://albumentations.readthedocs.io/en/latest/examples.html
        self.img_mask_transform = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=175, p=0.9, border_mode=cv2.BORDER_CONSTANT),
            A.Flip(),
            # A.Transpose(),
            # A.OneOf([
            #     A.ElasticTransform(),
            #     A.OpticalDistortion(),
            #     A.GridDistortion(),
            #     A.IAAPiecewiseAffine(),
            # ]),
            # A.OneOf([
            #         A.RandomCrop(height=self.size_crop,width=self.size_crop,p=0.5),  
            #         A.CenterCrop(height=self.size_crop,width=self.size_crop,p=0.5)
            # ]),            
            # A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0,p=0.5),
            ],p=0.9)

        self.img_pixel_transform = A.Compose([
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.IAASharpen(),
                A.IAAEmboss(),
                # A.RandomBrightnessContrast(),            
            ], p=0.3),
            A.HueSaturationValue(hue_shift_limit=3,sat_shift_limit=20,val_shift_limit=3,p=0.2),
        ],p=0.5)
        # Torch transform
        self.resize_transform = transforms.Resize(cfg.MODEL.IMAGE_SIZE, Image.NEAREST)
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(mean=cfg.TRAIN.NORMALIZE_MEAN, std=cfg.TRAIN.NORMALIZE_STD)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # maskpath = self.mask_list[idx]
        imagepath = self.img_list[idx]
        image_name = imagepath.split('/')[-1]
        maskpath = osp.join(self.mask_path, image_name)

        image = Image.open(imagepath).convert("RGB") #[W, H, C]
        mask = Image.open(maskpath)# .convert('L')
        original_width, original_height = image.size

        # augment when training only
        # if self.is_aug and self.is_train and idx % 2 == 0:
        if self.is_aug and self.is_train:
            if original_width < self.size_crop or original_height < self.size_crop:
                new_size = int(self.size_crop*1.2)
                image=image.resize((new_size, new_size))
                mask=mask.resize((new_size, new_size))

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
    def prepare_image(img_path, cfg, high_boost=False):
        '''
            Prepare an image ready to feed into PraNet model
        '''
        if high_boost:
            image = cv2.imread(img_path)
            blur_img = cv2.GaussianBlur(image, (7,7), 0)
            hb_img = cv2.addWeighted(image, 4, blur_img, -3, 0)
            image = Image.fromarray(hb_img).convert("RGB")
        
        else:
            image = Image.open(img_path).convert("RGB")
        
        raw_shape = {'height': image.size[1], 'width': image.size[0]}
        image = transforms.Resize(cfg.MODEL.IMAGE_SIZE, Image.NEAREST)(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=cfg.TRAIN.NORMALIZE_MEAN, std=cfg.TRAIN.NORMALIZE_STD)(image)

        return image, raw_shape
