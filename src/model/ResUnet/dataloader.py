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
        
        self.mask_list = glob.glob(
            os.path.join(self.train_mask_path, "*.jpg"), recursive=True
        )
        self.transform = transform
        self.image_size = cfg.MODEL.IMAGE_SIZE

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        maskpath = self.mask_list[idx]
        image_name = maskpath.split('/')[-1].split('.')[0]
        
        image = io.imread(os.path.join(self.train_img_path, f"{image_name}.jpg"))
        mask = io.imread(maskpath)
        
        image = cv2.resize(image, dsize=self.image_size)#, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, dsize=self.image_size)#, interpolation=cv2.INTER_CUBIC)
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        sample = {"sat_img": image, "map_img": gray_mask}

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
            "sat_img": transforms.functional.to_tensor(sat_img),
            "map_img": torch.from_numpy(map_img).unsqueeze(0).float().div(255),
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
