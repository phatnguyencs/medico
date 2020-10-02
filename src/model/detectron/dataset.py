import detectron2
import numpy as np
import os, json, cv2, random
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from skimage import measure 
from pycocotools import mask

def convert_mask_image_to_polygon_format(mask):
    '''
    Args:
        mask: np.array of binary image
    Return:

    '''
    fortran_gt_map = np.asfortranarray(mask)
    encoded_gt = mask.encode(fortran_gt_map)
    gt_area = mask.area(encoded_gt)
    gt_bbox = mask.toBbox(encoded_gt)
    contours = measure.find_contours(mask, 0.5)
    pass

def get_dataset_dict(csv_dir: str, data_dir: str, ):
    pass

class DetectronDataset(object):
    def __init__(self, list_dataset=['train', 'val'], list_dataset_file=['']):
        '''
        args:
            list_dataset: list of datasets to register to Detectron2 Dataset Catalog
        '''
        self.list_dataset = list_dataset

