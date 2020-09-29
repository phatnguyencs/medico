from .utils import (
    resize_img, 
    show_mask,
    test_resize,
    refine_mask,
)

import json , os
from tqdm import tqdm
import cv2
import os.path as osp

BBOXES_JSON = ''
MASKS_DIR = ''
DFS_MASKS_DIR = ''

boxes_dict = json.load(open(BBOXES_JSON, 'r'))
mask_dir = MASKS_DIR
save_dir = DFS_MASKS_DIR

for mask in tqdm(os.listdir(mask_dir)):
    if '.jpg' not in mask: 
        continue
    mask_name = mask.split('.')[0]
    save_name = mask_name
    save_path = osp.join(save_dir, f"{mask_name}.jpg")
    cv_mask = cv2.imread(osp.join(mask_dir, mask))
    mask_boxes = boxes_dict[mask_name]['bbox']
    refined_mask = refine_mask(cv_mask, mask_boxes)
    assert refined_mask.shape == cv_mask.shape, f"Incompatible shape: before mask: {cv_mask.shape}, after mask: {refined_mask.shape}"
    cv2.imwrite(save_path, refined_mask)
    # cv2_imshow(refined_mask)
    # print(f"before mask: {cv_mask.shape}, after mask: {refined_mask.shape}")


