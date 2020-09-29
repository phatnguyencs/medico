import pandas as pd 
from sklearn.model_selection import train_test_split
import os 
import os.path as osp
import glob
import json
import shutil

DATA_DIR = './data' 
IMAGE_DIR = osp.join(DATA_DIR, 'images')
MASK_DIR = osp.join(DATA_DIR, 'masks')
DFS_MASK_DIR = osp.join(DATA_DIR, 'dfs_masks')
BOX_JSON = osp.join(DATA_DIR, 'kavsir_bboxes.json')

box_info = json.load(open(BOX_JSON,'r'))
print(f"n boxes: {len(box_info.keys())}")
list_train_img = os.listdir(IMAGE_DIR)
list_mask_img = os.listdir(MASK_DIR)
list_dfs_mask = os.listdir(DFS_MASK_DIR)

def remove_hidden_jpg(list_jpg: list, file_dir: str):
    c = 0
    for filename in list_jpg:
        if filename[0] == '.':
            c += 1
            os.remove(osp.join(file_dir, filename))

    print(f"removed {c} images from {file_dir}")




