import pandas as pd
import cv2 
import os
import os.path as osp

data_dir = 'data'

def print_image_given_csv(csv_file: str, save_dir: str, draw_box=False):
    df = pd.read_csv(csv_file)
    for i, row in df.iterrows():
        img_name = row['image']
        nbox = int(row['nbox'])



    
