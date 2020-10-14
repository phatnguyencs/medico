from PIL import Image
import os
import os.path as osp
import json
import cv2
from tqdm import tqdm
import pandas as pd

from dataset.utils import create_coco_mask_annotation, get_image_size_given_path

class COCOconverter(object):
    def __init__(self, mask_dir: str, img_dir: str, box_json: str, csv_path=None):
        self.mask_dir=mask_dir
        self.img_dir=img_dir
        self.box_info = json.load(open(box_json, 'r'))
        self.default_dataset_id = 88
        self.default_color = '#7ad54d'
        self.list_imgs = []

        if csv_path is not None:
            self.list_imgs = pd.read_csv(csv_path)['image'].to_list()


    def _create_image_item(self, category_ids: list, img_name: str, img_id:str, width: int, height: int):
        res = {}
        res['path'] = osp.join(self.img_dir, f'{img_name}.jpg')
        res['height'], res['width'] = width, height
        res['category_ids'] = category_ids
        res['annotated'] = True
        res['annotating'] = []
        res['dataset_id'] = self.default_dataset_id
        res['file_name'] = res['path'].split('/')[-1]
        res['metadata'] = {}
        res['id'] = img_id
        
        return res

    def _create_annot_item(self, bbox, img_name, annot_id, image_id, category_id, width, height, iscrowd=False, color=None, creator='system'):
        '''
        Args:
            bbox: XYXY format
        Return:

        '''
        if color is None:
            color = self.default_color
        mask_path = osp.join(self.mask_dir, f"{img_name}.jpg")
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        segmentations, area = create_coco_mask_annotation(mask, bbox)
        
        bbox = [bbox['xmin'], bbox['ymin'], bbox['xmax']-bbox['xmin'], bbox['ymax']-bbox['ymin']]
        res = {
            'area': area,
            'bbox': bbox,
            'category_id': category_id,
            'color': color,
            'creator': creator,
            'dataset_id': self.default_dataset_id,
            'height': height,
            'id': annot_id,
            'image_id': image_id,
            'iscrowd': iscrowd,
            'metadata': {},
            'segmentation': segmentations,
            'width': width,
        }
        return res
        
    def _create_cat_item(self, cat_name, cat_id, supercategory="", color=None, creator="system", metadata={}):
        if color is None:
            color = self.default_color
        return {
            'color': color,
            'create': creator,
            'id': cat_id,
            'name': cat_name,
            'supercategory': supercategory,
            'metadata': {},
        }

    def _filter_with_csv(self, img_name):
        if len(self.list_imgs) == 0:
            return True
        if img_name in self.list_imgs:
            return True
        return False

    def process(self):
        categories, images, annotations = [], [], []
        category_map = {}
        category_id = 0
        image_id = 0
        annot_id = 0

        for img_name, value in tqdm(self.box_info.items()):
            if self._filter_with_csv(img_name) == False:
                continue

            img_id = str(image_id)
            image_id += 1
            img_cat_ids = []
            img_path = osp.join(self.img_dir, f'{img_name}.jpg')
            W, H, C = get_image_size_given_path(img_path)

            for box in value['bbox']:
                if category_map.get(box['label']) is None:
                    category_map[box['label']] = category_id
                    cat_item = self._create_cat_item(cat_name = box['label'], cat_id = category_id)
                    category_id += 1
                    categories.append(cat_item)


                annot_item = self._create_annot_item(box, img_name, annot_id, img_id, category_map[box['label']], width=W, height=H)    
                annotations.append(annot_item)
                annot_id += 1
                img_cat_ids.append(category_map[box['label']])

            img_cat_ids = list(set(img_cat_ids))
            img_item = self._create_image_item(category_ids=img_cat_ids, img_name=img_name, img_id=img_id, width=W, height=H)
            images.append(img_item)
        
        print(f"converted {image_id} samples to COCO format") 
        return {
            'annotations': annotations,
            'categories': categories,
            'images': images,
            'info': None,
            'licenses': None,
        }



    
