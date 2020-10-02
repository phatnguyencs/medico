from PIL import Image
import os
import os.path as osp
import json
import cv2
from utils import create_coco_mask_annotation, get_image_size_given_path

class COCOconverter(object):
    def __init__(self, mask_dir: str, img_dir: str, box_json: str):
        self.mask_dir=mask_dir
        self.img_dir=img_dir
        self.box_info = json.load(open(box_json), 'r')
        self.default_dataset_id = 88

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
            color = "7ad54d"
        mask_path = osp.join(self.mask_dir, f"{img_name}.jpg")
        mask = cv2.imread(mask_path)
        segmentations, area = create_coco_mask_annotation(mask, bbox)
        
        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
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
            'segmentation': [segmentations],
            'width': width,
        }
        return res
        
    def _create_cat_item(self, cat_name, cat_id, supercategory="", color=None, creator="system", metadata={}):
        if color is None:
            color = "7ad54d"
        return {
            'color': color,
            'create': creator,
            'id': cat_id,
            'name': cat_name,
            'supercategory': supercategory,
            'metadata': {},
        }

    def _create_categories(self):
        categories, images, annotations = [], [], []
        category_map = {}
        category_id = 0
        image_id = 0
        annot_id = 0
        for img_name, value in self.box_info.items():
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


                annot_item = self._create_annot_item(box, img_name, annot_id, img_id, category_map[box['label']])    
                annotations.append(annot_item)
                annot_id += 1
                img_cat_ids.append(category_map[box['label']])

            img_cat_ids = list(set(img_cat_ids))
            img_item = self._create_image_item(category_ids=img_cat_ids, img_name=img_name, img_id=img_id)
            images.append(img_item)
        
        return {
            'annotations': annotations,
            'categories': categories,
            'images': images,
            'info': None,
            'licenses': None,
        }



    
