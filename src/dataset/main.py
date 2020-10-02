from dataset.create_coco import COCOconverter
import os, json
import os.path as osp
import pickle

DATA_DIR = 'data'


if __name__ == "__main__":
    image_dir = osp.join(DATA_DIR, 'images')
    mask_dir = osp.join(DATA_DIR, 'dfs_masks')
    boxes_json = osp.join(DATA_DIR, 'kavsir_bboxes.json')

    converter = COCOconverter(mask_dir, image_dir, boxes_json)
    result = converter.process()

    with open(osp.join(DATA_DIR, 'coco_annotation.pkl'), 'wb') as f:
        pickle.dump(result, f)

    
    with open(osp.join(DATA_DIR, 'coco_annotation.json'), 'w') as f:
        json.dump(result, f, indent=4)
    
