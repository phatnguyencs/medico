from dataset.create_coco import COCOconverter
import os, json
import os.path as osp
import pickle

DATA_DIR = 'data'

if __name__ == "__main__":
    image_dir = osp.join(DATA_DIR, 'train_images')
    mask_dir = osp.join(DATA_DIR, 'dfs_masks')
    boxes_json = osp.join(DATA_DIR, 'kavsir_bboxes.json')
    csv = osp.join(DATA_DIR, 'train4.csv')
    save_name = 'train4'
    converter = COCOconverter(mask_dir, image_dir, boxes_json, csv)
    result = converter.process()

    with open(osp.join(DATA_DIR, f'{save_name}.pkl'), 'wb') as f:
        pickle.dump(result, f)

    
    with open(osp.join(DATA_DIR, f'{save_name}.json'), 'w') as f:
        json.dump(result, f, indent=4)
    
