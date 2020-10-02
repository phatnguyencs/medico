import pycocotools.mask as coco_mask
import json, funcy, shutil, os
import os.path as osp
import glob
import shutil
import cv2
import numpy as np
from skimage import measure

# ------------------------
# COCO HELPER

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)


def save_images(ls_img_ins, data_dir, save_dir):
    create_dir(save_dir)

    for img in ls_img_ins:
        # if img['file_name'] not in os.listdir(data_dir):
        #     continue
        source_img_dir = osp.join(data_dir, img['file_name'])
        target_img_dir = osp.join(save_dir, img['file_name'])
        shutil.copy(source_img_dir, target_img_dir)
    
    print(f"save {len(ls_img_ins)} images to {save_dir}")


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def create_dir(file_dir, is_print=False):
    try:
        os.mkdir(file_dir)
        if is_print:
            print("Directory " , file_dir ,  " Created ") 
    except FileExistsError:
        if is_print:
            print("Directory " , file_dir ,  " already exists")
        else:
            pass

def clean_images(total_dir, error_dirs: list):
    list_fail_images = []
    list_cleaned_images = []

    for e_dir in error_dirs:
        list_fail_images.extend(list(os.listdir(e_dir)))
    
    all_imgs = os.listdir(total_dir)

    for img in all_imgs:
        if len(list_fail_images) == 0:
            break

        if img in list_fail_images:
            list_fail_images.remove(img)
            continue
            
        list_cleaned_images.append(img)
    
    print(f"total: {len(all_imgs)}")
    print(f"cleaned: {len(list_cleaned_images)}")

    return list_cleaned_images

def get_cleaned_images(total_dir, list_error_dirs, save_dir):
    list_cleaned_images = clean_images(total_dir, list_error_dirs)
    
    for img_name in list_cleaned_images:
        shutil.copyfile(
            src = osp.join(total_dir, img_name),
            dst = osp.join(save_dir, img_name)
        )

def get_image_size_given_path(img_path: str):
    '''
    return image [H, W, C]
    '''
    img = cv2.imread(img_path)
    return img.shape

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def create_coco_mask_annotation(mask, bbox):
    '''
    Find contours around each sub-mask
    Args:
        mask: cv2 numpy gray image
    '''
    height, width = mask.shape[:2]
    # edged = cv2.Canny(mask, 30, 200)
    # contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
    dummy = np.zeros((height, width), dtype=np.uint)
    dummy[ymin:ymax+1, xmin:xmax+1] = 1
    mask *= dummy
    segmentations = binary_mask_to_polygon(mask/255)
    area = np.sum(mask/255.0)

    return segmentations, area
