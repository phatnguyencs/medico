import os 
import os.path as osp 
from glob import glob
import cv2 
import numpy as np
from PIL import Image

def resize_img(img_array, target_size):
    res = cv2.resize(img_array, dsize=target_size, interpolation=cv2.INTER_CUBIC)
    return res

def show_mask(img_array, mask_array):
    gray_mask = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)
    gray_mask[gray_mask > 0] = 1
    img_array[np.nonzero(gray_mask)] = (0,255,0)
    return img_array 

def test_resize(img_array, mask_array, target_size):
    img = resize_img(img_array, target_size)
    mask = resize_img(mask_array, target_size)

    return show_mask(img, mask)


def refine_patch(mask_array):
    HEIGHT, WIDTH = mask_array.shape[:2]
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]

    gray_mask = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)
    init_mask_ids = np.nonzero(gray_mask)

    visited = np.zeros(gray_mask.shape)

    root_path = []
    root_point = None

    def dfs(x, y):
        if visited[y, x] == 1:
            return []
        queue = []
        path = []
        start = (x, y)
        queue.append(start)
        path.append((y,x))
        visited[y, x] = 1
        while len(queue) > 0:
            point = queue.pop(0)
            
            for k in range(4):  
                next_x = point[0]+dx[k]
                next_y = point[1]+dy[k]
                if next_x < 0 or next_x >= WIDTH or next_y < 0 or next_y >= HEIGHT:
                    continue
                if visited[next_y, next_x] == 0 and gray_mask[next_y,next_x] > 0:
                    queue.append((next_x, next_y))
                    visited[next_y, next_x] = 1
                    path.append((next_y, next_x))


        return path

    max_dist = 0
    list_points = np.array(init_mask_ids).T 
    for point in list_points:
        path = dfs(point[1], point[0])
        if len(path) > max_dist:
            root_point = (point[0], point[1])
            root_path = path
            max_dist = len(path)
    
    
    refine_ids = np.array(root_path).T
    res = np.zeros(mask_array.shape)
    for point in root_path:
        res[point[0], point[1], :] = 255
    
    return res, refine_ids


def refine_mask(mask_array, list_bboxes: list):
    patches = []
    res = mask_array
    for box in list_bboxes:
        label, xmin, ymin, xmax, ymax = box.values()
        patch = mask_array[ymin:ymax+1, xmin:xmax+1, :]
        refined_patch, _ = refine_patch(patch)
        res[ymin:ymax+1, xmin:xmax+1, :] = refined_patch
    
    return res
