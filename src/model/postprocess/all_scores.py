import numpy as np
import cv2 
import os 
import os.path as osp
from tqdm import tqdm

PRED_DIR = 'result/fbrs/val4_mask' # dir to predicted mask (.npy file)
LABEL_DIR = 'data/dfs_masks' # dir to groundtruth mask (.npy file)

result_dir = ''
smooth = 1.0
eps = 1e-5

def thresholding(mat, thres=0.5):
    res = mat.copy()
    res[res>thres] = 1.0
    res[res<thres] = 0.0

    return res

def calculate_f_score(precision, recall, base=2):
    return (1 + base**2)*(precision*recall)/((base**2 * precision) + recall)


def eval_all_score(pred, label, thres = 0.5):
    H, W = pred.shape[:2]
    pred = pred.reshape(H*W)
    label = label.reshape(H*W)
    pred = thresholding(pred, thres)
    label = thresholding(label, thres)

    intersection = np.sum(pred*label)
    union = np.sum(pred) + np.sum(label) - intersection
    iou = intersection/(union + eps)
    dice_coeff = (2.0*intersection + smooth)/(np.sum(pred) + np.sum(label) + smooth)
    precision = intersection/(np.sum(label) + eps)
    recall = intersection/(np.sum(pred) + eps)
    f2 = calculate_f_score(precision, recall, 2)

    return {
        'dice_coeff': dice_coeff,
        'IoU': iou,
        'precision': precision,
        'recall': recall,
        'f2': f2 
    }

def load_mask_from_image(img_path):
    mask = cv2.imread(img_path)
    mask = mask[:,:,0]/255.0
    return mask

if __name__ == "__main__":
    total_score = {
        'dice_coeff': 0,
        'IoU': 0,
        'precision': 0,
        'recall': 0,
        'f2': 0, 
    }
    list_img = os.listdir(PRED_DIR)
    n_img = len(list_img)
    for img_name in tqdm(list_img):
        img_path = img_name
        mask_path = osp.join(LABEL_DIR, img_name.replace('.npy', '.jpg'))

        label = load_mask_from_image(mask_path) # [H, W]
        pred = np.load(osp.join(PRED_DIR, img_name)) #[H, W]

        scores = eval_all_score(pred, label, 0.5)
        for k in scores.keys():
            total_score[k] += scores[k]
    
    for k in total_score.keys():
        total_score[k] /= n_img

    print(total_score)
    
