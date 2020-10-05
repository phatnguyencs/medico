import os 
import os.path as osp 
from glob import glob
import cv2 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib as mpl



def draw_mask(image, mask, thres=0.5, alpha=0.5):
    if len(mask.shape) == 3:
        mask = mask.squeeze(2)
    mask[mask < thres] = 0.0
    # if np.max(image) == 1.0:
    #     image *= 255.0
    image[np.nonzero(mask)] = image[np.nonzero(mask)]*alpha + (1-alpha)*np.array([0,255,0], dtype=np.float)
    return image


def visualize_result(images, gts, masks, savepaths, scores, cfg):
    '''
    Args:
        images, masks, gts: np.array in the shape of [batch, H, W, C]
    '''
    batch_size, h, w = images.shape[0], images.shape[1], images.shape[2]
    dpi = mpl.rcParams['figure.dpi']
    fig_height, fig_width = h/float(dpi), w/float(dpi)
    for i in range(batch_size):
        fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(fig_width, fig_height))
        print(axes)
        fig.subplots_adjust(wspace=0.3)
        fig.subplots_adjust(hspace=0.3)

        axes[0, 0].imshow(draw_mask(images[i], gts[i], thres=cfg.INFERENCE.MASK_THRES))
        axes[0, 0].set_title('groundtruth')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(draw_mask(images[i], masks[i], thres=cfg.INFERENCE.MASK_THRES))
        axes[0, 1].set_title('predicted')
        axes[0, 1].axis('off')

        title = ','.join([f"{k}: {scores[k][i]:.02f}" for k in scores.keys()])
        fig.text(0.5, 0.04, title, ha='center', va='center')
        plt.savefig(savepaths[i])
        plt.clf()
    

