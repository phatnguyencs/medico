import os 
import os.path as osp 
from glob import glob
import cv2 
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib as mpl
from torchvision import transforms

# warnings.simplefilter("ignore", (UserWarning, FutureWarning))

def draw_mask(image, mask, thres=0.5, alpha=0.5, raw_shape=None):
    if len(mask.shape) == 3 :
        mask = mask.squeeze(2)
    mask[mask < thres] = 0.0

    if raw_shape is not None:
        # image = cv2.resize(image, dsize=(raw_shape['width'], raw_shape['height']))
        np2pil = transforms.ToPILImage()
        pil_mask = np2pil(mask)
        pil_mask = transforms.Resize((raw_shape['height'], raw_shape['width']), Image.NEAREST)(pil_mask)
        mask = np.array(pil_mask)
        
        # mask = cv2.resize(mask, dsize=(raw_shape['width'], raw_shape['height']))
    
    image = image.copy()

    image[np.nonzero(mask)] = image[np.nonzero(mask)]*alpha + (1-alpha)*np.array([0,0,255], dtype=np.float)
    return image

def visualize_prediction(images, masks, savepaths, cfg, raw_shape=None):
    batch_size, h, w = images.shape[0], images.shape[1], images.shape[2]
    dpi = mpl.rcParams['figure.dpi']

    for i in range(batch_size):
        if raw_shape is not None:
            h, w  = raw_shape['height'], raw_shape['width']
        
        fig_height, fig_width = h/float(dpi), w/float(dpi)

        fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(2*fig_width, 2*fig_height))
        fig.subplots_adjust(wspace=0.3)
        fig.subplots_adjust(hspace=0.3)

        axes[0, 0].imshow(images[i])
        axes[0, 0].set_title('input')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(draw_mask(images[i], masks[i], thres=cfg.INFERENCE.MASK_THRES, raw_shape={'height': h, 'width': w}))
        axes[0, 1].set_title('predicted')
        axes[0, 1].axis('off')

        plt.savefig(savepaths[i])
        plt.clf()

def visualize_validation(image_paths, gts, masks, savepaths, scores, cfg, raw_shape):
    '''
    Args:
        images, masks, gts: np.array in the shape of [batch, H, W, C]
    '''
    for i in range(len(image_paths)):
        h, w = raw_shape['height'][i], raw_shape['width'][i]
        image = cv2.imread(image_paths[i]) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        dpi = mpl.rcParams['figure.dpi']
        fig_height, fig_width = h/float(dpi), w/float(dpi)
        
        fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(2*fig_width, 2*fig_height))
        fig.subplots_adjust(wspace=0.3)
        fig.subplots_adjust(hspace=0.3)

        axes[0, 0].imshow(draw_mask(image, gts[i], thres=cfg.INFERENCE.MASK_THRES, raw_shape={'height': h, 'width': w}))
        axes[0, 0].set_title('groundtruth')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(draw_mask(image, masks[i], thres=cfg.INFERENCE.MASK_THRES, raw_shape={'height': h, 'width': w}))
        axes[0, 1].set_title('predicted')
        axes[0, 1].axis('off')

        title = ','.join([f"{k}: {scores[k][i]:.02f}" for k in scores.keys()])
        fig.text(0.5, 0.04, title, ha='center', va='center')
        plt.savefig(savepaths[i])
        plt.clf()

if __name__ == "__main__":
    img_dir = sys.argv[1] #dir to test images
    gt_dir = sys.argv[2] #dir to groundtruth masks
    pred_dir = sys.argv[3] #dir to predicted masks, containing *.npy files
    save_dir = sys.argv[4] # dir to save dir, drawn images will be saved at save_dir/*.jpg
    


