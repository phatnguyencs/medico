import os 
import os.path as osp 
from glob import glob
import cv2 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib as mpl
from torchvision import transforms

# warnings.simplefilter("ignore", (UserWarning, FutureWarning))

def thresholding_mask(preds, thres=0.5):
    preds[preds>=thres] = 1.0
    preds[preds<thres] = 0.0
    return preds

def reshape_image(image, raw_shape):
    np2pil = transforms.ToPILImage()
    pil_mask = np2pil(image)
    pil_mask = transforms.Resize((raw_shape['height'], raw_shape['width']), Image.NEAREST)(pil_mask)
    mask = np.array(pil_mask)

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

        fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(2*fig_width, 1.1*fig_height), dpi=dpi)
        fig.subplots_adjust(wspace=0.1)
        fig.subplots_adjust(hspace=0.1)

        axes[0, 0].imshow(images[i])
        axes[0, 0].set_title('input')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(draw_mask(images[i], masks[i], thres=cfg.INFERENCE.MASK_THRES, raw_shape={'height': h, 'width': w}))
        axes[0, 1].set_title('predicted')
        axes[0, 1].axis('off')

        plt.savefig(savepaths[i])
        plt.clf()


def setup_folder(save_dir: str):
    os.makedirs(osp.join(save_dir, '(0.7,inf]'), exist_ok=True)
    os.makedirs(osp.join(save_dir, '(0.5,0.7]'), exist_ok=True)
    os.makedirs(osp.join(save_dir, '(-inf,0.5]'), exist_ok=True)

def choose_img_folder_by_score(score):
    if score > 0.7:
        return '(0.7,inf]'
    elif score > 0.5 and score <= 0.7 :
        return '(0.5,0.7]'
    elif score <= 0.5:
        return '(-inf,0.5]'

def visualize_validation(image_paths, gts, masks, save_dir, scores, cfg, raw_shape, is_split=True):
    '''
    Args:
        images, masks, gts: np.array in the shape of [batch, H, W, C]
        scores: 
        save_dir: dir to save images. If is_split is True: 
            --> visualized images will be splitted in 3 different folders: (0.7,inf], (0.5,0.7], (-inf,0.5]
    '''
    if is_split:
        setup_folder(save_dir)

    for i in range(len(image_paths)):
        h, w = raw_shape['height'][i], raw_shape['width'][i]
        image = cv2.imread(image_paths[i]) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        dpi = mpl.rcParams['figure.dpi']
        fig_height, fig_width = h/float(dpi), w/float(dpi)
        
        n_images = 3
        fig, axes = plt.subplots(nrows=1, ncols=n_images, squeeze=False, figsize=(n_images*fig_width, 1.1*fig_height), dpi=dpi)
        fig.subplots_adjust(wspace=0.1)
        fig.subplots_adjust(hspace=0.2)

        axes[0, 0].imshow(image)
        axes[0, 0].set_title('input')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(draw_mask(image, gts[i], thres=cfg.INFERENCE.MASK_THRES, raw_shape={'height': h, 'width': w}))
        axes[0, 1].set_title('groundtruth')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(draw_mask(image, masks[i], thres=cfg.INFERENCE.MASK_THRES, raw_shape={'height': h, 'width': w}))
        axes[0, 2].set_title('predicted')
        axes[0, 2].axis('off')

        img_save_path = osp.join(save_dir, image_paths[i].split('/')[-1])

        if is_split:
            img_dice = scores['dice_coeff'][i]
            save_folder = choose_img_folder_by_score(img_dice)
            img_save_path = osp.join(save_dir, save_folder, image_paths[i].split('/')[-1])

        title = ','.join([f"{k}: {scores[k][i]:.02f}" for k in scores.keys()])

        fig.text(0.5, 0.04, title, ha='center', va='center')
        plt.savefig(img_save_path)
        plt.clf()
    

    

