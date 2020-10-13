import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import model.ResUnet.dataset as dataset
from model.ResUnet.utils import metrics
from model.ResUnet.core.res_unet import ResUnet
from model.ResUnet.core.res_unet_plus import ResUnetPlusPlus
import torch
import argparse
import os
import os.path as osp
import numpy as np
from model.ResUnet.utils import (
    get_parser,get_default_config,BCEDiceLoss,MetricTracker,jaccard_index,dice_coeff,MyWriter,
)
from model.ResUnet.init_config import setup
from model.ResUnet.utils.visualize import visualize_validation
from model.ResUnet.utils import augmentation as aug

# import conv_crf
# from model.ResUnet.convcrf import convcrf
from model.postprocess.apply_crf import get_dcrf_model, apply_dcrf_model

def refine_result(images, masks, crf_model, cfg):
    # Convert to numpy:
    np_masks = masks.cpu().numpy().transpose(0, 2, 3, 1) # [batch, 1, H, W] --> [batch, H, W, 1]
    np_imgs = images.cpu().numpy().transpose(0, 2, 3, 1) # [batch, 1, H, W] --> [batch, H, W, 1]
    batch_size = np_masks.shape[0]
    
    for i in range(batch_size):
        refined_mask = apply_dcrf_model(np_imgs[i], np_masks[i], crf_model, n_steps=cfg.INFERENCE.CRF_STEP) 
        np_masks[i] = np.expand_dims(refined_mask, axis=2) # [H, W] --> [H, W, 1]
    
    return torch.Tensor(np_masks.transpose(0, 3, 1, 2)) # covnert back to [batch, 1, H, W]

def main():
    # Setup
    args, cfg = setup()
    checkpoint_dir = f"{cfg.OUTPUT_DIR}/checkpoints"
    batch_size = cfg.INFERENCE.BATCH_SIZE

    dataset_val = dataset.ImageDataset(
        cfg, 
        img_path=osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.TRAIN_IMAGES),  
        mask_path=osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.TRAIN_MASKS),
        train=False,
    )
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, num_workers=4, shuffle=False)
    val_tracker = metrics.ValidationTracker()

    # Load checkpoint
    model = ResUnetPlusPlus(3).cuda()

    crf_model, crf_val_tracker = None, None

    if cfg.INFERENCE.CRF:
        crf_model = get_dcrf_model(cfg.MODEL.IMAGE_SIZE)
        crf_val_tracker = metrics.ValidationTracker()

    resume = osp.join(checkpoint_dir, 'best_model.pt')
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"LOADED MODEL SUCCESSFULLY")

    visualization_save_dir = osp.join(cfg.INFERENCE.SAVE_DIR, 'visualize_val')
    os.makedirs(visualization_save_dir, exist_ok=True)
    
    if cfg.INFERENCE.CRF:
        crf_visualization_save_dir = osp.join(cfg.INFERENCE.SAVE_DIR, 'visualize_val_crf')
        os.makedirs(crf_visualization_save_dir, exist_ok=True)

    # do inference
    model.eval()
    with torch.no_grad():
        for idx, data in tqdm(enumerate(val_dataloader)):
            inputs = data['sat_img'].cuda()
            labels = data['map_img'].cuda()

            # print(data['raw_shape'])
            img_paths, raw_shape = data['image_path'], data['raw_shape']

            outputs = model(inputs)
            
            imgs = inputs.cpu().permute(0, 2, 3, 1).numpy()
            gts = labels.cpu().numpy()
            
            if crf_model is not None:
                crf_preds = refine_result(inputs, outputs, crf_model, cfg)
                crf_all_scores = metrics.calculate_all_metrics(crf_preds.cuda(), labels)
                crf_preds = crf_preds.cpu().permute(0, 2, 3, 1).numpy()
                crf_val_tracker.update(crf_all_scores)
                img_names = [p.strip().split('/')[-1] for p in img_paths]
                save_paths = [osp.join(visualization_save_dir, p) for p in img_names]
                visualize_validation(img_paths, gts, crf_preds, save_paths, crf_all_scores, cfg, raw_shape)

            all_scores = metrics.calculate_all_metrics(outputs, labels)
            img_names = [p.strip().split('/')[-1] for p in img_paths]
            save_paths = [osp.join(visualization_save_dir, p) for p in img_names]
            preds = outputs.cpu().permute(0, 2, 3, 1).numpy()
            
            val_tracker.update(all_scores)
            visualize_validation(img_paths, gts, preds, save_paths, all_scores, cfg, raw_shape)
        
        val_tracker.to_json(osp.join(cfg.INFERENCE.SAVE_DIR, 'scores.json'))

        if crf_val_tracker:
            crf_val_tracker.to_json(osp.join(cfg.INFERENCE.SAVE_DIR, 'crf_scores.json'))

        

if __name__ == "__main__":
    main()





# ----------------------------------------
# code for convcrf: https://github.com/MarvinTeichmann/ConvCRF 
# TODO: How to train this module?
def setup_convcrf(cfg):
    config  = convcrf.get_default_conf()
    num_classes = 1
    config['filter_size'] = 7
    config['pyinn'] = False
    config['col_feats']['schan'] = 0.1 # as we are using normalized images
    shape = cfg.MODEL.IMAGE_SIZE

    model = convcrf.GaussCRF(conf=config, shape=shape, nclasses=num_classes)
    model.cuda()

    return model

def apply_convcrf(crf_model, images, preds):
    '''
    Args:
        images: torch tensor in shape of [batch, 3, H, W]
        preds: predicted masks in shape of [batch, 1, H, W]
    
    Return:
        output: refined masks in shape of [batch, 1, H, W]
    '''        
    output = crf_model.forward(unary=preds, img=images)
    return output 
    