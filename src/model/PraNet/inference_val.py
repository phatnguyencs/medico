import warnings
warnings.filterwarnings("ignore")
import torch
import argparse
import os
import os.path as osp
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model.PraNet.utils import (
    StructureLoss, BCEDiceLoss, clip_gradient, adjust_lr, AvgMeter, get_default_config, 
    free_gpu_memory, get_parser, MyWriter, TSA_StructureLoss
)
from model.PraNet.dataset import ImageDataset
from model.PraNet.network import MedicoNet
from model.PraNet.utils import metrics
from model.PraNet.utils.visualize import (
    visualize_validation, visualize_prediction, thresholding_mask 
)

# import conv_crf
# from model.ResUnet.convcrf import convcrf
from model.postprocess.apply_crf import get_dcrf_model, apply_dcrf_model

def refine_result(images, masks, crf_model, cfg):
    '''
    Args:
        images, masks: [B, 1, H, W]
    Return:
        refined_masks: [B, 1, H, W]
    '''

    # Convert to numpy:
    np_masks = masks.cpu().numpy().transpose(0, 2, 3, 1) # [batch, 1, H, W] --> [batch, H, W, 1]
    np_imgs = images.cpu().numpy().transpose(0, 2, 3, 1) # [batch, 1, H, W] --> [batch, H, W, 1]
    batch_size = np_masks.shape[0]
    
    for i in range(batch_size):
        refined_mask = apply_dcrf_model(np_imgs[i], np_masks[i], crf_model, n_steps=cfg.INFERENCE.CRF_STEP) 
        np_masks[i] = np.expand_dims(refined_mask, axis=2) # [H, W] --> [H, W, 1]
    
    return torch.Tensor(np_masks.transpose(0, 3, 1, 2)) # covnert back to [batch, 1, H, W]

def inference(model, cfg, dataset_type = 'val'):
    '''
    Args:
        dataset_type: 'train' or 'val', to inference on train/val set
                    >> output image: [input, groundtruth, prediction]
    '''
    # Setup
    batch_size = 1
    csv2load = cfg.DATA.VAL
    if dataset_type == 'train':
        csv2load = cfg.DATA.TRAIN


    dataset_val = ImageDataset(
        cfg, 
        img_path=osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.TRAIN_IMAGES),  
        mask_path=osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.TRAIN_MASKS),
        train=False,
        csv_file=csv2load
    )
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, num_workers=4, shuffle=False)
    val_tracker = metrics.ValidationTracker()

    crf_model, crf_val_tracker = None, None

    
    visualization_save_dir = osp.join(cfg.INFERENCE.SAVE_DIR, f'{dataset_type}_visualize_{cfg.INFERENCE.MASK_THRES:.01f}')
    res_json_path = osp.join(cfg.INFERENCE.SAVE_DIR, f'{dataset_type}_scores_{cfg.INFERENCE.MASK_THRES:.01f}.json')

    if cfg.INFERENCE.CRF:
        crf_model = get_dcrf_model(cfg.MODEL.IMAGE_SIZE)
        crf_val_tracker = metrics.ValidationTracker()
        res_json_path.replace('.json', '_crf.json')

    if cfg.INFERENCE.TTA:
        visualization_save_dir += '_tta'
        res_json_path.replace('.json', '_tta.json')

    os.makedirs(visualization_save_dir, exist_ok=True)

    
    if cfg.INFERENCE.CRF:
        crf_visualization_save_dir = osp.join(cfg.INFERENCE.SAVE_DIR, f'{dataset_type}_visualize_crf_{cfg.INFERENCE.MASK_THRES:.01f}')
        os.makedirs(crf_visualization_save_dir, exist_ok=True)

    # do inference
    model.eval()
    with torch.no_grad():
        for idx, data in tqdm(enumerate(val_dataloader)):
            inputs = data['sat_img'].cuda()
            labels = data['map_img'].cuda() # [B, H, W]
            img_paths = data['image_path']
            raw_shape = {
                'height': data['raw_shape']['height'][0].item(),
                'width': data['raw_shape']['width'][0].item(),
            }
            if cfg.INFERENCE.TTA:
                outputs = model.predict_mask_tta(inputs, raw_shape)
            else:
                outputs = model.predict_mask(inputs, raw_shape)

            outputs = thresholding_mask(outputs, cfg.INFERENCE.MASK_THRES) # [B, H, W]
            imgs = inputs.cpu().permute(0, 2, 3, 1).numpy() # [B, C, H, W] --> [B, H, W, C]
            gts = labels.cpu().numpy()# [B, H, W]
            
            if crf_model is not None:
                crf_preds = refine_result(inputs, outputs.unsqueeze(1), crf_model, cfg)
                crf_all_scores = metrics.calculate_all_metrics(crf_preds.cuda(), labels)
                crf_preds = crf_preds.cpu().permute(0, 2, 3, 1).numpy()
                crf_val_tracker.update(crf_all_scores) 
                img_names = [p.strip().split('/')[-1] for p in img_paths]
                save_paths = [osp.join(crf_visualization_save_dir, p) for p in img_names]
                visualize_validation(img_paths, gts, crf_preds, crf_visualization_save_dir, crf_all_scores, cfg, raw_shape)

            pred_masks = outputs.cpu().numpy()
            all_scores = metrics.calculate_all_metrics_raw_size(outputs, labels, raw_shape)
            img_names = [p.strip().split('/')[-1] for p in img_paths]
            save_paths = [osp.join(visualization_save_dir, p) for p in img_names]
            
            val_tracker.update(all_scores)
            visualize_validation(img_paths, gts, np.expand_dims(pred_masks, axis=0), visualization_save_dir, all_scores, cfg, raw_shape)
        
        val_tracker.to_json(res_json_path)

        if crf_val_tracker:
            crf_val_tracker.to_json(osp.join(cfg.INFERENCE.SAVE_DIR, f'{dataset_type}_crf_scores_{cfg.INFERENCE.MASK_THRES:.01f}.json'))


def setup():
    args = get_parser()
    cfg = get_default_config()
    cfg.merge_from_file(args.config)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return args, cfg

def prepare_model(cfg, checkpoint_dir: str):
    model = MedicoNet(cfg)
    resume = checkpoint_dir
    # if 'best_model.pt' not in checkpoint_dir:
    #     resume = osp.join(checkpoint_dir, 'best_model.pt')
    # else:
    #     resume = checkpoint_dir

    model.load_checkpoint(resume)
    model.to_device()
    print(f"LOADED MODEL SUCCESSFULLY")

    model.eval()
    return model

if __name__ == "__main__":
    args, cfg = setup()

    # Load checkpoint
    model = prepare_model(cfg, cfg.CHECKPOINT_PATH)

    # print("inference train set")
    # inference(model, cfg, 'train')
    print("inference val set")
    inference(model, cfg, 'val')





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
    args:
        images: torch tensor in shape of [batch, 3, H, W]
        preds: predicted masks in shape of [batch, 1, H, W]
    
    Return:
        output: refined masks in shape of [batch, 1, H, W]
    '''        
    output = crf_model.forward(unary=preds, img=images)
    return output 
    