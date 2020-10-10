import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import dataset
from model.ResUnet.utils import metrics
from model.ResUnet.core.res_unet import ResUnet
from model.ResUnet.core.res_unet_plus import ResUnetPlusPlus
import torch
import argparse
import os
import os.path as osp
from model.ResUnet.utils import (
    get_parser,get_default_config,BCEDiceLoss,MetricTracker,jaccard_index,dice_coeff,MyWriter,
)
from model.ResUnet.init_config import setup
from model.ResUnet.utils.visualize import visualize_validation
from model.ResUnet.utils import augmentation as aug

def main():
    # Setup
    args, cfg = setup()
    checkpoint_dir = f"{cfg.OUTPUT_DIR}/checkpoints"
    batch_size = cfg.INFERENCE.BATCH_SIZE

    image_transforms, label_transforms = aug.create_transform(cfg, 'val')
    dataset_val = dataset.ImageDataset(
        cfg, 
        img_path=osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.TRAIN_IMAGES),  
        mask_path=osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.TRAIN_MASKS),
        train=True,
        image_transform=transforms.Compose(image_transforms),
        label_transform=transforms.Compose(label_transforms),
    )
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, num_workers=4, shuffle=False)


    val_tracker = metrics.ValidationTracker()

    # Load checkpoint
    model = ResUnetPlusPlus(3).cuda()

    resume = osp.join(checkpoint_dir, 'best_model.pt')
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"LOADED MODEL SUCCESSFULLY")

    visualization_save_dir = osp.join(cfg.INFERENCE.SAVE_DIR, 'visualize_train')
    os.makedirs(visualization_save_dir, exist_ok=True)
    # do inference
    model.eval()
    with torch.no_grad():
        for idx, data in tqdm(enumerate(val_dataloader)):
            inputs = data['sat_img'].cuda()
            labels = data['map_img'].cuda()

            # print(data['raw_shape'])
            img_paths, raw_shape = data['image_path'], data['raw_shape']

            outputs = model(inputs)
            all_scores = metrics.calculate_all_metrics(outputs, labels)
            img_names = [p.strip().split('/')[-1] for p in img_paths]
            save_paths = [osp.join(visualization_save_dir, p) for p in img_names]

            imgs = inputs.cpu().permute(0, 2, 3, 1).numpy()
            gts = labels.cpu().numpy()
            preds = outputs.cpu().permute(0, 2, 3, 1).numpy()
            
            val_tracker.update(all_scores)
            visualize_validation(img_paths, gts, preds, save_paths, all_scores, cfg, raw_shape)
        
        val_tracker.to_json(osp.join(cfg.INFERENCE.SAVE_DIR, 'scores.json'))
        

if __name__ == "__main__":
    main()