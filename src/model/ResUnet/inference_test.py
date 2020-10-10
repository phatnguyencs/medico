import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import glob
import dataset
from model.ResUnet.utils import metrics
from model.ResUnet.core.res_unet import ResUnet
from model.ResUnet.core.res_unet_plus import ResUnetPlusPlus
import torch
import argparse
import os
import os.path as osp
from model.ResUnet.utils import (
    get_parser,
    get_default_config,
    BCEDiceLoss,
    MetricTracker,
    jaccard_index,
    dice_coeff,
    MyWriter,
)
from model.ResUnet.init_config import setup
from model.ResUnet.utils.visualize import visualize_validation, visualize_prediction
from model.ResUnet.dataset import ImageDataset



def main():
    # Setup
    args, cfg = setup()
    checkpoint_dir = f"{cfg.OUTPUT_DIR}/checkpoints"
    batch_size = cfg.INFERENCE.BATCH_SIZE

    test_img_dir = osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.TEST_IMAGES)
    list_imgs = os.listdir(test_img_dir)

    if cfg.MODEL.NAME == 'res_unet_plus':
        model = ResUnetPlusPlus(3).cuda()
    else:
        model = ResUnet(3, 64).cuda()

    resume = osp.join(checkpoint_dir, 'best_model.pt')
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"LOADED MODEL SUCCESSFULLY")

    model.eval()
    vis_savedir = osp.join(cfg.INFERENCE.SAVE_DIR, 'visualize_test')
    os.makedirs(vis_savedir, exist_ok=True)
    with torch.no_grad():
        for img_path in tqdm(list_imgs):
            if img_path[0] == '.':
                continue
            img_path = osp.join(test_img_dir, img_path)
            img_input, raw_shape = ImageDataset.prepare_image(img_path, cfg)
            img_input = img_input.cuda().unsqueeze(0)
            output = model(img_input)

            img_name = img_path.split('/')[-1]
            save_path = osp.join(vis_savedir, img_name)
            visualize_prediction(
                images = img_input.cpu().permute(0, 2, 3, 1).numpy(),
                masks = output.cpu().permute(0, 2, 3, 1).numpy(),
                savepaths=[save_path],
                cfg=cfg,
                raw_shape=raw_shape
            )



if __name__ == "__main__":
    main()