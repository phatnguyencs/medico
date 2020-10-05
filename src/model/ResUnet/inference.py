import warnings
warnings.simplefilter("ignore", (UserWarning, FutureWarning))

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
    get_parser,
    get_default_config,
    BCEDiceLoss,
    MetricTracker,
    jaccard_index,
    dice_coeff,
    MyWriter,
)
from model.ResUnet.init_config import setup


def main():
    # Setup
    args, cfg = setup()
    checkpoint_dir = f"{cfg.OUTPUT_DIR}/checkpoints"

    # Create val dataset:
    val_dataset = dataset.ImageDataset(
        cfg, False, transform=transforms.Compose([dataset.ToTensorTarget()])
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, num_workers=4, shuffle=False
    )

    # Load checkpoint
    if cfg.MODEL.NAME == 'res_unet_plus':
        model = ResUnetPlusPlus(3).cuda()
    else:
        model = ResUnet(3, 64).cuda()
    resume = osp.join(checkpoint_dir, 'best_model.pt')
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"LOADED MODEL SUCCESSFULLY")

    # do inference
    model.eval()
    with torch.no_grad():
        for idx, data in tqdm(enumerate(val_dataloader)):
            inputs = data['sat_img'].cuda()
            labels = data['map_img'].cuda()

            outputs = model(inputs)

            



            


    


if __name__ == "__main__":
    main()