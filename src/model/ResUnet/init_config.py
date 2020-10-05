from model.ResUnet.utils import (
    get_parser,
    get_default_config,
    BCEDiceLoss,
    MetricTracker,
    jaccard_index,
    dice_coeff,
    MyWriter,
)
import torch
import argparse
import os
import os.path as osp

def setup():
    args = get_parser()
    cfg = get_default_config()
    cfg.merge_from_file(args.config)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return args, cfg


