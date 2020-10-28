import warnings
warnings.simplefilter("ignore", (UserWarning, FutureWarning))

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from model.ResUnet.dataset import ImageDataset
from model.ResUnet.utils import metrics
from model.ResUnet.core.res_unet_plus import ResUnetPlusPlus
# from model.ResUnet.core.res_unet_plus_crf import ResUnetPlusPlus

import torch
import argparse
import os
import os.path as osp
import numpy as np
from PIL import Image
from model.ResUnet.utils import augmentation as aug
from model.ResUnet.utils import (
    get_parser, get_default_config, BCEDiceLoss, MetricTracker, jaccard_index, dice_coeff, MyWriter, free_gpu_memory
)
from model.ResUnet.network import Model

seed = 88
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


def save_checkpoint(model, epoch, optimizer, best_score, save_path, criterion=None):
    dict_to_save = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "best_score": best_score,
        "optimizer": optimizer.state_dict(),
    }
    
    if criterion is not None:
        dict_to_save['tsa_thres_history'] = criterion.thres_history
    torch.save(dict_to_save,save_path)
    print("Saved checkpoint to: %s" % save_path)


def do_train(cfg):
    resume = cfg.CHECKPOINT_PATH
    num_epochs = cfg.SOLVER.EPOCH
    checkpoint_dir = "{}/{}".format(cfg.OUTPUT_DIR, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("{}/{}".format(cfg.OUTPUT_DIR, 'log'), exist_ok=True)
    writer = MyWriter("{}/{}".format(cfg.OUTPUT_DIR, 'log'))
    save_path = os.path.join(checkpoint_dir, "best_model.pt" )

    # starting params
    best_loss, best_score = 999, 0.0
    start_epoch = 0
    step = 0
    not_improve_count = 0
    is_val=False

    # get model
    model = Model(cfg)
    model.to_device()
    
    # optimizer
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.get_weights(), lr=cfg.SOLVER.LR, weight_decay=1e-5)

    # decay LR
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", 
    #                 patience=cfg.TRAIN.SCHEDULER_PATIENCE, factor=cfg.TRAIN.SCHEDULER_FACTOR, verbose=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[150, 250],gamma=0.1)
    
    # optionally resume from a checkpoint
    if resume != '':
        checkpoint = model.load_checkpoint(resume)
        start_epoch = checkpoint["epoch"]
        best_score = checkpoint["best_score"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("=> loaded checkpoint '{}', epoch {}, best_score: {}".format(resume, checkpoint["epoch"], checkpoint['best_score']))

    print(f"LOADED MODEL")

    # get data
    train_dataset = ImageDataset(
        cfg, 
        img_path=osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.TRAIN_IMAGES),  
        mask_path=osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.TRAIN_MASKS),
        train=True,
        csv_file=cfg.DATA.TRAIN
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.SOLVER.BATCH_SIZE, num_workers=4, shuffle=True
    )

    if cfg.DATA.VAL != '':
        is_val=True
        val_dataset = ImageDataset(
            cfg, 
            img_path=osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.TRAIN_IMAGES),  
            mask_path=osp.join(cfg.DATA.ROOT_DIR, cfg.DATA.TRAIN_MASKS),
            train=False,
            csv_file=cfg.DATA.VAL
        )
        val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=False)


    # set up binary cross entropy and dice loss
    criterion = metrics.TSA_BCEDiceLoss(cfg, num_steps = cfg.SOLVER.EPOCH*int((len(train_dataset)-1)/cfg.SOLVER.BATCH_SIZE + 1))
    val_criterion = metrics.BCEDiceLoss()

    #----------------------- START TRAINING --------------------------------
    for epoch in range(start_epoch, num_epochs):
        model.set_train()
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 20)

        train_acc = metrics.MetricTracker()
        train_loss = metrics.MetricTracker()
        
        # iterate over data
        loader = tqdm(train_dataloader, desc="training")
        for idx, data in enumerate(loader):
            # get the inputs and wrap in Variable
            inputs = data["sat_img"].cuda() # [batch, C, H, W]
            labels = data["map_img"].cuda() # [batch, H, W]
            optimizer.zero_grad()
            
            # forward
            outputs = model(inputs) # [batch, 1, H, W]
            loss = criterion(outputs, labels)
            
            # backward
            loss.backward()
            optimizer.step()

            train_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
            train_loss.update(loss.data.item(), outputs.size(0))
            
            # if step % cfg.SOLVER.LOGGING_STEP == 0:
            loader.set_description(
                "Training Loss: {:.4f}, dice_coeff: {:.4f}".format(train_loss.avg, train_acc.avg)
            )
        
        free_gpu_memory()
        # tensorboard logging
        print(f"Training Loss: {train_loss.avg:.4f}, dice_coeff: {train_acc.avg:.4f}")
        writer.log_training(train_loss.avg, train_acc.avg, epoch)

        if not is_val:
            continue
        #----------------------- START VALIDATING --------------------------------
        valid_metrics = do_val(val_dataloader, model, val_criterion, writer, epoch)
        lr_scheduler.step(epoch)

        # store best loss and save a model checkpoint
        if valid_metrics["dice_coeff"] > best_score:
            best_score = valid_metrics["dice_coeff"]
            model.save_checkpoint(save_path, epoch, best_score, optimizer)
            # save_checkpoint(model, epoch, optimizer, best_score, save_path)
            not_improve_count = 0
        else:
            not_improve_count += 1
            if (cfg.SOLVER.EARLY_STOPPING != -1) and (not_improve_count >= cfg.SOLVER.EARLY_STOPPING):
                break
        
        # save last model
        if epoch == num_epochs - 1:
            save_checkpoint(model, num_epochs, optimizer, best_score, save_path.replace('best_model', 'final_model'))

def do_val(valid_loader, model, criterion, logger, step):

    # logging accuracy and loss
    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()

    # switch to evaluate mode
    model.set_eval()

    # Iterate over data.
    with torch.no_grad():
        for idx, data in tqdm(enumerate(valid_loader), desc="validation"):
            # get the inputs and wrap in Variable
            inputs = data["sat_img"].cuda()
            labels = data["map_img"].cuda()
            
            outputs = model(inputs)
            # outputs = torch.nn.functional.sigmoid(outputs)

            loss = criterion(outputs, labels)

            valid_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
            valid_loss.update(loss.data.item(), outputs.size(0))

            if idx == 0:
                logger.log_images(inputs.cpu(), labels.cpu(), outputs.cpu(), step)

        logger.log_validation(valid_loss.avg, valid_acc.avg, step)

    print("Validation Loss: {:.4f} dice-coeff: {:.4f}".format(valid_loss.avg, valid_acc.avg))
    return {"valid_loss": valid_loss.avg, "dice_coeff": valid_acc.avg}

def setup():
    args = get_parser()
    cfg = get_default_config()
    cfg.merge_from_file(args.config)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return args, cfg

if __name__ == "__main__":
    args, cfg = setup()
    do_train(cfg)
    