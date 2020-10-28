import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import argparse
import os
import os.path as osp
from torch.utils.data import DataLoader
from torchvision import transforms

from model.PraNet.utils import (
    StructureLoss, BCEDiceLoss, clip_gradient, adjust_lr, AvgMeter, get_default_config, 
    free_gpu_memory, get_parser, MyWriter
)
from model.PraNet.dataset import ImageDataset
from model.PraNet.network import MedicoNet
from model.PraNet.utils import metrics

import warnings
warnings.simplefilter("ignore", (UserWarning, FutureWarning))

seed = 88
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


def epoch_val(valid_loader, model, criterion, logger, step):
    # logging accuracy and loss
    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()

    # switch to evaluate mode
    model.set_eval()

    # Iterate over data.
    with torch.no_grad():
        for idx, data in tqdm(enumerate(valid_loader), desc="validation"):
            # get the inputs and wrap in Variable
            inp = data["sat_img"].cuda()
            gt = data["map_img"].cuda() #[B, H, W]
            raw_shape = {
                'height': gt.shape[1],
                'width': gt.shape[2],
            }
            
            output = model.predict_mask(inp, raw_shape)
            # outputs = torch.nn.functional.sigmoid(outputs)
            loss = criterion(output, gt)

            valid_acc.update(metrics.dice_coeff(output, gt), output.size(0))
            valid_loss.update(loss.data.item(), output.size(0))

            if idx == 0:
                logger.log_images(inp.cpu(), gt.cpu(), gt.cpu(), step)

        logger.log_validation(valid_loss.avg, valid_acc.avg, step)

    print("Validation Loss: {:.4f} dice-coeff: {:.4f}".format(valid_loss.avg, valid_acc.avg))
    return {"valid_loss": valid_loss.avg, "dice_coeff": valid_acc.avg}

def epoch_train(model, dataloader, criterion, optimizer, trainsize):
    '''
        Training logic for each epoch. 
        Remember to turn model on training mode before calling this function.
    '''
    model.set_train()
    train_acc = metrics.MetricTracker()
    loss_record2, loss_record3, loss_record4, loss_record5 = metrics.MetricTracker(), metrics.MetricTracker(), metrics.MetricTracker(), metrics.MetricTracker()
    total_loss_record = metrics.MetricTracker()
    epoch_result = {}

    # iterate over the dataset
    loader = tqdm(dataloader, desc="training")
    for idx, data in enumerate(loader):
        # get the inputs and wrap in Variable
        inputs = data["sat_img"].cuda() # [batch, C, H, W]
        labels = data["map_img"].cuda() # [batch, H, W]
        raw_shape = {
            'height': labels.shape[1],
            'width': labels.shape[2],
        }
        
        optimizer.zero_grad()
        
        # forward
        outputs = model(inputs) # [batch, 1, H, W]
        predictions = model.get_prediction_from_output(outputs, raw_shape)
        
        assert predictions.shape == labels.shape, "output shape must equals to input shape"
        
        # loss
        loss, loss2, loss3, loss4, loss5 = criterion(outputs, labels) 
        
        # backward
        loss.backward()
        optimizer.step()

        loss_record2.update(loss2, inputs.size(0))
        loss_record3.update(loss3, inputs.size(0))
        loss_record4.update(loss4, inputs.size(0))
        loss_record5.update(loss5, inputs.size(0))
        total_loss_record.update(loss, inputs.size(0))

        train_acc.update(metrics.dice_coeff(predictions, labels), inputs.size(0))

    epoch_result['total_loss'] = total_loss_record.avg
    epoch_result['dice_coef'] = train_acc.avg
    epoch_result['loss_2'] = loss_record2.avg
    epoch_result['loss_3'] = loss_record3.avg
    epoch_result['loss_4'] = loss_record4.avg
    epoch_result['loss_5'] = loss_record5.avg

    return epoch_result

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
    model = MedicoNet(cfg)
    model.to_device()
    # optimizer
    optimizer = torch.optim.Adam(model.get_weights(), lr=cfg.SOLVER.LR, weight_decay=1e-5)
    # decay LR
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[150, 250],gamma=0.1)
    
    # optionally resume from a checkpoint
    if resume != '':
        checkpoint = model.load_checkpoint(resume)
        start_epoch = checkpoint["epoch"]
        best_score = checkpoint["best_score"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("=> loaded checkpoint from epoch {}, best_score: {}".format(checkpoint["epoch"], checkpoint['best_score']))
    else:
        cfg.CHECKPOINT_PATH = osp.join(cfg.OUTPUT_DIR, 'checkpoints/best_model.pt')

    print(f"LOADED MODEL SUCCESSFULLY")

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


    train_criterion = StructureLoss()
    val_criterion = BCEDiceLoss()

    #----------------------- START TRAINING --------------------------------
    for epoch in range(start_epoch, num_epochs):
        model.set_train()
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 20)

        train_params = {
            'model': model, 'optimizer': optimizer, 
            'dataloader': train_dataloader, 'criterion': train_criterion,
            'trainsize': cfg.MODEL.IMAGE_SIZE
        }
        log = epoch_train(**train_params)
        free_gpu_memory()
        
        print(f"loss2: {log['loss_2']:.4f}, loss3: {log['loss_3']:.4f}, loss4: {log['loss_4']:.4f}, loss5: {log['loss_5']:.4f}, total_loss: {log['total_loss']:.2f}, dice_coeff: {log['dice_coef']:.4f}")

        # tensorboard logging
        writer.log_training(log['total_loss'], log['dice_coef'], epoch)

        if not is_val:
            continue
        #----------------------- START VALIDATING --------------------------------
        valid_metrics = epoch_val(val_dataloader, model, val_criterion, writer, epoch)
        lr_scheduler.step(epoch)

        # store best loss and save a model checkpoint
        if valid_metrics["dice_coeff"] > best_score:
            best_score = valid_metrics["dice_coeff"]
            model.save_checkpoint(save_path, epoch, best_score, optimizer)
            print(f"save checkpoint at epoch {epoch} with lr: {optimizer.param_groups[0]['lr']}")
            not_improve_count = 0
        else:
            not_improve_count += 1
            if (cfg.SOLVER.EARLY_STOPPING != -1) and (not_improve_count >= cfg.SOLVER.EARLY_STOPPING):
                break
        
        # save last model
        if epoch == num_epochs - 1:
            model.save_checkpoint(cfg.CHECKPOINT_PATH, epoch, best_score, optimizer)

def setup():
    args = get_parser()
    cfg = get_default_config()
    cfg.merge_from_file(args.config)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return args, cfg

if __name__ == "__main__":
    args, cfg = setup()
    do_train(cfg)
    