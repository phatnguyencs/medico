import warnings
warnings.simplefilter("ignore", (UserWarning, FutureWarning))

from utils.hparams import HParam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import dataset
from utils import metrics
from core.res_unet import ResUnet
from core.res_unet_plus import ResUnetPlusPlus
import torch
import argparse
import os
import os.path as osp

from utils import (
    get_parser,
    get_default_config,
    BCEDiceLoss,
    MetricTracker,
    jaccard_index,
    dice_coeff,
    MyWriter,
)

def do_train(cfg, name):
    resume = cfg.CHECKPOINT_PATH
    num_epochs = cfg.SOLVER.EPOCH
    checkpoint_dir = "{}/{}".format(cfg.OUTPUT_DIR, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("{}/{}".format(cfg.OUTPUT_DIR, 'log'), exist_ok=True)
    writer = MyWriter("{}/{}".format(cfg.OUTPUT_DIR, 'log'))

    # get model
    if cfg.MODEL.NAME == 'res_unet_plus':
        model = ResUnetPlusPlus(3).cuda()
    else:
        model = ResUnet(3, 64).cuda()
    print(f"LOADED MODEL")

    # set up binary cross entropy and dice loss
    criterion = metrics.BCEDiceLoss()

    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    # decay LR
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # starting params
    best_loss = 999
    start_epoch = 0
    # optionally resume from a checkpoint
    if resume != '':
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                resume, checkpoint["epoch"]
            )
        )
       
    # get data
    train_transforms = [
        dataset.AdjustContrast(),
        # dataset.AdjustBrightness(),
        # dataset.Rotate(),
        dataset.ToTensorTarget(),
    ]
    mass_dataset_train = dataset.ImageDataset(
        cfg, True, transform=transforms.Compose(train_transforms)
    )
    train_dataloader = DataLoader(
        mass_dataset_train, batch_size=cfg.SOLVER.BATCH_SIZE, num_workers=4, shuffle=True
    )

    if cfg.DATA.VAL != '':
        mass_dataset_val = dataset.ImageDataset(
            cfg, False, transform=transforms.Compose([dataset.ToTensorTarget()])
        )
        val_dataloader = DataLoader(
            mass_dataset_val, batch_size=1, num_workers=4, shuffle=False
        )

    step = 0
    not_improve_count = 0
    for epoch in range(start_epoch, num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 20)

        # step the learning rate scheduler
        lr_scheduler.step()

        # run training and validation
        # logging accuracy and loss
        train_acc = metrics.MetricTracker()
        train_loss = metrics.MetricTracker()
        
        # iterate over data
        loader = tqdm(train_dataloader, desc="training")
        for idx, data in enumerate(loader):

            # get the inputs and wrap in Variable
            inputs = data["sat_img"].cuda()
            labels = data["map_img"].cuda()
            # print(labels.shape)
            # print(torch.max(labels))
            # print(torch.min(labels))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # prob_map = model(inputs) # last activation was a sigmoid
            # outputs = (prob_map > 0.3).float()
            outputs = model(inputs)
            # outputs = torch.nn.functional.sigmoid(outputs)

            loss = criterion(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()

            train_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
            train_loss.update(loss.data.item(), outputs.size(0))

            # tensorboard logging
            if step % cfg.SOLVER.LOGGING_STEP == 0:
                writer.log_training(train_loss.avg, train_acc.avg, step)
                loader.set_description(
                    "Training Loss: {:.4f} Acc: {:.4f}".format(
                        train_loss.avg, train_acc.avg
                    )
                )

            # Validatiuon
            step += 1
        
        if cfg.DATA.VAL == '':
            continue

        valid_metrics = do_val(
            val_dataloader, model, criterion, writer, epoch
        )
        save_path = os.path.join(checkpoint_dir, "best_model.pt" )
        # store best loss and save a model checkpoint
        if valid_metrics["valid_loss"] < best_loss:
            best_loss = valid_metrics["valid_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "arch": cfg.MODEL.NAME,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            print("Saved checkpoint to: %s" % save_path)
            not_improve_count += 1
            if not_improve_count % cfg.SOLVER.EARLY_STOPPING == 0:
                break
        else:
            not_improve_count = 0


def do_val(valid_loader, model, criterion, logger, step):

    # logging accuracy and loss
    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()

    # switch to evaluate mode
    model.eval()

    # Iterate over data.
    with torch.no_grad():
        for idx, data in enumerate(tqdm(valid_loader, desc="validation")):
            # get the inputs and wrap in Variable
            inputs = data["sat_img"].cuda()
            labels = data["map_img"].cuda()
            

            # forward
            # prob_map = model(inputs) # last activation was a sigmoid
            # outputs = (prob_map > 0.3).float()
            outputs = model(inputs)
            # outputs = torch.nn.functional.sigmoid(outputs)

            loss = criterion(outputs, labels)

            valid_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
            valid_loss.update(loss.data.item(), outputs.size(0))
            if idx == 0:
                logger.log_images(inputs.cpu(), labels.cpu(), outputs.cpu(), step)
        logger.log_validation(valid_loss.avg, valid_acc.avg, step)

    print("Validation Loss: {:.4f} Acc: {:.4f}".format(valid_loss.avg, valid_acc.avg))
    model.train()
    return {"valid_loss": valid_loss.avg, "valid_acc": valid_acc.avg}


def setup():
    args = get_parser()
    cfg = get_default_config()
    cfg.merge_from_file(args.config)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return args, cfg

if __name__ == "__main__":
    args, cfg = setup()
    do_train(cfg, name=args.name)
    