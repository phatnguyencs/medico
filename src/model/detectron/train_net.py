import logging
import os
import os.path as osp
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
from utils import create_dir
import copy

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch, DefaultPredictor
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
import detectron2.data.transforms as T
from detectron2.data import detection_utils

from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.modeling import GeneralizedRCNNWithTTA, build_model
from detectron2.data.datasets import register_coco_instances
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.solver import build_lr_scheduler, build_optimizer

## CenterMask override modules
from centermask.evaluation import COCOEvaluator
from centermask.config import get_cfg
from centermask.checkpoint import AdetCheckpointer

# My config
from centermask.custom_detectron import build_detection_val_loader


logger = logging.getLogger(__name__)


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

from tqdm import tqdm
def do_val(cfg, model, val_dataloader):
    with torch.no_grad():
        val_loss = 0
        tmp = None
        for idx, inputs in tqdm(enumerate(val_dataloader)):
            outputs = model(inputs)
            loss_dict_reduced = {
                k: v.item() for k, v in comm.reduce_dict(outputs).items()
            }
            tmp = loss_dict_reduced
            reduced_loss = sum(loss for loss in loss_dict_reduced.values())
            val_loss += reduced_loss
    
    return val_loss

def do_train(cfg, model, resume=False, val_set='firevysor_val'):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1, last_epoch=-1)
    metric = 0
    print_every = 50

    tensorboard_dir = osp.join(cfg.OUTPUT_DIR, 'tensorboard')
    checkpoint_dir = osp.join(cfg.OUTPUT_DIR, 'checkpoints')
    create_dir(tensorboard_dir)
    create_dir(checkpoint_dir)

    checkpointer = AdetCheckpointer(
        model, checkpoint_dir, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )


    writers = (
        [
            CommonMetricPrinter(max_iter),
            # JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(tensorboard_dir),
        ]
        if comm.is_main_process()
        else []
    )
    data_loader = build_detection_train_loader(cfg)
    val_dataloader = build_detection_val_loader(cfg, val_set)

    logger.info("Starting training from iteration {}".format(start_iter))

    # [PHAT]: Create a log file
    log_file = open(cfg.MY_CUSTOM.LOG_FILE, 'w')

    best_loss = 1e6
    count_not_improve = 0
    train_size = 2177
    epoch_size = int(train_size/cfg.SOLVER.IMS_PER_BATCH)
    n_early_epoch = 10

    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)     
            losses = sum(loss for loss in loss_dict.values())

            assert torch.isfinite(losses).all(), loss_dict

            # Update loss dict
            loss_dict_reduced = {
                k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)


            # Early stopping
            if (iteration > start_iter) and ((iteration-start_iter) % epoch_size == 0):
                val_loss = do_val(cfg, model, val_dataloader)
                
                if val_loss >= best_loss:
                    count_not_improve += 1
                    # stop if models doesn't improve after <n_early_epoch> epoch
                    if count_not_improve == epoch_size*n_early_epoch:
                        break
                else:
                    count_not_improve = 0
                    best_loss = val_loss
                    periodic_checkpointer.save("best_model_early")
                
                # print(f"epoch {iteration//epoch_size}, val_loss: {val_loss}")
                log_file.write(f"Epoch {(iteration-start_iter)//epoch_size}, val_loss: {val_loss}\n")
                comm.synchronize()
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            lr = optimizer.param_groups[0]["lr"]
            storage.put_scalar("lr", lr, smoothing_hint=False)
            scheduler.step()


            if iteration - start_iter > 5 and ((iteration-start_iter) % print_every == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
                
                # Write my log
                log_file.write(f"[iter {iteration}, best_loss: {best_loss}] total_loss: {losses}, lr: {lr}\n")
            
            periodic_checkpointer.step(iteration)
    
    log_file.close()

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.IMS_PER_BATCH = 2
    # cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    # cfg.SOLVER.MAX_ITER = 200
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.FCOS.NUM_CLASSES = 5
    cfg.MY_CUSTOM.LOG_FILE = os.path.join(cfg.OUTPUT_DIR, 'my_log.txt')

    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg

def main(args):
    cfg = setup(args)
    model = build_model(cfg)

    register_coco_instances("firevysor_train", {}, "data/Split_CleanedImage/train_annot.json", "data/Split_CleanedImage/train")
    register_coco_instances("firevysor_val", {}, "data/Split_CleanedImage/val_annot.json", "data/Split_CleanedImage/val")
    register_coco_instances("hardcases_val", {}, "data/annotations/hard_cases.json", "data/hard_cases")

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, val_set='firevysor_val')
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )



