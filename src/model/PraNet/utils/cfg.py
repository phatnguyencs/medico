from yacs.config import CfgNode as CN
import os 
import os.path as osp
from contextlib import redirect_stdout


def save_config(save_dir: str, cfg: CN):
    save_path = osp.join(save_dir, 'config.yml')
    with open(save_path, 'w') as f:
        with redirect_stdout(f): print(cfg.dump())
    

def get_default_config():
    cfg = CN()

    # define your config content here
    cfg.DATA = CN()
    cfg.SOLVER = CN()
    cfg.MODEL = CN()
    cfg.INFERENCE = CN()
    cfg.TRAIN = CN()
    cfg.TRAIN.TSA = CN()

    # ---------------------- TRAIN CONFIG ------------------------
    cfg.TRAIN.AUGMENT = True
    cfg.TRAIN.NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    cfg.TRAIN.NORMALIZE_STD = [0.229, 0.224, 0.225]
    cfg.TRAIN.SCHEDULER_PATIENCE = 10
    cfg.TRAIN.SCHEDULER_FACTOR = 0.1
    cfg.TRAIN.SCHEDULER_MILESTONES_LOW = 150
    cfg.TRAIN.SCHEDULER_MILESTONES_HIGH = 250
    cfg.TRAIN.SIZE_RATES = [1.0]

    # TODO: need to re-assign these params
    cfg.TRAIN.TSA.ALPHA = 5
    cfg.TRAIN.TSA.TEMPERATURE = 5
    cfg.TRAIN.TSA.NUMSTEPS = 20

    # ---------------------- COMMON CONFIG ------------------------
    cfg.OUTPUT_DIR = ''
    cfg.CHECKPOINT_PATH = ''

    # ---------------------- DATA CONFIG ------------------------
    cfg.DATA.ROOT_DIR = ''
    cfg.DATA.TRAIN = 'train.csv'
    cfg.DATA.TRAIN_IMAGES = ''
    cfg.DATA.TEST_IMAGES = ''
    cfg.DATA.TRAIN_MASKS = ''
    cfg.DATA.VAL = 'val.csv'

    # ---------------------- MODEL CONFIG ------------------------
    cfg.MODEL.NAME = 'res_unet_plus'
    cfg.MODEL.IMAGE_SIZE = (512, 512)
    cfg.MODEL.CHANNEL = 3
    cfg.MODEL.BACKBONE = 'resnet50' # or 'resnet101'
    

    # ---------------------- VISUALIZE RESULT --------------------
    cfg.INFERENCE.SAVE_DIR = ''
    cfg.INFERENCE.MASK_THRES = 0.5
    cfg.INFERENCE.BATCH_SIZE = 1
    cfg.INFERENCE.CRF =  False
    cfg.INFERENCE.CRF_STEP = 10
    cfg.INFERENCE.TTA = False

    # ---------------------- SOLVER CONFIG ------------------------
    cfg.SOLVER.EPOCH = 100
    cfg.SOLVER.BATCH_SIZE = 4
    cfg.SOLVER.EARLY_STOPPING = 10
    cfg.SOLVER.VALIDATION_EVERY = -1 # 
    cfg.SOLVER.LOGGING_STEP = 10
    cfg.SOLVER.LR = 1e-3

    return cfg

