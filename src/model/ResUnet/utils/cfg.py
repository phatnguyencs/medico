from yacs.config import CfgNode as CN


def get_default_config():
    cfg = CN()

    # define your config content here
    cfg.DATA = CN()
    cfg.SOLVER = CN()
    cfg.MODEL = CN()
    cfg.INFERENCE = CN()

    # ---------------------- COMMON CONFIG ------------------------
    cfg.OUTPUT_DIR = ''
    cfg.CHECKPOINT_PATH = ''

    # ---------------------- DATA CONFIG ------------------------
    cfg.DATA.ROOT_DIR = ''
    cfg.DATA.TRAIN = 'train.csv'
    cfg.DATA.TRAIN_IMAGES = ''
    cfg.DATA.TRAIN_MASKS = ''
    cfg.DATA.VAL = 'val.csv'

    # ---------------------- MODEL CONFIG ------------------------
    cfg.MODEL.NAME = 'res_unet_plus'
    cfg.MODEL.IMAGE_SIZE = (512, 512)
    

    # ---------------------- VISUALIZE RESULT --------------------
    cfg.INFERENCE.SAVE_DIR = ''
    cfg.INFERENCE.MASK_THRES = 0.5

    # ---------------------- SOLVER CONFIG ------------------------
    cfg.SOLVER.EPOCH = 100
    cfg.SOLVER.BATCH_SIZE = 4
    cfg.SOLVER.EARLY_STOPPING = 10
    cfg.SOLVER.VALIDATION_EVERY = -1 # 
    cfg.SOLVER.LOGGING_STEP = 10
    cfg.SOLVER.LR = 1e-3

    return cfg

