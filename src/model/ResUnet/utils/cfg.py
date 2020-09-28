from yacs.config import CfgNode as CN


def get_default_config():
    cfg = CN()

    # define your config content here
    cfg.DATA = CN()
    cfg.SOLVER = CN()
    cfg.MODEL = CN()

    # ---------------------- COMMON CONFIG ------------------------
    cfg.OUTPUT_DIR = ''
    cfg.CHECKPOINT_PATH = None

    # ---------------------- DATA CONFIG ------------------------
    cfg.DATA.ROOT_DIR = ''
    cfg.DATA.TRAIN = ''
    cfg.DATA.TRAIN_IMAGES = ''
    cfg.DATA.TRAIN_MASKS = ''
    cfg.DATA.VAL = None

    # ---------------------- MODEL CONFIG ------------------------
    cfg.MODEL.NAME = 'res_unet_plus'
    cfg.MODEL.IMAGE_SIZE = (512, 512)
    

    # ---------------------- SOLVER CONFIG ------------------------
    cfg.SOLVER.EPOCH = 20000
    cfg.SOLVER.BATCH_SIZE = 4
    cfg.SOLVER.EARLY_STOPPING = 10
    cfg.SOLVER.VALIDATION_EVERY = -1 # 
    cfg.SOLVER.LOGGING_STEP = 50
    cfg.SOLVER.LR = 1e-3

    return cfg

