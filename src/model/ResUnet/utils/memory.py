import torch


def free_gpu_memory():
    '''
        Free GPU memory
    '''
    import gc
    gc.collect()
    torch.cuda.empty_cache()

