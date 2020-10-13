import numpy as np 

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral

def get_dcrf_model(img_shape, n_labels=2):
    '''
    Args:
        img_shape: (height, width)
    '''
    return dcrf.DenseCRF2D(img_shape[1], img_shape[0], n_labels)

def apply_dcrf_model(image, mask, model, n_steps=5):
    '''
    Args:
        image: original image, np.array of shape [H, W, C]
        masks: predicted mask, np.array of shape [H, W, C], here C = 1
            --> should be thresholding first
    Output:
        refined_mask: np.array of shape [H, W, C]
    '''
    annotated_label = mask.astype(np.int32)

    ## Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)
    n_labels = 2

    ## Setting up the CRF model
    # d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], n_labels)

    ## Get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    model.setUnaryEnergy(U)

    ## This adds the color-independent term, features are the locations only.
    model.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    ## Run Inference for 10 steps
    Q = model.inference(n_steps)

    ## Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((image.shape[0], image.shape[1]))