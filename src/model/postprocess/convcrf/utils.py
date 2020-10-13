from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp

import logging
from model.ResUnet.convcrf import convcrf

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


if __name__ == '__main__':
    logging.info("Hello World.")



def setup_convcrf(cfg):
    config  = convcrf.get_default_conf()
    num_classes = 1
    config['filter_size'] = 7
    config['pyinn'] = False
    # config['col_feats']['schan'] = 0.1 # as we are using normalized images
    shape = cfg.MODEL.IMAGE_SIZE

    model = convcrf.GaussCRF(conf=config, shape=shape, nclasses=num_classes)
    model.cuda()

    return model

def _get_simple_unary(batched=False):
    unary1 = np.zeros((10, 10), dtype=np.float32)
    unary1[:, [0, -1]] = unary1[[0, -1], :] = 1

    unary2 = np.zeros((10, 10), dtype=np.float32)
    unary2[4:7, 4:7] = 1

    unary = np.vstack([unary1.flat, unary2.flat])
    unary = (unary + 1) / (np.sum(unary, axis=0) + 2)

    if batched:
        unary = unary.reshape(tuple([1]) + unary)

    return unary


def _get_simple_img(batched=False):

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[2:8, 2:8, :] = 255

    if batched:
        img = img.reshape(tuple([1]) + img)

    return img
