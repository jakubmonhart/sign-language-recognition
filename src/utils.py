import logging
import os
import sys
import numpy as np

import torch

def create_logger(save_dir):
    log_path = os.path.join(save_dir, 'out.log')
    logger = logging.getLogger('training')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    index = np.argmax(memory_available[:-1])  # Skip the 7th card --- it is reserved for evaluation!!!
 
    return index   # Returns index of the gpu with the most memory available


def get_device(gpu=0):   # Manually specify gpu
    if torch.cuda.is_available():
        device = torch.device(gpu)
    else:
        device='cpu'
 
    return device