import torch
from torch import nn
from torchvision import models, transforms

import os
import argparse
import json
import numpy as np
import time

from utils import create_logger, get_free_gpu, get_device, save_best
from data import WLASL, DebugSampler
import videotransforms
from metrics import AverageMeter, topk_accuracy

data_path = '../data'
subset = 300

real_batch_size = 2 # can't fit more into gpu memory  

json_file = os.path.join(data_path, 'WLASL_v0.3.json')
videos_path = os.path.join(data_path, 'videos')
train_transforms = transforms.Compose([videotransforms.RandomCrop(224)])
val_transforms = train_transforms

train_dataset = WLASL(json_file=json_file, videos_path=videos_path,
                      transforms=train_transforms, split='train', subset=subset)
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=real_batch_size, shuffle=True)

val_dataset = WLASL(json_file=json_file, videos_path=videos_path,
                      transforms=val_transforms, split='val', subset=subset)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=real_batch_size, shuffle=True)

print('dataset loaded')
print('loading each video')

for batch in train_dl:
  pass
  
print('loading done')  
