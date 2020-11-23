import torch
from torch import nn
from torchvision import models, transforms

import os
import argparse
import json
import numpy as np

from utils import create_logger, get_free_gpu, get_device
from data import WLASL
import videotransforms
from metrics import AverageMeter, topk_accuracy

class Conv2dRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        # We want to embed frames of videos, get rid of classifier part of vgg16 and only use features.
        self.vgg16 = models.vgg16(pretrained=True, progress=True).features
        
        # Size of concatenated output of vgg16 features for (3x224x224) rgb image.
        OUT_DIM = 25088
        
        # Init RNN part.
        self.gru = nn.GRU(input_size=OUT_DIM, hidden_size=args.gru_hidden_size, num_layers=2) 

        # Fully connected layer for classificator
        self.fc = nn.Linear(args.gru_hidden_size, args.num_classes)
        
    def forward(self, x):
        '''
        Input dimension: (B x C x T x H x W)
            - B: batch size
            - C: number of channels, 3 (rgb)
            - T: number of consecutive frames o video
            - HxW: dimension of image (224x224) after crop
        '''
        
        # Convnet need input of size(N, C, H, W), merge T and B to N
        batch_size = x.shape[0]
        seq_len = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch_size*seq_len, *x.shape[2:])
        
        o = self.vgg16(x)

        # GRU needs input of shape (seq_len, batch, input_size)
        o = o.reshape(batch_size, seq_len, -1)
        o = o.permute(1,0,2)
        
        # Only need first output
        o = self.gru(o)[0]

        # Fully connected layer for classification
        o = o.reshape(seq_len*batch_size, self.gru.hidden_size)
        o = self.fc(o)

        o = o.reshape(seq_len, batch_size, self.fc.out_features)
        o = o.permute(1,0,2)

        # Return outputs at each time step and mean of all outputs
        o = torch.cat([o, torch.mean(o, dim=1, keepdim=True)], dim=1) 

        return o


# ***** Functions for train *****

def train(args):
    # Create directory for model checkpoints and log
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=2)

    # Logger
    logger = create_logger(args.save_dir)

    # Set gpu
    if torch.cuda.is_available():
        i = get_free_gpu()
        device = get_device(gpu=i)
    else:
        device = 'cpu'
    logger.info('using device: {}'.format(device))

    # Data
    json_file = os.path.join(args.data_path, 'WLASL_v0.3.json')
    videos_path = os.path.join(args.data_path, 'videos')
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224)])
    val_transforms = train_transforms

    train_dataset = WLASL(json_file=json_file, videos_path=videos_path,
                          transforms=train_transforms, split='train')
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = WLASL(json_file=json_file, videos_path=videos_path,
                          transforms=val_transforms, split='val')
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Model, loss, optimizer
    m = Conv2dRNN(args).to(device)
    optimizer = torch.optim.Adam(m.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Print args
    logger.info('using args: \n' + json.dumps(vars(args), sort_keys=True, indent=2))

    # Train loop
    for t in range(args.n_epochs):

        # Train
        losses = AverageMeter()
        m.train()
        for batch in train_dl:
            X = batch['X'].to(device)
            label = batch['label'].to(device)
            
            # [per frame logits, mean of all frames logits]
            logits = m(X)
            
            # Create label for each logit
            label = torch.cat([l.repeat(logits.shape[1],1) for l in label], dim=0)
            
            # Squeeze time sequence and batch into one dimension
            logits = logits.reshape(logits.shape[0]*logits.shape[1], logits.shape[2])
            
            loss = criterion(logits, label.squeeze())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.update(loss.item())

        train_loss = losses.avg

        # Validate
        with torch.no_grad():
            top1 = AverageMeter()
            top5 = AverageMeter()
            top10 = AverageMeter()
            losses = AverageMeter()
            
            m.eval()
            for batch in val_dl:
                X = batch['X'].to(device)
                label = batch['label'].to(device)
                
                # [per frame logits, mean of all frames logits]
                logits = m(X)

                # Create label for each logit
                label = torch.cat([l.repeat(logits.shape[1],1) for l in label], dim=0)
                
                # Squeeze time sequence and batch into one dimension
                logits = logits.reshape(logits.shape[0]*logits.shape[1], logits.shape[2])

                losses.update(criterion(logits, label.squeeze()).item())

                # Update metrics
                acc1, acc5, acc10 = topk_accuracy(logits, label, topk=(1,5,10))
                top1.update(acc1.item())
                top5.update(acc5.item())
                top10.update(acc10.item())
            
            val_loss = losses.avg

        # Log info
        logger.info('epoch: {} train loss: {:.4f} val loss: {:.4f} top1acc {:.4f} top5acc {:.4f} top10acc {:.4f}'.format(t+1,
            train_loss, val_loss, top1.avg, top5.avg, top10.avg))


def main():
    # Arguments
    args, _ =  argparse.ArgumentParser().parse_known_args()

    args.n_epochs = 10
    args.num_classes = 2000
    args.gru_hidden_size = 256
    args.lr = 1e-3
    args.batch_size = 2

    args.data_path = '../data/sample-data'
    args.save_dir = '../runs/conv2d-rnn'

    train(args)

   

if __name__ == '__main__':
    main()