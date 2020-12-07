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

import wandb

MAX_NUM_CLASSES = 2000

class Conv2dRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        # We want to embed frames of videos, get rid of classifier part of vgg16 and only use features.
        self.vgg16 = models.vgg16(pretrained=True, progress=True).features
        
        # Freeze training of pretrained vgg
        if args.freeze_vgg:
            for param in self.vgg16.parameters():
                param.require_grad = False
        
        # Size of concatenated output of vgg16 features for (3x224x224) rgb image.
        OUT_DIM = 25088
        
        # Init RNN part.
        self.gru = nn.GRU(input_size=OUT_DIM, hidden_size=args.gru_hidden_size, num_layers=2) 

        # Fully connected layer for classificator
        self.fc = nn.Linear(args.gru_hidden_size, min(MAX_NUM_CLASSES, args.subset))
        
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
        # print(o.shape)

        # GRU needs input of shape (seq_len, batch, input_size)
        o = o.reshape(batch_size, seq_len, -1)
        # print(o.shape)
        o = o.permute(1,0,2)
        
        # Only need first output
        o = self.gru(o)[0]
        # print(o.shape)

        # Fully connected layer for classification
        o = o.reshape(seq_len*batch_size, self.gru.hidden_size)
        o = self.fc(o)

        o = o.reshape(seq_len, batch_size, self.fc.out_features)
        o = o.permute(1,0,2)
        # print(o.shape)
        # Return outputs at each time step and mean of all outputs
        o = torch.cat([o, torch.mean(o, dim=1, keepdim=True)], dim=1) 
        # print(o.shape)

        return o


# ***** Functions for train *****

def train(args):
    # Init wandb
    run = wandb.init(config = args, project='sign-language-recognition')

    # Create directory for model checkpoints and log
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # Save args    
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
    
    # Prepare early stop
    stopped = False
    best_epoch = 0
    best_loss = torch.Tensor([float('Inf')])

    # Data
    
    real_batch_size = 2 # can't fit more into gpu memory  
    
    json_file = os.path.join(args.data_path, 'WLASL_v0.3.json')
    videos_path = os.path.join(args.data_path, 'videos')
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224)])
    val_transforms = train_transforms
   
    # Debug data
    if args.debug_dataset: 
        train_dataset = WLASL(json_file=json_file, videos_path=videos_path,
                              transforms=train_transforms, split='train', subset=args.subset)
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=real_batch_size, sampler=DebugSampler(args.debug_dataset, len(train_dataset)))
        val_dl = train_dl
    else:    
        train_dataset = WLASL(json_file=json_file, videos_path=videos_path,
                              transforms=train_transforms, split='train', subset=args.subset)
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=real_batch_size, shuffle=True)

        val_dataset = WLASL(json_file=json_file, videos_path=videos_path,
                              transforms=val_transforms, split='val', subset=args.subset)
        val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=real_batch_size, shuffle=True)
    logger.info('data loaded')
    
    # Model, loss, optimizer
    m = Conv2dRNN(args).to(device)
    optimizer = torch.optim.Adam(m.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Resume train
    start_epoch = 0
    if args.resume_train:
        checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint.pt.tar'), map_location=torch.device('cpu'))
        best_epoch = checkpoint['epoch']
        m.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        m = m.to(device)
        best_loss = checkpoint['best_val_loss']
        start_epoch = best_epoch + 1
        
        # Change learning rate
        for g in optimizer.param_groups:
            g['lr'] = args.lr
    
        logger.info('Resuming training from epoch {} with best loss {:.4f}'.format(
            start_epoch, best_loss))

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_schedule_factor, patience=args.lr_schedule_patience, threshold=args.lr_schedule_threshold)

    # Watch model with wandb
    run.watch(m, log='all', log_freq=5)

    # Print args
    logger.info('using args: \n' + json.dumps(vars(args), sort_keys=True, indent=2))

    # Train loop
    for t in range(args.n_epochs):
        t+=start_epoch
        # Train
        losses = AverageMeter()
        batch_time = AverageMeter()
        m.train()
        
        start_t = time.time()
        for i, batch in enumerate(train_dl):
          
            # Run the forward pass multiple times and accumulate gradient (to be able to use large batch size)
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
            losses.update(loss.item())

            if (i%(args.batch_size//real_batch_size)) == 0:
                # Optimize with accumulated gradient
                optimizer.step()
                optimizer.zero_grad()

                batch_time.update(time.time()-start_t)
                start_t = time.time()

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
            
             
            # Save best model
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, t
                save_best(args, t, m, optimizer, best_loss)
            
            # Check early stop
            if t >= best_epoch + args.early_stop:
                logger.info('EARLY STOP')
                break
            
        # Log info
        logger.info('epoch: {} train loss: {:.4f} val loss: {:.4f} top1acc {:.4f} top5acc {:.4f} top10acc {:.4f} lr: {:.2e} time per batch {:.1f} s'.format(t+1,
            train_loss, val_loss, top1.avg, top5.avg, top10.avg, optimizer.param_groups[0]['lr'], batch_time.avg))

        # Wandb log
        run.log({'train_loss': train_loss, 'val_loss': val_loss, 'top1_acc': top1.avg, 'top5_acc': top5.avg, 'top10_acc': top10.avg, 'lr': optimizer.param_groups[0]['lr']})
        
        # Scheduler step
        if args.use_lr_scheduler:
            scheduler.step(val_loss)


def main():
    # Arguments
    parser = argparse.ArgumentParser()

    # args.n_epochs = 200
    # args.gru_hidden_size = 256
    # args.lr = 1e-2
    # args.batch_size = 2
    # args.freeze_vgg = True

    # args.data_path = '../data'
    # args.save_dir = '../runs/conv2d-rnn_freezed_vgg'
    # args.subset= 100
    # args.resume_train=False
    # args.debug_dataset = 0
    # args.early_stop = 10

    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--gru_hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--freeze_vgg', type=bool, default=False)
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--save_dir', type=str, default='../runs/default')
    parser.add_argument('--subset', type=int, default=100)
    parser.add_argument('--resume_train', type=bool, default=False)
    parser.add_argument('--debug_dataset', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--use_lr_scheduler', type=bool, default=True)
    parser.add_argument('--lr_schedule_patience', type=int, default=5)
    parser.add_argument('--lr_schedule_factor', type=float, default=0.5)
    parser.add_argument('--lr_schedule_threshold', type=float, default=1e-4)
    
    args, _  = parser.parse_known_args() 
    train(args)

   

if __name__ == '__main__':
    main()
