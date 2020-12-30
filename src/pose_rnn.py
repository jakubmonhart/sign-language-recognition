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

class PoseRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        INPUT_SIZE = 248
        self.gru = nn.GRU(input_size=INPUT_SIZE, hidden_size=args.gru_hidden_size, num_layers=2) 

        # Fully connected layer for classificator
        self.fc = nn.Sequential([nn.Linear(args.gru_hidden_size, args.gru_hidden_size), nn.Linear(args.gru_hidden_size, min(MAX_NUM_CLASSES, args.subset))]
        
    def forward(self, x):
        '''
        Input dimension: (B x T x N)
            - B: batch size
            - T: number of consecutive frames o video
            - N: number of keypoint coordinates
        '''

        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        
        # GRU input dimension needs to be (seq_len, batch, input_size)
        x = x.permute(1,0,2)
    
        # Only need first output
        o = self.gru(x)[0]
        # # print(o.shape)

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
    run = wandb.init(name=args.save_dir[len('../runs/'):], config = args, project='sign-language-recognition')

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
    real_batch_size = args.batch_size 
    json_file = os.path.join(args.data_path, 'WLASL_v0.3.json')
    videos_folder = os.path.join(args.data_path, 'videos')
    keypoints_folder = os.path.join(args.data_path, 'keypoints')
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224)])
    val_transforms = train_transforms
   
    # Debug data
    if args.debug_dataset: 
        train_dataset = WLASL(json_file=json_file, videos_folder=videos_folder,
                              keypoints_folder=keypoints_folder,
                              transforms=train_transforms, split='train', subset=args.subset,
                              keypoints=True)
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=real_batch_size, sampler=DebugSampler(args.debug_dataset, len(train_dataset)))
        val_dl = train_dl
    else:    
        train_dataset = WLASL(json_file=json_file, videos_folder=videos_folder,
                              keypoints_folder=keypoints_folder,
                              transforms=train_transforms, split='train', subset=args.subset,
                              keypoints=True)
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=real_batch_size, shuffle=True)

        val_dataset = WLASL(json_file=json_file, videos_folder=videos_folder,
                              keypoints_folder=keypoints_folder,
                              transforms=val_transforms, split='val', subset=args.subset,
                              keypoints=True)
        val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=real_batch_size, shuffle=True)
    logger.info('data loaded')
    logger.info(f'size of train dataset: {len(train_dataset)}')
    logger.info(f'size of validation dataset: {len(val_dataset)}')
    
    # Model, loss, optimizer
    m = PoseRNN(args).to(device)
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


# ********* Evaluation functions **********

def test(model_path, device):

  # Test dataset
  subset = 100
  data_path = '../data'
  json_file = os.path.join(data_path, 'WLASL_v0.3.json')
  videos_folder = os.path.join(data_path, 'videos')
  keypoints_folder = os.path.join(data_path, 'keypoints')
 
  test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
  test_dataset = WLASL(json_file=json_file, videos_folder=videos_folder,
                        keypoints_folder=keypoints_folder,
                        transforms=test_transforms, split='test', subset=subset)
  test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
  print('data loaded')
  print(f'len of test dataset: {len(test_dataset)}')
  
  # Load model
  m = load_model(model_path).to(device)

  with torch.no_grad():
    top1 = AverageMeter()
    top5 = AverageMeter()
    top10 = AverageMeter()
    
    top1_per_class  = {}
    top5_per_class  = {}
    top10_per_class  = {}

    for i in range(subset):
      top1_per_class[i] = AverageMeter()
      top5_per_class[i] = AverageMeter()
      top10_per_class[i] = AverageMeter()

    m.eval()
    for batch in tqdm(test_dl):
      X = batch['X'].to(device)
      label = batch['label'].to(device)
      
      # [per frame logits, mean of all frames logits]
      logits = m(X)

      # Update metrics
      acc1, acc5, acc10 = topk_accuracy(logits[:,-1], label[0], topk=(1,5,10))
      top1.update(acc1.item())
      top1_per_class[label[0].item()].update(acc1.item())
      top5.update(acc5.item())
      top5_per_class[label[0].item()].update(acc5.item())
      top10.update(acc10.item())
  

def load_model(model_path):
  with open(os.path.join(model_path, 'args.json')) as f:
    args = argparse.Namespace(**json.load(f))
  m = Conv2dRNN(args)

  checkpoint = torch.load(os.path.join(model_path, 'checkpoint.pt.tar'), map_location=torch.device('cpu'))
  m.load_state_dict(checkpoint['model'])
  return m


def predict_video(model_path, video_path, device):

  # Load model
  m = load_model(model_path).to(device)
  
  # Load rgb frames from video
  frames = load_rgb_frames_from_video(video_path, 0, -1, True)
  
  crop = videotransforms.CenterCrop(224)
  frames = video_to_tensor(crop(frames))

  logits = m(frames.unsqueeze(0).to(device))

  return logits[0,-1]


# ************ main *********
def main():
    # Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--gru_hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--freeze_vgg', default=False, action='store_true')
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--save_dir', type=str, default='../runs/default')
    parser.add_argument('--subset', type=int, default=100)
    parser.add_argument('--resume_train', default=False, action='store_true')
    parser.add_argument('--debug_dataset', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--use_lr_scheduler', default=False, action='store_true')
    parser.add_argument('--lr_schedule_patience', type=int, default=5)
    parser.add_argument('--lr_schedule_factor', type=float, default=0.5)
    parser.add_argument('--lr_schedule_threshold', type=float, default=1e-4)
    
    args, _  = parser.parse_known_args() 
    print(args)
    train(args)

   

if __name__ == '__main__':
    main()
