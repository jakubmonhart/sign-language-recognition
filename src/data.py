import json
import pandas as pd
import os
import cv2
import numpy as np
import torch
import random
import math

from torch.utils.data import Dataset, Sampler


def get_class_list(json_file):
    with open(json_file) as ipf:
        content = json.load(ipf)
    
    class_list = []
    for entry in content:
        class_list.append(entry['gloss'])
        
    return class_list


def load_rgb_frames_from_video(video_path, start_f, num_f, verbose):

    vidcap = cv2.VideoCapture(video_path)

    frames = []

    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    for offset in range(min(num_f, int(total_frames - start_f))):
        success, img = vidcap.read()
        
        if (not success):
            if verbose:
                print('{} unsuccesfull read at frame: {}/{}'.format(
                    video_path ,offset, min(num_f, int(total_frames - start_f))))
                
            break

        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        if w > 256 or h > 256:
            img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))
    
        # Converts pixel values to [-1, 1]
        img = (img / 255.) * 2 - 1

        frames.append(img)

    vidcap.release() 
    return np.asarray(frames, dtype=np.float32)
 
    
def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


class WLASL(Dataset):
    def __init__(self, json_file='data/WLASL_v0.3.json', videos_path='data/sample-videos', split='train', subset=2000,  transforms=None, verbose=False):
        self.class_list = get_class_list(json_file)
        self.num_classes = len(self.class_list)
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        dataset = {'video_path': [], 'label': [], 'bounding_box': [], 'num_frames': []}

        for class_id in range(min(subset, len(data))):
            # Use only N most occuring glosses (N specified by subset). Default 2000 is total number of glosses in WLASL dataset.
            if class_id > (subset+1):
                break

            for video in data[class_id]['instances']:

                if video['split'] != split:
                    continue

                video_path = os.path.join(videos_path, video['video_id'] + '.mp4')

                if not os.path.exists(video_path):
                    continue
                    
                
                num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))

                dataset['label'].append(class_id)
                dataset['video_path'].append(video_path)
                dataset['bounding_box'].append(video['bbox'])
                dataset['num_frames'].append(num_frames)
        
        self.data = dataset
        self.transforms = transforms
        self.verbose = verbose
    
    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, key):
        '''
        Values of pixels are converted to [-1, 1] in load_rgb_frames_from_video() call.
        '''
        
        label = self.data['label'][key]
        video_path = self.data['video_path'][key]
        num_frames = self.data['num_frames'][key]
        
        total_frames = 50 

        # Choose 50 random consecutive frames
        try:
            start_f = random.randint(0, num_frames - total_frames - 1)
        except ValueError:
            start_f = 0
        

        # print('\n********\n')
        # print('video path: {}'.format(video_path))
        # print('Number of frames {}, Start frame: {}'.format(num_frames, start_f))
        # TODO - resize the frames so that bounding box is in center and 256 pixels in diagonal, crop rest of the image?
        imgs = load_rgb_frames_from_video(video_path, start_f, total_frames, self.verbose)
        if len(imgs.shape) < 4:
            print('Wrong formate of images.')
            print('Path to video is: {}'.format(video_path))
            print('shape of imgs: {}'.format(imgs.shape))
            print('imgs: {}'.format(imgs))
        # imgs = load_rgb_frames_from_video(video_path, start_f, total_frames, True)
        # print('shape loaded imgs: {}'.format(imgs.shape))
        
        if self.transforms is not None:
            imgs = self.transforms(imgs)
        
        # Pad frames if video is shorter than total_frames
        imgs = self.pad(imgs, total_frames)
        # print('shape after pad and transform: {}'.format(imgs.shape))
        
        ret_img = video_to_tensor(imgs)
        return {'X': ret_img, 'label': torch.LongTensor([label])}
        # return [1]

    
    def pad(self, imgs, total_frames):
        if imgs.shape[0] < total_frames:
            num_padding = total_frames - imgs.shape[0]

            if num_padding:
                prob = np.random.random_sample()
                if prob > 0.5:
                    pad_img = imgs[0]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_imgs = np.concatenate([pad, imgs], axis=0)
                else:
                    pad_img = imgs[-1]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_imgs = np.concatenate([imgs, pad], axis=0)
        else:
            padded_imgs = imgs
            
        return padded_imgs
        
        
class DebugSampler(Sampler):
    def __init__(self, n_samples, dataset_len=None):
        self.n_samples = n_samples
        self.dataset_len = dataset_len
        
        if self.dataset_len is  None:
            self.samples = range(self.n_samples)
        else:
            self.samples = random.sample(range(self.dataset_len), self.n_samples)

    def __iter__(self):
        return iter(self.samples)
        
    def __len__(self):
        return self.n_samples

