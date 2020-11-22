import json
import pandas as pd
import os
import cv2
import numpy as np
import torch
import random
import math

from torch.utils.data import Dataset


def get_class_list(json_file):
    with open(json_file) as ipf:
        content = json.load(ipf)
    
    class_list = []
    for entry in content:
        class_list.append(entry['gloss'])
        
    return class_list


def load_rgb_frames_from_video(video_path, start_f, num_f):

    vidcap = cv2.VideoCapture(video_path)

    frames = []

    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    for offset in range(min(num_f, int(total_frames - start_f))):
        success, img = vidcap.read()
        
        if not success:
            print('{} unsuccesfull read at frame: {}/{}'.format(
                video_path ,offset-start_f, min(num_f, int(total_frames - start_f))))
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
    def __init__(self, json_file='data/WLASL_v0.3.json', videos_path='data/sample-videos', split='train', transforms=None):
        self.class_list = get_class_list(json_file)
        self.num_classes = len(self.class_list)
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        dataset = {'video_path': [], 'label': [], 'bounding_box': [], 'num_frames': []}

        for class_id in range(len(data)):
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
        
        # TODO - resize the frames so that bounding box is in center and 256 pixels in diagonal, crop rest of the image?
        imgs = load_rgb_frames_from_video(video_path, start_f, total_frames)
        
        if self.transforms is not None:
            imgs = self.transforms(imgs)
        
        # Pad frames if video is shorter than total_frames
        imgs = self.pad(imgs, total_frames)
        
        ret_img = video_to_tensor(imgs)
        return {'X': ret_img, 'label': torch.LongTensor([label])}
    
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