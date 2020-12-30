import json
import pandas as pd
import os
import cv2
import numpy as np
import torch
import random
import math

from torch.utils.data import Dataset, Sampler


# ********* datasets *********

class WLASL(Dataset):
    def __init__(self, json_file='data/WLASL_v0.3.json', videos_folder='data/sample-videos',
     keypoints_folder = 'data/sample-keypoints', keypoints=False, split='train', subset=2000, 
     transforms=None, verbose=False):
        
        self.class_list = get_class_list(json_file)
        self.num_classes = len(self.class_list)
        self.keypoints = keypoints
        
        self.split = split
        self.rescale_mode = args.rescale_mode

        with open(json_file, 'r') as f:
            data = json.load(f)
        
        dataset = {'video_path': [], 'keypoints_path': [], 'label': [], 'bounding_box': [], 'num_frames': []}

        for class_id in range(min(subset, len(data))):
            # Use only N most occuring glosses (N specified by subset). Default 2000 is total number of glosses in WLASL dataset.
            if class_id > (subset+1):
                break

            for video in data[class_id]['instances']:

                if video['split'] != split:
                    continue

                video_path = os.path.join(videos_folder, video['video_id'] + '.mp4')
                keypoints_path = os.path.join(keypoints_folder, video['video_id'] + '.npz')

                if keypoints:
                    if not os.path.exists(keypoints_path):
                        continue

                else:
                    if not os.path.exists(video_path):
                        continue
                
                num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))

                dataset['label'].append(class_id)
                dataset['video_path'].append(video_path)
                dataset['keypoints_path'].append(keypoints_path)
                dataset['bounding_box'].append(video['bbox'])
                dataset['num_frames'].append(num_frames)
        
        self.data = dataset
        self.transforms = transforms
        self.verbose = verbose

        if self.keypoints:
            self.__getitem__ = self.get_keypoints
        else:
            self.__getitem__ = self.get_video
    
    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, key):
        if self.keypoints:
            return self.get_keypoints(key)
        else:
            return self.get_video(key)


    def get_keypoints(self, key):
        label = self.data['label'][key]
        keypoints_path = self.data['keypoints_path'][key]
        num_frames = self.data['num_frames'][key]

        total_frames = 50

        # Load keypoints
        keypoints = np.load(keypoints_path, allow_pickle=True)


        # print(keypoints_path)
        # print(num_frames)
        if keypoints['pose'].ndim == 1:
          # Keypoints are not concatenated
          # Some data also contain none values - need to drop those

          pose = np.concatenate([k for k in keypoints['pose'] if k is not None], axis=0)
          pose = np.concatenate([pose[:,:8,:2], pose[:,15:19,:2]], axis=1)
         
          face  = np.concatenate([k for k in keypoints['face'] if k is not None], axis=0)[:,:,:2]

          left_hand = np.concatenate([k for k in keypoints['hand'][:,0] if k is not None], axis=0)[:,:,:2]
          right_hand = np.concatenate([k for k in keypoints['hand'][:,1] if k is not None], axis=0)[:,:,:2]

          # print('not concatenated')
          # print(pose.shape)
          # print(face.shape)
          # print(left_hand.shape)
          # print(right_hand.shape)
          
        else:
          pose = np.concatenate([keypoints['pose'][:,:8,:2], keypoints['pose'][:,15:19,:2]], axis=1)
          face = keypoints['face'][:,:,:2]
          left_hand = keypoints['hand'][:,0,:,:2]
          right_hand = keypoints['hand'][:,1,:,:2]

        # Fill in non-detected values
        pose = fill_zeros(pose)
        face = fill_zeros(face)
        left_hand = fill_zeros(left_hand)
        right_hand = fill_zeros(right_hand)

        # Rescale
        pose = rescale(pose, self.rescale_mode)
        face = rescale(face, self.rescale_mode)       
        left_hand = rescale(left_hand, self.rescale_mode)
        right_hand = rescale(right_hand, self.rescale_mode)

        # Concatenate keypoints
        X = [pose.reshape(pose.shape[0], -1),
             face.reshape(face.shape[0], -1),
             left_hand.reshape(left_hand.shape[0], -1),
             right_hand.reshape(right_hand.shape[0], -1)]

        X = np.concatenate(X, axis=1)

        num_frames = X.shape[0]
        # Choose 50 random consecutive frames
        try:
            start_f = random.randint(0, num_frames - total_frames - 1)
        except ValueError:
            start_f = 0

        X = X[start_f:(start_f+50)]

        X = self.pad_keypoints(X, total_frames)

        return {'X': torch.from_numpy(X), 'label': torch.LongTensor([label])}

    def get_video(self, key, test=False):
        '''
        Values of pixels are converted to [-1, 1] in load_rgb_frames_from_video() call.
        '''
        
        label = self.data['label'][key]
        video_path = self.data['video_path'][key]
        num_frames = self.data['num_frames'][key]
        
        total_frames = 50 

        if self.split != 'test':
          # Choose 50 random consecutive frames
          try:
              start_f = random.randint(0, num_frames - total_frames - 1)
          except ValueError:
              start_f = 0
        else:
          # Test - use all frames
          start_f = 0
          total_frames = num_frames
        

        # print('\n********\n')
        # print('video path: {}'.format(video_path))
        # print('Number of frames {}, Start frame: {}'.format(num_frames, start_f))
        # TODO - resize the frames so that bounding box is in center and 256 pixels in diagonal, crop rest of the image?
        imgs = load_rgb_frames_from_video(video_path, start_f, total_frames, self.verbose)
        if len(imgs.shape) < 4:
            print('Wrong format of images.')
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

    def pad_keypoints(self, keypoints, total_frames):
        if keypoints.shape[0] < total_frames:
            num_padding = total_frames - keypoints.shape[0]

            prob = np.random.random_sample()

            # np.concatenate([np.tile(np.expand_dims(X[0], axis=0), (10, 1)), X[0:40]], axis=0).shape

            if prob > 0.5:
                pad = np.tile(np.expand_dims(keypoints[0], axis=0), (num_padding, 1))
                padded_keypoints = np.concatenate([pad, keypoints], axis=0)
            else:
                pad = np.tile(np.expand_dims(keypoints[-1], axis=0), (num_padding, 1))
                padded_keypoints = np.concatenate([keypoints, pad], axis=0)

        else:
            padded_keypoints = keypoints

        return padded_keypoints




# ******* helper functions *********

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

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_f == -1:
      num_f = total_frames

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


# Functions to fill non-detected keypoints
def fill_zeros_with_last(arr):
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]


def fill_zeros(body_part):
    
    for k in range(body_part.shape[1]):
        keypoint = body_part[:,k]

        # x values
        keypoint[:,0] = fill_zeros_with_last(keypoint[:,0])
        keypoint[:,0][::-1] = fill_zeros_with_last(keypoint[:,0][::-1])

        # y values
        keypoint[:,1] = fill_zeros_with_last(keypoint[:,1])
        keypoint[:,1][::-1] = fill_zeros_with_last(keypoint[:,1][::-1])
        body_part[:,k] = keypoint
    
    return body_part


# Functions for rescale
def rescale(body_part, mode):
  if mode == 'separately':
    maxx = body_part[:,:,0].max()
    minx = body_part[:,:,0].min()
    maxy = body_part[:,:,1].max()
    miny = body_part[:,:,1].min()
     
    if (maxx-minx) == 0:
      pass
    else:
      # rescale x
      body_part[:,:,0] = (body_part[:,:,0] - minx)*2/(maxx-minx) - 1
      
    if (maxy-miny) == 0:
      pass
    else:
      # rescale y
      body_part[:,:,1] = (body_part[:,:,1] - miny)*2/(maxy-miny) - 1
    
  elif mode == 'globally':
    body_part = 2*body_part/256.0 - 1
  else:
    pass

  return body_part
    
        


# ******* debug helpers *********
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

