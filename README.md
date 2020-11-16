# sign-language-recognition

## Dataset
WLASL dataset: [https://arxiv.org/abs/1910.11006], [https://github.com/dxli94/WLASL]

Returns videos as torch.FloatTensor of shape (C x T x H x W)
  - C - number of channels, 3 (rgb)
  - T - number of consecutive frames of video (currently set to constant 50)
  - H x W - height x widht of frames, set to 224 x 224 due to RandomCrop
  
## I3D model
[https://arxiv.org/abs/1705.07750], [https://github.com/piergiaj/pytorch-i3d]
