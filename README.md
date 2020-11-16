# sign-language-recognition

## Dataset
Returns videos as torch.FloatTensor of shape (C x T x H x W)
  - C - number of channels, 3 (rgb)
  - T - number of consecutive frames of video (currently set to constant 50)
  - H x W - height x widht of frames, set to 224 x 224 due to RandomCrop
