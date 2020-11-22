from torch import nn
from torchvision import models

class Conv2dRNN(nn.Module):
    def __init__(self):
        super().__init__()
        # We want to embed frames of videos, get rid of classifier part of vgg16 and only use features.
        self.vgg16 = models.vgg16(pretrained=True, progress=True).features
        
        # Size of concatenated output of vgg16 features for (3x224x224) rgb image.
        OUT_DIM = 25088
        
        # Init RNN part.
        self.gru = nn.GRU(input_size=OUT_DIM, hidden_size=256, num_layers=2) 
        
    def forward(self, x):
        # Input with dimension (batch_size, seq_len: 50 (might change - number of consecutive frames from video), rgb: 3, W: 224, H: 224)
        
        # Convnet need input of size(batch_size, W, H)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.reshape(batch_size*seq_len, *x.shape[2:])
        o = self.vgg16(x)
        
        # GRU needs input of shape (seq_len, batch, input_size)
        o = o.reshape(batch_size, seq_len, -1)
        o = o.permute(1,0,2)

        return self.gru(o)