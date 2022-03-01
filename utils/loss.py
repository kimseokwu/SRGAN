import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = vgg16(pretrained=True)
    
    def forward(self, x):
        pass