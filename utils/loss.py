import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg = vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:18])
    
    def forward(self, x):
        x = self.feature_extractor(x)
        return x