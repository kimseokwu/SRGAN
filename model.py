import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.input = nn.Sequential(nn.Conv2D(3, 64, kernel_size=9),
                                   nn.PReLU(inplace=True)
                                  )
        
        self.resid_block1 = nn.Sequential(nn.Conv2D(64, 64, kernel_size=3),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(inplace=True),
                                         nn.Conv2D(64, 64, kernel_size=3),
                                         nn.BatchNorm2d(64))
        
        self.resid_block2 = nn.Sequential(nn.Conv2D(64, 64, kernel_size=3),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(inplace=True),
                                         nn.Conv2D(64, 64, kernel_size=3),
                                         nn.BatchNorm2d(64))
        
        self.resid_block3 = nn.Sequential(nn.Conv2D(64, 64, kernel_size=3),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(inplace=True),
                                         nn.Conv2D(64, 64, kernel_size=3),
                                         nn.BatchNorm2d(64))
        
        self.resid_block4 = nn.Sequential(nn.Conv2D(64, 64, kernel_size=3),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(inplace=True),
                                         nn.Conv2D(64, 64, kernel_size=3),
                                         nn.BatchNorm2d(64))
        
        self.resid_block5 = nn.Sequential(nn.Conv2D(64, 64, kernel_size=3),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(inplace=True),
                                         nn.Conv2D(64, 64, kernel_size=3),
                                         nn.BatchNorm2d(64))
        
        self.output_layer1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3),
                                          nn.BatchNorm2d(64))
        
        self.output_layer2 = nn.Sequential(nn.Conv2d(64, 256, kernel_size=3),
                                          nn.PixelShuffle(),
                                          nn.PReLU(inplace=True),
                                          nn.Conv2d(256, 256, kernel_size=3),
                                          nn.PixelShuffle(),
                                          nn.PReLU(inplace=True),
                                          nn.Conv2D(256, 3, kernel_size=3))

    
    def forward(self, x):
        x = self.input(x)
        
        return x
    
    
class Discriminator(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, x):
        return x