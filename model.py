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
        r1 = self.resid_block1(x) + x
        r2 = self.resid_block2(r1) + r1
        r3 = self.resid_block3(r2) + r2
        r4 = self.resid_block4(r3) + r3
        r5 = self.resid_block5(r4) + r4
        output = self.output_layer1(r5) + x
        output = self.output_layer2(output)
        
        return x
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1),
                                         nn.LeakyReLU(inplace=True))
        
        self.conv_block1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1),
                                        nn.BatchNorm2d(128),
                                        nn.LeakyReLU(inplace=True))
        
        self.conv_block2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2),
                                         nn.BatchNorm2d(128),
                                         nn.LeakyReLU(inplace=True))
        
        self.conv_block3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1),
                                         nn.BatchNorm2d(256),
                                         nn.LeakyReLU(inplace=True))
        
        self.conv_block4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=2),
                                         nn.BatchNorm2d(256),
                                         nn.LeakyReLU(inplace=True))
        
        self.conv_block5 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                         nn.BatchNorm2d(512),
                                         nn.LeakyReLU(inplace=True))
        
        self.conv_block6 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2),
                                         nn.BatchNorm2d(512),
                                         nn.LeakyReLU(inplace=True))
        
        self.output_layer = nn.Sequential(nn.Linear(512*6*6, 1024),
                                          nn.LeakyReLU(inplace=True),
                                          nn.Linear(1024, 1),
                                          nn.Sigmoid())
    
    def forward(self, x):
        
        x = self.input_layer(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = self.output_layer(x)
        
        return x
    
class SRGAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(SRGAN, self).__init__()
        self.G = generator
        self.D = discriminator
    
    def forward(self, x):
        generated_image = self.G(x)
        result = self.D(result)
        return result
        