import torch
from model import SRGAN

def train_model(dataloader, generator, discriminator, epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    pass