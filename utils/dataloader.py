import glob
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# image transform parameter
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class ImageDataset(Dataset):
    def __init__(self, file_path, shape):
        h, w = shape
        
        self.file_list = sorted(glob.glob(file_path + '/*.*'))
        
        self.LR_transform = transforms.Compose([
            transforms.Resize((h // 4, w // 4), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        self.HR_transform = transforms.Compose([
            transforms.Resize((h, w), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        self.LR = []
        self.HR = []
        
        for file in self.file_list:
            img = Image.open(file)
            self.HR.append(self.HR_transform(img))
            self.LR.append(self.LR_transform(img))
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        return {'LR': self.LR[index],
                'HR': self.HR[index]}
        
