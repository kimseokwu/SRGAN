from torch.utils.data import dataset

class ImageDataset(dataset):
    def __init__(self, LR_image, HR_image):
        self.LR = LR_images
        self.HR = HR_images
        data = [(LR, HR) for LR, HR in zip(LR_images, HR_images)]
        
    def __len__(self):
        return len(data)
    
    def __getitem__(self, index):
        return {'LR': data[index][0],
                'HR': data[index][1]}