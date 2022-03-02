from torch.utils.data import DataLoader

import utils.dataloader as dataloader
import utils.loss as loss
import model
from train import train_model

FILE_PATH = './image'
EPOCHS = 1
BATCH_SIZE = 1
lr = 0.001

# initiate model

discriminator =  model.Discriminator()
generator = model.Generator()
feature_extractor = loss.VGG()

# load dataloader
train_datasets = dataloader.ImageDataset(FILE_PATH, (512, 512))
train_data_loader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

if __name__ == '__main__':
    generator, discriminator = train_model(train_data_loader, generator, discriminator, feature_extractor, EPOCHS, lr )
