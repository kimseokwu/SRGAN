from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms


def train_model(dataloader, generator, discriminator, feature_extractor, epochs, learning_rate=1e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    generator.to(device)
    discriminator.to(device)
    feature_extractor.to(device)
    feature_extractor.eval()
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        for i, imgs in enumerate(dataloader):
            
            LR = imgs['LR'].to(device)
            HR = imgs['HR'].to(device)
            
            # discriminator labels
            real = torch.tensor(()).new_ones((LR.size()[0], *discriminator.output_shape), requires_grad=False).to(device)
            fake = torch.tensor(()).new_zeros((LR.size()[0], *discriminator.output_shape), requires_grad=False).to(device)

            
            # Generator 학습
            optimizer_G.zero_grad()
            generated_image = generator(LR)
            
            # content loss
            gen_features = feature_extractor(generated_image)
            real_features = feature_extractor(HR)
            content_criterion = nn.L1Loss()
            content_loss = content_criterion(gen_features, real_features.detach())
            
            # Adversarial loss
            adv_criterion = nn.MSELoss()
            adv_loss = adv_criterion(discriminator(generated_image), real)
                        
            # Generator loss
            generator_loss = 1e-3 * content_loss + adv_loss
            generator_loss.backward()
            optimizer_G.step()
            
            # Discriminator 학습
            optimizer_D.zero_grad()
            
            discriminator_criterion = nn.MSELoss()
            real_loss = discriminator_criterion(discriminator(HR), real) 
            fake_loss = discriminator_criterion(discriminator(generated_image.detach()), fake)
            
            discriminator_loss = (real_loss + fake_loss) / 2
            discriminator_loss.backward()
            optimizer_D.step()
            
            # print loss
            
            print(f'EPOCH: {epoch + 1}, BATCH: {i + 1}, Discriminator loss: {discriminator_loss.item()}, Generator loss: {generator_loss.item()}')
            
            if epoch % 5 == 0:
                torch.save(generator.state_dict(), f'saved_model/generator_{epoch + 1}.pth')
                torch.save(discriminator.state_dict(), f'saved_model/discriminator_{epoch + 1}.pth')


    return generator, discriminator

def super_resolution(img_path, generator):
    generator.eval()
    img = Image.open(img_path)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    LR_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    img = LR_transform(img)
    
    with torch.no_grad():
        generated_image = generator(img)
    
    return generated_image