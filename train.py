import torch
import torch.nn as nn

def train_model(dataloader, generator, discriminator, feature_extractor, epochs, learning_rate):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    generator.to(device)
    discriminator.to(device)
    feature_extractor.to(device)
    feature_extractor.eval()
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        for imgs in dataloader:
            
            LR = imgs['LR']
            HR = imgs['HR']
            
            # discriminator labels
            real = torch.tensor(()).new_ones((LR.size()[0], *discriminator.output_shape), requires_grad=False)
            fake = torch.tensor(()).new_zeros((LR.size()[0], *discriminator.output_shape), requires_grad=False)

            
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

    return generator, discriminator