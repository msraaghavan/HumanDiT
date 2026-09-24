import torch
import torch.nn as nn
import torch.nn.functional as F

class HLCGenerator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64):
        super(HLCGenerator, self).__init__()
        
        # Initial dense layer
        self.dense = nn.Linear(latent_dim, 4 * 4 * ngf * 8)
        
        # Main convolutional blocks
        self.conv_blocks = nn.ModuleList([
            # Block 1: 4x4 -> 8x8
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True)
            ),
            # Block 2: 8x8 -> 16x16
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True)
            ),
            # Block 3: 16x16 -> 32x32
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True)
            ),
            # Block 4: 32x32 -> 64x64
            nn.Sequential(
                nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        ])
        
        # Latency-consistent skip connections
        self.skip_connections = nn.ModuleList([
            nn.Conv2d(ngf * 4, ngf * 2, 1),
            nn.Conv2d(ngf * 2, ngf, 1),
            nn.Conv2d(ngf, 3, 1)
        ])
        
    def forward(self, z):
        # Initial dense layer
        x = self.dense(z)
        x = x.view(-1, 512, 4, 4)
        
        # Process through conv blocks with skip connections
        skip_features = []
        for i, block in enumerate(self.conv_blocks[:-1]):
            x = block(x)
            skip_features.append(x)
        
        # Final block
        x = self.conv_blocks[-1](x)
        
        # Add skip connections
        for i, skip in enumerate(skip_features):
            skip = F.interpolate(skip, size=x.shape[2:])
            skip = self.skip_connections[i](skip)
            x = x + skip
            
        return x

class HLCDiscriminator(nn.Module):
    def __init__(self, ndf=64):
        super(HLCDiscriminator, self).__init__()
        
        # Initial conv block
        self.initial = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Main conv blocks
        self.conv_blocks = nn.ModuleList([
            # Block 1: 32x32
            nn.Sequential(
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Block 2: 16x16
            nn.Sequential(
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Block 3: 8x8
            nn.Sequential(
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True)
            )
        ])
        
        # Final layers
        self.final = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
        # Latency-consistent attention
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ndf * 2, 1, 1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                nn.Conv2d(ndf * 4, 1, 1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                nn.Conv2d(ndf * 8, 1, 1),
                nn.Sigmoid()
            )
        ])
        
    def forward(self, x):
        x = self.initial(x)
        
        # Process through conv blocks with attention
        for i, block in enumerate(self.conv_blocks):
            x = block(x)
            # Apply attention
            att = self.attention[i](x)
            x = x * att
            
        x = self.final(x)
        return x.view(-1, 1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Example usage
if __name__ == "__main__":
    # Test the models
    latent_dim = 100
    batch_size = 4
    
    # Create models
    generator = HLCGenerator(latent_dim)
    discriminator = HLCDiscriminator()
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Test forward pass
    z = torch.randn(batch_size, latent_dim)
    fake_images = generator(z)
    output = discriminator(fake_images)
    
    print(f"Generator output shape: {fake_images.shape}")
    print(f"Discriminator output shape: {output.shape}") 