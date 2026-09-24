# Hybrid Latency-Consistent GAN (HLC-GAN)

This repository contains an implementation of a Hybrid Latency-Consistent GAN (HLC-GAN), which is designed to provide efficient and consistent generation of images while maintaining high quality output.

## Architecture Overview

### Generator (HLCGenerator)

The generator architecture implements several key features for efficient and consistent image generation:

1. **Initial Dense Layer**
   - Transforms the latent vector into a 4x4 feature map
   - Provides a stable starting point for the generation process

2. **Progressive Upsampling**
   - Four convolutional blocks that progressively increase spatial dimensions
   - 4x4 → 8x8 → 16x16 → 32x32 → 64x64
   - Each block includes:
     - Transposed Convolution
     - Batch Normalization
     - ReLU activation

3. **Latency-Consistent Skip Connections**
   - Implements skip connections between different resolution levels
   - Uses 1x1 convolutions for efficient feature processing
   - Helps maintain feature consistency across different scales

### Discriminator (HLCDiscriminator)

The discriminator architecture is designed for efficient feature extraction and classification:

1. **Initial Convolution Block**
   - First downsampling layer
   - LeakyReLU activation for better gradient flow

2. **Progressive Downsampling**
   - Three convolutional blocks for feature extraction
   - Each block includes:
     - Convolution
     - Batch Normalization
     - LeakyReLU activation

3. **Attention Mechanism**
   - Implements attention at each resolution level
   - Helps focus on important features
   - Improves discrimination quality

## Key Features

- **Latency Consistency**: The architecture is designed to maintain consistent processing times across different input sizes
- **Skip Connections**: Helps maintain feature consistency and improve gradient flow
- **Attention Mechanism**: Focuses on important features for better discrimination
- **Efficient Processing**: Uses 1x1 convolutions and optimized layer configurations

## Usage

```python
from hlc_gan import HLCGenerator, HLCDiscriminator, weights_init

# Initialize models
latent_dim = 100
generator = HLCGenerator(latent_dim)
discriminator = HLCDiscriminator()

# Initialize weights
generator.apply(weights_init)
discriminator.apply(weights_init)

# Generate images
z = torch.randn(batch_size, latent_dim)
fake_images = generator(z)

# Discriminate images
output = discriminator(fake_images)
```

## Model Parameters

### Generator Parameters
- `latent_dim`: Dimension of the input latent vector (default: 100)
- `ngf`: Base number of generator filters (default: 64)

### Discriminator Parameters
- `ndf`: Base number of discriminator filters (default: 64)

## Output Specifications

- Generator output: Tensor of shape (batch_size, 3, 64, 64)
- Discriminator output: Tensor of shape (batch_size, 1)

## Training Tips

1. **Learning Rate**: Start with a learning rate of 0.0002
2. **Batch Size**: Use batch sizes between 32-128 depending on available memory
3. **Optimizer**: Adam optimizer with beta1=0.5, beta2=0.999
4. **Loss Function**: Binary Cross Entropy Loss

## Advantages

1. **Consistent Latency**: The architecture maintains consistent processing times
2. **Feature Preservation**: Skip connections help maintain important features
3. **Attention Mechanism**: Improves discrimination quality
4. **Efficient Processing**: Optimized for both training and inference

## Requirements

- PyTorch >= 1.7.0
- CUDA (optional, for GPU acceleration)
- NumPy
- torchvision

## Integration

This HLC-GAN implementation can be easily integrated with existing GAN training pipelines. The architecture is compatible with standard GAN training procedures and can be used as a drop-in replacement for traditional GAN architectures. 