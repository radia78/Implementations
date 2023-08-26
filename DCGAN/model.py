import torch
import torch.nn as nn
from dataclasses import dataclass

# model implementation
class DeepConvolutionalGenerator(nn.Module):
    def __init__(self, config):
        super(DeepConvolutionalGenerator, self).__init__()

        # save shape info
        self.init_size = config.init_size
        self.init_hidden_dim = config.init_hidden_dim

        # project the latent dim vector into a size x size image with 1024 channels
        self.latent2img = nn.Linear(config.latent_dim, config.init_hidden_dim * config.init_size ** 2)

        # fractionally strided (transpose convolution) convolution on long tensor
        self.layers = nn.Sequential(
            # first layer
            nn.ConvTranspose2d(config.init_hidden_dim, config.init_hidden_dim // 2**1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.init_hidden_dim // 2**1),
            nn.ReLU(True),
            # second layer
            nn.ConvTranspose2d(config.init_hidden_dim // 2**1, config.init_hidden_dim // 2**2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.init_hidden_dim // 2**2),
            nn.ReLU(True),
            # third layer
            nn.ConvTranspose2d(config.init_hidden_dim // 2**2, config.init_hidden_dim // 2**3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.init_hidden_dim // 2**3),
            nn.ReLU(True),
            # fourth layer
            nn.ConvTranspose2d(config.init_hidden_dim // 2**3, config.n_channels,  kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        """
        Args:
            z: The latent input
        """
        img = self.latent2img(z)

        return self.layers(img.view(-1, self.init_hidden_dim, self.init_size, self.init_size))

class DeepConvolutionalDiscriminator(nn.Module):
    def __init__(self, config):
        super(DeepConvolutionalDiscriminator, self).__init__()

        # project the final convolved tensor into a single linear unit
        self.img2logit = nn.Linear(config.init_hidden_dim * config.init_size ** 2, 1)

        # convolutional layers (transform wide tensor to long tensor)
        self.layers = nn.Sequential(
            # first layer
            nn.Conv2d(config.n_channels, config.init_hidden_dim // 2**3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(config.negative_slope, inplace=True),
            nn.Dropout2d(config.dropout),
            # second layer
            nn.Conv2d(config.init_hidden_dim // 2**3, config.init_hidden_dim // 2**2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.init_hidden_dim // 2**2),
            nn.LeakyReLU(config.negative_slope, inplace=True),
            nn.Dropout2d(config.dropout),
            # third layer
            nn.Conv2d(config.init_hidden_dim // 2**2, config.init_hidden_dim // 2**1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.init_hidden_dim // 2**1),
            nn.LeakyReLU(config.negative_slope, inplace=True),
            nn.Dropout2d(config.dropout),
            # fourth layer
            nn.Conv2d(config.init_hidden_dim // 2**1, config.init_hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.init_hidden_dim),
            nn.LeakyReLU(config.negative_slope, inplace=True),
            nn.Dropout2d(config.dropout)
        )
    
    def forward(self, input):
        """
        Args:
            input: The image tensor
        """
        output = self.layers(input)

        return self.img2logit(output.flatten(start_dim=1))

# configs 
@dataclass
class DeepConvolutionalGeneratorConfig:
    init_hidden_dim: int=1024
    init_size: int=4
    latent_dim: int=100
    n_channels: int=1

@dataclass
class DeepConvolutionDiscriminatorConfig:
    init_hidden_dim: int=1024
    init_size: int=4
    n_channels: int=1
    negative_slope: float=0.2
    dropout: float=0.1
    