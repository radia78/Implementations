import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from dataclasses import dataclass
import torch.functional as F
from utils import *
import torch.nn as nn

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

        return self.layers(img.view(-1, self.init_size, self.init_size, self.init_hidden_dim))

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

def load_training_objs():
    # load the MNIST handwritten dataset
    train_dataset = MNIST(
        root='./mnist_data/',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )) # normalize
        ]) 
    )
    # initiate the configs
    generator_config = DeepConvolutionalGeneratorConfig()
    discriminator_config = DeepConvolutionDiscriminatorConfig()

    # intitate the models
    generator = DeepConvolutionalGenerator(generator_config)
    discriminator = DeepConvolutionalDiscriminator(discriminator_config)

    # load the optimizer
    G_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    return generator, discriminator, G_optimizer, D_optimizer


def main(gpu, total_epochs, batch_size):
    train_dataset, generator, discriminator, D_optimizer, G_optimizer = load_training_objs()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # initiate the trainer class
    trainer = TrainerGAN(
        gpu, 
        generator,
        discriminator,
        train_loader,
        G_optimizer,
        D_optimizer
    )
    trainer.train(total_epochs, "checkpoint.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # its 'mps' for mac m1
    main(device, args.epochs, args.batch_size)
    