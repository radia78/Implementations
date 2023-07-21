import sys
sys.path.insert(1, "/Users/radia/Projects/ml_projects/Text2Image")

import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from model import *
from utils import *

def load_training_objs():
    # load the MNIST handwritten dataset
    train_dataset = MNIST(
        root='./mnist_data/',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(64),
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

    return train_dataset, generator, discriminator, G_optimizer, D_optimizer


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
    
    device = torch.device("mps" if torch.cuda.is_available() else "cpu") # its 'mps' for mac m1
    main(device, args.epochs, args.batch_size)
    