import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from model import *

def setup_logging(run_name):
    os.makedirs("GAN", "models", exist_ok=True)
    os.makedirs("GAN", "results", exist_ok=True)
    os.makedirs("GAN", "images", exist_ok=True)
    os.makedirs(os.path.join("GAN", "models", run_name), exist_ok=True)
    os.makedirs(os.path.join("GAN", "results", run_name), exist_ok=True)
    os.makedirs(os.path.join("GAN", "images", run_name), exist_ok=True)

def load_training_objects():
    # load the MNIST handwritten dataset
    train_dataset = MNIST(
        root='./mnist_data/',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ), (0.5, )) # normalize the three channels
        ]) 
    )
    # load the model
    # hidden dimension = 256, latent space size = 128, flattened image size = 28**2
    generator = Generator(g_input_dim=100, h_input_dim=256, g_out_dim=28**2)
    discriminator = Discriminator(d_input_dim=28**2, h_input_dim=256, dropout=0.1)

    # load the optimizer
    D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
    G_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)

    return train_dataset, generator, discriminator, D_optimizer, G_optimizer

def D_train(x_real, D_optimizer, D, G, device):
    # send the data batch into the gpu
    x_real = x_real.flatten(start_dim=1).to(device)

    # clear out the gradients
    D_optimizer.zero_grad(set_to_none=True)

    # load the dummy data
    fake_inputs = torch.randn(x_real.shape[0], 100, device=device)
    real_labels = torch.ones(x_real.shape[0], 1, device=device)

    # train the discriminator
    G_fake_output = G(fake_inputs, 0.2) # data generated from generator
    D_fake_output = D(G_fake_output, 0.2) # fake logits from discriminator
    D_real_output = D(x_real, 0.2) # real logits from discriminator
    D_loss = compute_discriminator_loss(
        D_real_output,
        real_labels,
        D_fake_output,
        torch.zeros(x_real.shape[0], 1, device=device)
    )
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()

def G_train(G_optimizer, D, G, batch_size, device):
    # clear out the gradients
    G_optimizer.zero_grad(set_to_none=True)

    # load the dummy data
    fake_inputs = torch.randn(batch_size, 100, device=device)
    real_labels = torch.ones(batch_size, 1, device=device)

    # train the discriminator
    D_fake_output = D(G(fake_inputs, 0.2), 0.2) # fake logits from discriminator

    # compute generator loss
    G_loss = F.binary_cross_entropy_with_logits(D_fake_output, real_labels)
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()

def train(
        run_name,
        total_epochs: int,
        G,
        D,
        G_optimizer,
        D_optimizer,
        train_loader,
        device
):
    
    setup_logging(run_name)
    # send the model to the gpu
    G.to(device)
    D.to(device)
    
    # log the loss values
    logger = SummaryWriter(os.path.join("runs", run_name))

    for epoch in range(total_epochs):
        # turn the training mode on
        D.train()
        G.train()

        # Create the list of losses
        D_loss = []
        G_loss = []

        # progress bar
        pbar = tqdm(train_loader)

        # loop through each batch
        for i, (x_real, _) in enumerate(pbar):

            # send the data batch into the gpu
            x_real = x_real.flatten(start_dim=1).to(device)

            # loss
            dloss = D_train(x_real, D_optimizer, D, G, device)
            gloss = G_train(G_optimizer, D, G, x_real.shape[0], device)

            D_loss.append(dloss) # train the discriminator and output loss
            G_loss.append(gloss)

            pbar.set_postfix(gloss=gloss, dloss=dloss)

        avg_gloss = sum(G_loss)/len(G_loss)
        avg_dloss = sum(D_loss)/len(D_loss)
        logger.add_scalar("Generator Loss", avg_gloss, global_step=epoch)
        logger.add_scalar("Discriminator Loss", avg_dloss, global_step=epoch)

        if (epoch == 0) or (epoch + 1 == total_epochs):
            # save the dictionary
            checkpoint = G.state_dict()
            PATH = os.path.join("models", run_name, "ckpt.pt")
            torch.save(checkpoint, PATH)
            print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
            
            # save the image results
            G.eval()
            images_path = os.path.join("images", run_name, f"image_{epoch + 1}.png")
            gen_img = G(torch.randn(128).to(device), 0.2)
            plt.imsave(images_path, gen_img.reshape(28, 28).detach(), cmap="gray")
            

def main(gpu, total_epochs, batch_size):
    train_dataset, generator, discriminator, D_optimizer, G_optimizer = load_training_objects()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train("gan_implementation", total_epochs, generator, discriminator, G_optimizer, D_optimizer, train_loader, gpu)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=64, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    args.epochs = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps") # its 'mps' for mac m1
    main(device, args.epochs, args.batch_size)
