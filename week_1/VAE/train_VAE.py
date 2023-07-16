import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from matplotlib import pyplot as plt
from vae_implementation import *

def load_training_objects():
    # load the MNIST handwritten dataset
    train_dataset = MNIST(
        root='./mnist_data/',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )) # normalize the three channels
        ]) 
    )
    # load the model
    model = VariationalAutoEncoder(
        imgsz=28**2,
        h_input_dim=512,
        latent_dim=100,
        negative_slope=0.2,
        dropout=0.1
    )

    # load the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    return train_dataset, model, optimizer

def train(
        total_epochs: int,
        model,
        optimizer,
        train_loader,
        device
):
    
    # send the model to the gpu
    model.to(device)

    # save the number of training samples
    train_len = len(train_loader)

    for epoch in range(total_epochs):
        # turn the training mode on
        model.train()

        # create a list of losses
        tot_kld_loss = []
        tot_recons_loss = []
        tot_model_loss = []

        # loop through each batch
        for x_real, _ in train_loader:

            minibatch_weight = x_real.shape[0]/train_len
            
            optimizer.zero_grad(set_to_none=True) # zero out gradients
            x_real = x_real.flatten(start_dim=1).to(device) # send the data to the GPU

            recons, mu, logvar = model(x_real) # get the mean, variance, and reconstructed data
            model_loss, kld_loss, recons_loss = compute_vae_loss(x_real, recons, mu, logvar, minibatch_weight) # compute the loss

            model_loss.backward()
            optimizer.step()
            
            # append the losses
            tot_kld_loss.append(kld_loss)
            tot_recons_loss.append(recons_loss)
            tot_model_loss.append(model_loss)

            # mean losses
            avg_kld_loss = sum(tot_kld_loss)/len(tot_kld_loss)
            avg_recons_loss = sum(tot_recons_loss)/len(tot_recons_loss)
            avg_model_loss = sum(tot_model_loss)/len(tot_model_loss)

        print(f"Epoch {epoch} | Avg Model Loss: {avg_model_loss} |Avg KLD Loss: {avg_kld_loss} | Avg Recons Loss: {avg_recons_loss}")

        if (epoch + 1) % 25 == 0:
            # save the dictionary
            checkpoint = model.state_dict()
            PATH = "checkpoint.pt"
            torch.save(checkpoint, PATH)
            print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

            # save the image results
            model.eval()
            dec_img, _, _ = model(x_real[0])
            plt.imsave(f"{epoch}-result.png", dec_img.reshape(28, 28).cpu().detach(), cmap="gray")
            
def main(gpu, total_epochs, batch_size):
    train_dataset, model, optimizer = load_training_objects()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train(total_epochs, model, optimizer, train_loader, gpu)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # its 'mps' for mac m1
    main(device, args.epochs, args.batch_size)
