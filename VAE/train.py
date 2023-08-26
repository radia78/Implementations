import torch
import os
import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from matplotlib import pyplot as plt
from model import *
from torch.utils.tensorboard import SummaryWriter 

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DistributedSampler

def load_training_objects(args):
    # load the MNIST handwritten dataset
    dataset = CIFAR10(
        root='./mnist_data/',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ), (0.5)) # normalize the three channels
        ]) 
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=DistributedSampler(dataset),
        shuffle=False
    )

    # create the model
    model_config = VAEConfig()
    model = VariationalAutoEncoder(model_config)
    model = model.to(args.gpu_id)
    model = DDP(model, device_ids=[args.gpu_id])

    # load the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    return dataloader, model, optimizer

def setup_logging(args):
    os.makedirs("VAE", "models", args.run_name)
    os.makedirs("VAE", "images", args.run_name)
    os.makedirs(os.path.join("VAE", "models", args.run_name))
    os.makedirs(os.path.join("VAE", "images", args.run_name))

def train(args):
    setup_logging(args)
    dataloader, model, optimizer = load_training_objects(args)
    logger = SummaryWriter(os.path.join("VAE", "runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        dataloader.sampler.set_epoch(epoch)
        pbar = tqdm(dataloader)
        model.train()

        # create a list of losses
        tot_kld_loss = []
        tot_recons_loss = []
        tot_model_loss = []

        print(f"Starting epoch {epoch + 1} on GPU {args.gpu_id}: ")

        # loop through each batch
        for i, (x_real, _) in enumerate(pbar):
            minibatch_weight = x_real.shape[0]/l
            x_real = x_real.to(args.gpu_id)

            optimizer.zero_grad(set_to_none=True) # zero out gradients
            recons, mu, logvar = model(x_real) # get the mean, variance, and reconstructed data
            model_loss, kld_loss, recons_loss = compute_vae_loss(x_real, recons, mu, logvar, minibatch_weight) # compute the loss

            model_loss.backward()
            optimizer.step()

            pbar.set_postfix(Loss=model_loss)
            
            # append the losses
            tot_kld_loss.append(kld_loss)
            tot_recons_loss.append(recons_loss)
            tot_model_loss.append(model_loss)

        # mean losses
        avg_kld_loss = sum(tot_kld_loss)/len(tot_kld_loss)
        avg_recons_loss = sum(tot_recons_loss)/len(tot_recons_loss)
        avg_model_loss = sum(tot_model_loss)/len(tot_model_loss)

        # register losses
        logger.add_scaler("Model Loss", avg_model_loss)
        logger.add_scalar("Reconstruction Loss", avg_recons_loss)
        logger.add_scalar("Kullback-Leibler Divergence Loss", avg_kld_loss)
        
        if epoch == 0 or (epoch + 1) == args.epochs:
            # save the dictionary
            checkpoint = model.module.state_dict()
            PATH = os.path.join("VAE", "models", args.run_name, "chkpt.pt")
            torch.save(checkpoint, PATH)

            # save the image results
            model.eval()
            sample = torch.randn(1, 3, 32, 32, device=args.gpu_id)
            dec_img, _, _ = model(sample)
            image_path = os.makedirs(os.path.join("VAE", "images", args.run_name, f"{epoch + 1}_result.png"))
            plt.imsave(image_path, dec_img.reshape(32, 32).cpu().detach())
            
def main(args):
    init_process_group(backend='nccl')
    train(args)
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    args = parser.parse_args()
    args.gpu_id = int(os.environ['LOCAL_RANK'])
    args.batch_size = 64
    args.run_name = "2023_08_25"
    args.epochs = 500

    main(args)
