from model import Diffusion
import os
import torch
from tqdm import tqdm
from unet import UNet
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.tensorboard import SummaryWriter 

# import distributed model stuff for multigpu train
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

from utils import plot_images, save_images, setup_logging, get_data

LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])

def ddp_setup():
    """
    Declare the master address for the machine w rank 0
    """
    init_process_group(backend='nccl')

def train(args):
    # setup up the logs and models
    setup_logging(args.run_name)
    gpu_id = args.gpu_id
    dataloader = get_data(args)
    model = UNet().to(gpu_id)
    model = DDP(model, device_ids=[gpu_id])
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, device=gpu_id)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        dataloader.sampler.set_epoch(epoch) # setting the shuffler for each GPU
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(gpu_id)
            t = diffusion.sample_timesteps(images.shape[0]).to(gpu_id)
            x_t, noise = diffusion.forward_process(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if args.gpu_id == 0 and (epoch + 1) % 100 == 0:
            sampled_images = diffusion.sample(model, n=images.shape[0])
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            torch.save(model.module.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

def main(args):
    ddp_setup()
    train(args)
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncoditional"
    args.epochs = 500
    args.batch_size = 12
    args.img_size = 64
    args.ddp = True
    args.gpu_id = LOCAL_RANK
    args.lr = 3e-4
    
    main(args)
