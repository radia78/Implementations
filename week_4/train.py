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

def ddp_setup(rank: int, world_size: int):
    """
    Declare the master address for the machine w rank 0
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

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

def main(rank, world_size, config):
    ddp_setup(rank, world_size)
    train(config)
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    config = parser.parse_args()
    config.run_name = "DDPM_Uncoditional"
    config.epochs = 500
    config.batch_size = 12
    config.img_size = 64
    config.ddp = True
    config.lr = 3e-4

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, config), nprocs=world_size)
