import torch
import tqdm
import logging
import numpy as np

"""
Diffusion model implementation from https://github.com/dome272/Diffusion-Models-pytorch
"""

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps # the number of markov chains
        self.beta_start = beta_start # starting level of noise
        self.beta_end = beta_end # ending level of noise
        self.img_size = img_size # the image size
        self.device = device # device of the model

        self.beta = self.noise_schedule().to(device) # the schedule noise (default: linear)
        self.alpha = 1 - self.beta # the alpha in the model
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) # alpha hat in the model

    def noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def forward_process(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] # extending the float to the image size
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None] # extending the float to the image size
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    def sample_timesteps(self, n): # sampling from the t time steps
        return torch.randint(low=1, high=self.noise_steps, size=(n, ))
    
    def sample(self, model, n): # following the algorithm from the DPPM paper
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device) # get a random image 
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    z = torch.randn_like(x)
                else:
                    z = torch.zeors_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * z
            
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    