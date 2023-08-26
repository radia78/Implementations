import torch
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, norm=True):
        super(ConvolutionalBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        if norm:
            self.norm = nn.BatchNorm2d(out_channels)
        self.norm_bool = norm
        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm_bool:
            x = self.norm(x)
        x = self.act(x)
        return self.dropout(x)
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, norm=True):
        super(UpsampleBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        if norm:
            self.norm = nn.BatchNorm2d(out_channels)
        self.norm_bool = norm 
        self.act = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.up(x)
        if self.norm_bool:
            x = self.norm(x)
        x = self.act(x)
        return self.dropout(x)

# create the generator and discriminator
class VariationalAutoEncoder(nn.Module):
    def __init__(self, config):
        super(VariationalAutoEncoder, self).__init__()

        # building encoder
        self.encoder = nn.Sequential(
            #input layer
            ConvolutionalBlock(config.n_channels, config.h_dim // 16, config.dropout, False),
            #first layer
            ConvolutionalBlock(config.h_dim // 16, config.h_dim // 8, config.dropout),
            #second layer
            ConvolutionalBlock(config.h_dim // 8, config.h_dim // 4, config.dropout),
            #third layer
            ConvolutionalBlock(config.h_dim // 4, config.h_dim // 2, config.dropout),
            #fourth layer
            ConvolutionalBlock(config.h_dim // 2, config.h_dim, config.dropout),
        )

        self.fc_mu = nn.Linear(config.h_dim, config.z_dim) # mean
        self.fc_var = nn.Linear(config.h_dim, config.z_dim) # variation

        self.decoder = nn.Sequential(
            #input layer
            UpsampleBlock(config.z_dim, config.h_dim, config.dropout),
            #first layer
            UpsampleBlock(config.h_dim, config.h_dim // 2, config.dropout),
            #second layer
            UpsampleBlock(config.h_dim // 2, config.h_dim // 4, config.dropout),
            #third layer
            UpsampleBlock(config.h_dim // 4, config.h_dim // 8, config.dropout),
            #fourth
            UpsampleBlock(config.h_dim // 8, config.h_dim // 16, config.dropout)
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(config.h_dim // 16, config.n_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def encode(self, x):
        result = self.encoder(x)
        result = rearrange(result, "b c h w -> b h w c")

        # split the result to the parameterized mean and var
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var
    
    def decode(self, z):
        z = rearrange(z, "b h w c -> b c h w")
        result = self.decoder(z)
        result = self.final_layer(result)

        return result
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)

        return mu + noise * std
    
    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar
    
def compute_vae_loss(input, recons, mu, logvar, minibatch_weight):

    recons_loss = F.mse_loss(input, recons) # the difference between the real data and the reconstructed data
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 -logvar.exp(), dim=1)) # difference between distributions

    return minibatch_weight * kld_loss + recons_loss, kld_loss, recons_loss

@dataclass
class VAEConfig:
    n_channels = 3
    h_dim = 512
    dropout = 0.3
    z_dim = 100
