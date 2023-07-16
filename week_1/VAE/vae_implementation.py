import torch
import torch.nn as nn
import torch.nn.functional as F

# create the generator and discriminator
class VariationalAutoEncoder(nn.Module):
    def __init__(self,
        imgsz: int,
        h_input_dim: int,
        latent_dim: int,
        negative_slope: float,
        dropout: float
    ) -> None:
        """
        Args:
            input_dim: The dimension of the real data
            h_dim: The hidden dimension for the MLP and gets exponated multiple times
        """
        super(VariationalAutoEncoder, self).__init__()

        # building encoder
        self.encoder = nn.Sequential(
            # first layer
            nn.Linear(imgsz, h_input_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            # second layer
            nn.Linear(h_input_dim, h_input_dim // 2**1),
            nn.LeakyReLU(negative_slope=negative_slope),
            # third layer
            nn.Linear(h_input_dim // 2, h_input_dim // 2**2),
            nn.LeakyReLU(negative_slope=negative_slope)
        )

        self.fc_mu = nn.Linear(self.encoder[-2].out_features, latent_dim) # mean
        self.fc_var = nn.Linear(self.encoder[-2].out_features, latent_dim) # variation

        # build decoder
        self.decoder = nn.Sequential(
            # first layer
            nn.Linear(latent_dim, h_input_dim // 2**2),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout),
            # second layer
            nn.Linear(h_input_dim // 2**2, h_input_dim // 2**1),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout),
            # third layer
            nn.Linear(h_input_dim // 2**1, h_input_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout)
        )
        self.final_layer = nn.Sequential(
            nn.Linear(h_input_dim, imgsz),
            nn.Tanh()
        )

    def encode(self, x):
        """
        Args:
            x: The input image data
        """
        result = self.encoder(x)

        # split the result to the parameterized mean and var
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var
    
    def decode(self, z):
        """
        Args:
            z: latent space input
        """
        result = self.decoder(z)
        result = self.final_layer(result)

        return result
    
    def reparameterize(self, mu, logvar):
        """
        Args: 
            mu: Computed mean of from the real data distribution
            logvar: The logarithmic variance computed from the real data distribution
        """
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
