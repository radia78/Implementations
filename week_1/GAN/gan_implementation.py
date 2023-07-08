import torch
import torch.nn as nn
import torch.nn.functional as F

# create the generator and discriminator
class Generator(nn.Module):
    def __init__(self, 
        g_input_dim: int, 
        h_input_dim: int, 
        g_out_dim: int
    ) -> None:
        """
        Args:
            g_input_dim: Latent space input shape
            h_input_dim: Hidden layer size for the fully connected layers
            g_out_dim: Flattened dimension of the 
        """
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, h_input_dim)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_out_dim)
    
    def forward(self, x, negative_slope: float):
        """
        Args:
            x: Latent space input tensor
            negative_slope: Parameter for the Leaky ReLU activation function
        """
        x = F.leaky_relu(self.fc1(x), negative_slope)
        x = F.leaky_relu(self.fc2(x), negative_slope)
        x = F.leaky_relu(self.fc3(x), negative_slope)
        return torch.tanh(self.fc4(x)) # only output the logits

class Discriminator(nn.Module):
    def __init__(self, 
        d_input_dim: int,
        h_input_dim: int,
        dropout: float
    ) -> None:
        """
        Args:
            d_input_dim: Latent space input shape
            h_input_dim: Hidden layer size for the fully connected layers
        """
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, h_input_dim * 4)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, negative_slope: float):
        """
        Args:
            x: Latent space input tensor
            negative_slope: Parameter for the Leaky ReLU activation function
            dropout: Regularization for the Discriminator
        """
        x = self.dropout(F.leaky_relu(self.fc1(x), negative_slope))
        x = self.dropout(F.leaky_relu(self.fc2(x), negative_slope))
        x = self.dropout(F.leaky_relu(self.fc3(x), negative_slope))
        return self.fc4(x) # only output the logits
    
# create the loss function for both the discriminator and generator
def compute_discriminator_loss(
    real_logits,
    real_labels,
    fake_logits,
    fake_labels  
):
    """
    Args:
        real_logits: The logit outputs from the discriminator given the data from the real distribution
        real_lables: Tensor of 1s that indicate that the logits are from the real distribution
        fake_logits: The logit outputs from the discriminator given the data from the generator
        fake_labels: Tensor of 0s that indicate that the logits are from the fake distribution
    """
    real_loss = F.binary_cross_entropy_with_logits(real_logits, real_labels, reduction='mean') # loss on the real inputs
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_labels, reduction='mean') # loss on the fake inputs

    return real_loss + fake_loss
