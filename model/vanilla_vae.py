import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any, Union
from torch import Tensor
from model.base import BaseVAE

class VanillaVAE(BaseVAE):
    def __init__(self, input_dim, hidden_dim, latent_dim, **kwargs):
        super(VanillaVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Latent space
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x: Tensor) -> List[Tensor]:
        """Encode input and return mu and log_var"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return [mu, log_var]

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent variable to reconstruction"""
        return self.decoder(z)

    def forward(self, x: Tensor, **kwargs) -> List[Tensor]:
        """Forward pass returning [reconstruction, input, mu, log_var]"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return [reconstruction, x, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        VAE loss function with reconstruction loss + KL divergence
        Using the loss from your loss.py file
        """
        reconstruction = args[0]  # reconstructed_x
        x = args[1]              # original input
        mu = args[2]             # mean
        log_var = args[3]        # log variance
        
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstruction, x, reduction='sum')
        
        # KL divergence loss
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        total_loss = reconstruction_loss + kl_divergence
        
        return {
            'loss': total_loss,
            'Reconstruction_Loss': reconstruction_loss.detach(),
            'KLD': kl_divergence.detach()
        }

    def sample(self, num_samples: int, current_device: Union[int, str], **kwargs) -> Tensor:
        """Sample from latent space"""
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        return self.decode(z)

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """Generate reconstruction for input x"""
        return self.forward(x)[0]
        
    def get_latent_representation(self, x: Tensor) -> Tensor:
        """Get latent representation (mu) for clustering/visualization"""
        mu, log_var = self.encode(x)

        return mu     