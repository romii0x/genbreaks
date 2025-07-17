"""
VAE model for breakbeat generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class BreakbeatVAE(nn.Module):
    """variational autoencoder for breakbeat generation"""
    
    def __init__(self, latent_dim: int = 256, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims
        
        # encoder with adaptive pooling to handle variable input sizes
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # fixed encoder output size
        self.encoder_output_size = 256 * 8 * 8
        
        # VAE layers
        self.fc_mu = nn.Linear(self.encoder_output_size, self.latent_dim)
        self.fc_var = nn.Linear(self.encoder_output_size, self.latent_dim)
        self.fc_decode = nn.Linear(self.latent_dim, self.encoder_output_size)
        
        # decoder with adaptive upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Tanh()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, target_shape: Tuple[int, int] = None):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 8, 8) # reshape to match decoder input
        decoded = self.decoder(h)
        
        # adaptive upsampling to target shape if provided
        if target_shape is not None:
            decoded = F.interpolate(decoded, size=target_shape, mode='bilinear', align_corners=False)
        
        return decoded
    
    def forward(self, x, target_shape: Tuple[int, int] = None):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        if target_shape is None:
            target_shape = (x.size(2), x.size(3))  # use input shape if no target specified
        recon = self.decode(z, target_shape)
        return recon, mu, log_var
    
    def generate(self, num_samples: int = 1, device = 'cpu', target_shape: Tuple[int, int] = None):
        """generate new samples from random latent vectors"""
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            generated = self.decode(z, target_shape)
        return generated 