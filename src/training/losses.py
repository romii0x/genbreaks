"""
loss functions for VAE training
"""

import torch
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, log_var, beta: float = 1.0):
    """VAE loss function with reconstruction and KL divergence terms"""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss 