"""
training logic for the VAE model
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List
import os

from ..models.vae import BreakbeatVAE
from ..data.dataset import SpectrogramDataset
from .losses import vae_loss


class VAETrainer:
    def __init__(self, model: BreakbeatVAE, device: torch.device):
        self.model = model
        self.device = device
        self.trained = False
    
    def train(self, audio_files: List[str], epochs: int = 500, batch_size: int = 16, 
              learning_rate: float = 0.0001, beta: float = 1.0, sample_length: int = 44100):
        """train the VAE model"""
        if not audio_files:
            raise ValueError("No audio files provided")
        
        print(f"training VAE on {len(audio_files)} audio files")
        print(f"training config: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
        
        # create dataset and dataloader
        dataset = SpectrogramDataset(audio_files, sample_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        # optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
        
        # training loop
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0
            total_recon = 0
            total_kl = 0
            
            for batch in dataloader:
                batch = batch.to(self.device)
                
                optimizer.zero_grad()
                
                # forward pass
                recon, mu, log_var = self.model(batch, target_shape=(batch.size(2), batch.size(3)))
                
                # calculate loss
                loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, log_var, beta)
                
                # backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
            
            # calculate averages
            avg_loss = total_loss / len(dataset)
            avg_recon = total_recon / len(dataset)
            avg_kl = total_kl / len(dataset)
            
            scheduler.step(avg_loss)
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model("models/audio_vae_best.pth")
            
            # Save last model every 50 epochs
            if (epoch + 1) % 50 == 0:
                self.save_model("models/audio_vae_last.pth")
            
            if (epoch + 1) % 20 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"  Loss: {avg_loss:.4f} (best: {best_loss:.4f})")
                print(f"  Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")
                print(f"  LR: {current_lr:.6f}")
        
        print("training completed")
        self.trained = True
    
    def save_model(self, filepath: str):
        """save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'trained': self.trained
        }, filepath)
    
    def load_model(self, filepath: str):
        """load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.trained = checkpoint.get('trained', True)
        print(f"model loaded from {filepath}") 