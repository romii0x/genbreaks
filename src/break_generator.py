"""
breakbeat sample generator module

this module generates breakbeat audio samples using a variational autoencoder
trained on breakbeat spectrograms.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple
import os
import librosa
import soundfile as sf


class SpectrogramDataset(Dataset):
    #dataset for training on spectrogram audio representations
    
    def __init__(self, audio_files: List[str], sample_length: int = 44100, n_mels: int = 128):
        self.audio_files = audio_files
        self.sample_length = sample_length
        self.n_mels = n_mels
        self.spectrograms = []
        
        print(f"Loading {len(audio_files)} audio files...")
        
        for file_path in audio_files:
            try:
                # load audio
                audio, sr = librosa.load(file_path, sr=44100)
                
                # split into chunks
                for i in range(0, len(audio) - sample_length, sample_length // 4):  # 75% overlap
                    chunk = audio[i:i + sample_length]
                    if len(chunk) == sample_length:
                        # convert to mel spectrogram
                        mel_spec = librosa.feature.melspectrogram(
                            y=chunk, sr=44100, n_mels=n_mels, n_fft=2048, 
                            hop_length=512, fmax=8000
                        )
                        # convert to log scale
                        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
                        # normalize to [-1, 1]
                        log_mel = (log_mel + 80) / 80  # Assuming -80dB to 0dB range
                        log_mel = np.clip(log_mel, -1, 1)
                        
                        self.spectrograms.append(log_mel)
                        
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded {len(self.spectrograms)} spectrogram samples")
        if len(self.spectrograms) == 0:
            raise ValueError("No valid spectrograms loaded")
    
    def __len__(self):
        return len(self.spectrograms)
    
    def __getitem__(self, idx):
        spec = self.spectrograms[idx]
        return torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # channel dimension


class BreakbeatModel(nn.Module):
    # variational autoencoder for breakbeat generation
    
    def __init__(self, input_shape: Tuple[int, int] = (128, 87), latent_dim: int = 256):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.h, self.w = input_shape
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )
        
        # calculate flattened size
        self.encoder_output_size = 256 * 8 * 5
        
        # VAE layers
        self.fc_mu = nn.Linear(self.encoder_output_size, self.latent_dim)
        self.fc_var = nn.Linear(self.encoder_output_size, self.latent_dim)
        self.fc_decode = nn.Linear(self.latent_dim, self.encoder_output_size)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, (1, 8), 1, 0),
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
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 8, 5)
        return self.decoder(h)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var
    
    def generate(self, num_samples: int = 1, device = 'cpu'):
        # generate new samples from random latent vectors
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            generated = self.decode(z)
        return generated


def vae_loss(recon_x, x, mu, log_var, beta: float = 1.0):
    """VAE loss function with reconstruction and KL divergence terms"""
    # reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def spectrogram_to_audio(spectrogram: np.ndarray, sr: int = 44100, n_fft: int = 2048, 
                        hop_length: int = 512) -> np.ndarray:
    """Convert mel spectrogram back to audio using Griffin-Lim algorithm"""
    # denormalize spectrogram
    spec = spectrogram * 80 - 80  # Back to dB scale
    spec = librosa.db_to_power(spec)
    
    # convert mel spectrogram to linear spectrogram
    mel_to_stft = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=spec.shape[0], fmax=8000)
    stft = np.dot(mel_to_stft.T, spec)
    
    # use griffinlim to reconstruct audio
    audio = librosa.griffinlim(stft, n_iter=32, hop_length=hop_length, n_fft=n_fft)
    
    return audio


class BreakbeatGenerator:
    # main class for generating breakbeat samples
    
    def __init__(self, sample_length: int = 44100):
        self.sample_length = sample_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BreakbeatModel().to(self.device)
        self.trained = False
        
        print(f"Breakbeat VAE initialized on {self.device}")
        print(f"Sample length: {sample_length} samples ({sample_length/44100:.2f} seconds)")
    
    def train(self, audio_files: List[str], epochs: int = 500, 
              batch_size: int = 16, learning_rate: float = 1e-4, beta: float = 1.0):
        # train the VAE model
        if not audio_files:
            raise ValueError("No audio files provided")
        
        print(f"Training VAE on {len(audio_files)} audio files for {epochs} epochs")
        
        # create dataset and dataloader
        dataset = SpectrogramDataset(audio_files, self.sample_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        # optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
        
        # training loop
        self.model.train()
        best_loss = float('inf')
        
        try:
            for epoch in range(epochs):
                total_loss = 0
                total_recon = 0
                total_kl = 0
                
                for batch in dataloader:
                    batch = batch.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # forward pass
                    recon, mu, log_var = self.model(batch)
                    
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
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                
                if (epoch + 1) % 20 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch + 1}/{epochs}")
                    print(f"  Loss: {avg_loss:.4f} (best: {best_loss:.4f})")
                    print(f"  Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")
                    print(f"  LR: {current_lr:.6f}")
        
        except KeyboardInterrupt:
            print(f"\nTraining interrupted at epoch {epoch + 1}")
            # Auto-save on interrupt
            self.save_model(f"interrupted_vae_epoch_{epoch + 1}.pth")
        
        print("Training completed")
        self.trained = True
    
    def generate_sample(self, output_path: Optional[str] = None) -> np.ndarray:
        #Generate a single breakbeat sample
        if not self.trained and not hasattr(self, '_loaded'):
            print("Warning: Model not trained. Results may be poor.")
        
        # generate spectrogram
        spec_tensor = self.model.generate(1, str(self.device))
        spec = spec_tensor.cpu().numpy()[0, 0]  # Remove batch and channel dims
        
        # convert to audio
        audio = spectrogram_to_audio(spec)
        
        # trim/pad to exact length
        if len(audio) > self.sample_length:
            audio = audio[:self.sample_length]
        elif len(audio) < self.sample_length:
            audio = np.pad(audio, (0, self.sample_length - len(audio)))
        
        # normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        if output_path:
            sf.write(output_path, audio, 44100)
            print(f"generated sample saved to: {output_path}")
        
        return audio
    
    def generate_multiple_samples(self, num_samples: int = 5, 
                                output_dir: str = "generated_breaks") -> List[np.ndarray]:
        #Generate multiple breakbeat samples
        os.makedirs(output_dir, exist_ok=True)
        samples = []
        
        # generate all spectrograms at once for efficiency
        spec_tensors = self.model.generate(num_samples, str(self.device))
        
        for i in range(num_samples):
            spec = spec_tensors[i, 0].cpu().numpy()
            audio = spectrogram_to_audio(spec)
            
            # trim/pad to exact length
            if len(audio) > self.sample_length:
                audio = audio[:self.sample_length]
            elif len(audio) < self.sample_length:
                audio = np.pad(audio, (0, self.sample_length - len(audio)))
            
            # normalize
            audio = audio / np.max(np.abs(audio)) * 0.8
            samples.append(audio)
            
            # save with timestamp
            import time
            timestamp = int(time.time())
            output_path = os.path.join(output_dir, f"gen_break_{timestamp}_{i+1}.wav")
            sf.write(output_path, audio, 44100)
            print(f"saved sample {i+1} to {output_path}")
        
        return samples
    
    def save_model(self, filepath: str):
        # save the trained model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'sample_length': self.sample_length,
            'trained': self.trained
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        # load a trained model
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.sample_length = checkpoint.get('sample_length', 44100)
        self.trained = checkpoint.get('trained', True)
        self._loaded = True
        print(f"Model loaded from {filepath}")


def find_breakbeat_files(directory: str) -> List[str]:
    # find all wav files in directory
    breakbeat_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                breakbeat_files.append(file_path)
    
    return breakbeat_files