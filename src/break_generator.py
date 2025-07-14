"""
breakbeat sample generator module

this module generates breakbeat audio samples using a neural network
trained on breakbeat wav files.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import os
import librosa
import soundfile as sf


class AudioSampleDataset(Dataset):
    # dataset for training on actual audio samples
    
    def __init__(self, audio_files: List[str], sample_length: int = 44100):
        # initialize the dataset
        # args:
        #   audio_files: list of paths to breakbeat wav files
        #   sample_length: length of each training sample in samples
        self.audio_files = audio_files
        self.sample_length = sample_length
        self.samples = []
        
        # load and process all audio files
        for file_path in audio_files:
            try:
                audio, sr = librosa.load(file_path, sr=44100)
                # split into chunks of sample_length
                for i in range(0, len(audio) - sample_length, sample_length // 2):  # 50% overlap
                    chunk = audio[i:i + sample_length]
                    if len(chunk) == sample_length:
                        self.samples.append(chunk)
            except Exception as e:
                print(f"error loading {file_path}: {e}")
        
        print(f"loaded {len(self.samples)} audio samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # normalize
        sample = sample / np.max(np.abs(sample))
        return torch.tensor(sample, dtype=torch.float32)


class BreakbeatModel(nn.Module):
    # neural network for generating breakbeat audio
    
    def __init__(self, sample_length: int = 44100, hidden_size: int = 512):
        # initialize the generator
        # args:
        #   sample_length: length of generated audio in samples
        #   hidden_size: size of hidden layers
        super().__init__()
        self.sample_length = sample_length
        self.hidden_size = hidden_size
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(sample_length, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, sample_length),
            nn.Tanh()
        )
        
    def forward(self, x):
        # forward pass
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def generate_sample(self, noise: Optional[torch.Tensor] = None) -> np.ndarray:
        # generate a new breakbeat sample
        # args:
        #   noise: optional noise input (if None, uses random noise)
        # returns:
        #   generated audio as numpy array
        self.eval()
        
        with torch.no_grad():
            if noise is None:
                # generate random noise
                noise = torch.randn(1, self.sample_length, device=next(self.parameters()).device)
            
            # generate sample
            generated = self.forward(noise)
            
            # convert to numpy and normalize
            audio = generated.cpu().numpy().flatten()
            audio = audio / np.max(np.abs(audio)) * 0.8  # normalize to 80%
            
            return audio


class BreakbeatGenerator:
    # main class for generating breakbeat samples
    
    def __init__(self, sample_length: int = 44100):
        # initialize the generator app
        # args:
        #   sample_length: length of generated samples in samples (1 second at 44.1kHz)
        self.sample_length = sample_length
        self.model = BreakbeatModel(sample_length)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"breakbeat generator initialized on {self.device}")
        print(f"sample length: {sample_length} samples ({sample_length/44100:.2f} seconds)")
    
    def train(self, audio_files: List[str], epochs: int = 100, 
              batch_size: int = 8, learning_rate: float = 0.001):
        # train the model on breakbeat audio files
        # args:
        #   audio_files: list of paths to breakbeat wav files
        #   epochs: number of training epochs
        #   batch_size: batch size for training
        #   learning_rate: learning rate for optimizer
        if not audio_files:
            raise ValueError("No audio files provided")
        
        print(f"training on {len(audio_files)} audio files for {epochs} epochs")
        
        # create dataset and dataloader
        dataset = AudioSampleDataset(audio_files, self.sample_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # forward pass
                optimizer.zero_grad()
                output = self.model(batch)
                
                # calculate loss
                loss = criterion(output, batch)
                
                # backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # print progress
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"epoch {epoch + 1}/{epochs}, loss: {avg_loss:.6f}")
        
        print("training completed")
        self._trained = True
    
    def generate_sample(self, output_path: Optional[str] = None) -> np.ndarray:
        # generate a new breakbeat sample
        # args:
        #   output_path: optional path to save the generated audio
        # returns:
        #   generated audio as numpy array
        audio = self.model.generate_sample()
        
        if output_path:
            sf.write(output_path, audio, 44100)
            print(f"generated sample saved to: {output_path}")
        
        return audio
    
    def generate_multiple_samples(self, num_samples: int = 5, 
                                output_dir: str = "generated_breaks") -> List[np.ndarray]:
        # generate multiple breakbeat samples
        # args:
        #   num_samples: number of samples to generate
        #   output_dir: directory to save samples
        # returns:
        #   list of generated audio arrays
        os.makedirs(output_dir, exist_ok=True)
        samples = []
        
        for i in range(num_samples):
            audio = self.generate_sample()
            samples.append(audio)
            
            # save with timestamp
            import time
            timestamp = int(time.time())
            output_path = os.path.join(output_dir, f"breakbeat_{timestamp}_{i+1}.wav")
            sf.write(output_path, audio, 44100)
            print(f"saved sample {i+1} to {output_path}")
        
        return samples
    
    def save_model(self, filepath: str):
        # save the trained model
        torch.save(self.model.state_dict(), filepath)
        print(f"model saved to {filepath}")
    
    def load_model(self, filepath: str):
        # load a trained model
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"model loaded from {filepath}")


def find_breakbeat_files(directory: str) -> List[str]:
    # find all wav file breakbeats
    # args:
    #   directory: directory to search
    # returns:
    #   list of wav file paths
    breakbeat_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                breakbeat_files.append(file_path)
    
    return breakbeat_files