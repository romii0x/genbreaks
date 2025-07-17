"""
dataset for breakbeat spectrograms
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from typing import List


class SpectrogramDataset(Dataset):
    """dataset for training on spectrogram audio representations"""
    
    def __init__(self, audio_files: List[str], sample_length: int = 44100, n_mels: int = 128):
        self.audio_files = audio_files
        self.sample_length = sample_length
        self.n_mels = n_mels
        self.spectrograms = []
        
        print(f"loading {len(audio_files)} audio files...")
        
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
        
        print(f"loaded {len(self.spectrograms)} spectrogram samples")
        if len(self.spectrograms) == 0:
            raise ValueError("No valid spectrograms loaded")
    
    def __len__(self):
        return len(self.spectrograms)
    
    def __getitem__(self, idx):
        spec = self.spectrograms[idx]
        return torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # channel dimension 