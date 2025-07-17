"""
audio generation logic
"""

import torch
import numpy as np
import soundfile as sf
import os
import time
from typing import List, Tuple

from ..models.vae import BreakbeatVAE
from ..utils.audio import spectrogram_to_audio, calculate_spectrogram_shape


class BreakbeatGenerator:
    """main class for generating breakbeat samples"""
    
    def __init__(self, model: BreakbeatVAE, device: torch.device, sample_length: int = 44100):
        self.model = model
        self.device = device
        self.sample_length = sample_length
        self.trained = False
        
        print(f"Breakbeat VAE initialized on {device}")
        print(f"Sample length: {sample_length} samples ({sample_length/44100:.2f} seconds)")
    
    def generate_samples(self, num_samples: int = 5, output_dir: str = "output") -> List[np.ndarray]:
        """generate multiple breakbeat samples"""
        os.makedirs(output_dir, exist_ok=True)
        samples = []
        
        target_shape = calculate_spectrogram_shape(self.sample_length)
        
        spec_tensors = self.model.generate(num_samples, str(self.device), target_shape)
        
        for i in range(num_samples):
            spec = spec_tensors[i, 0].cpu().numpy()
            audio = spectrogram_to_audio(spec, target_length=self.sample_length)
            
            audio = audio / np.max(np.abs(audio)) * 0.8
            samples.append(audio)
            
            timestamp = int(time.time())
            output_path = os.path.join(output_dir, f"gen_break_{timestamp}_{i+1}.wav")
            sf.write(output_path, audio, 44100)
            print(f"saved sample {i+1} to {output_path}")
        
        return samples 