"""
audio processing utilities
"""

import numpy as np
import librosa
from typing import Tuple


def calculate_spectrogram_shape(sample_length: int, n_mels: int = 128, hop_length: int = 512) -> Tuple[int, int]:
    """calculate spectrogram dimensions from sample length"""
    time_steps = (sample_length // hop_length) + 1
    return (n_mels, time_steps)


def spectrogram_to_audio(spectrogram: np.ndarray, target_length: int = None, sr: int = 44100, n_fft: int = 2048, 
                        hop_length: int = 512) -> np.ndarray:
    """convert mel spectrogram back to audio using Griffin-Lim algorithm"""
    # denormalize spectrogram
    spec = spectrogram * 80 - 80  # Back to dB scale
    spec = librosa.db_to_power(spec)
    
    # convert mel spectrogram to linear spectrogram
    mel_to_stft = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=spec.shape[0], fmax=8000)
    stft = np.dot(mel_to_stft.T, spec)
    
    # use griffinlim to reconstruct audio
    audio = librosa.griffinlim(stft, n_iter=32, hop_length=hop_length, n_fft=n_fft)
    
    # force the exact target length
    if target_length is not None:
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        elif len(audio) > target_length:
            audio = audio[:target_length]
    
    return audio 