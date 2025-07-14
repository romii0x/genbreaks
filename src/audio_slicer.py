"""
audio slicer module for genbreaks

this module handles slicing audio files for breakbeat generation.
"""

import numpy as np
import librosa
from typing import List


class AudioSlicer:
    def __init__(self, sample_rate: int = 44100):
        # initialize the audio slicer
        # args:
        #   sample_rate: target sample rate for audio processing
        self.sample_rate = sample_rate
        self.slices = []
        self.slice_boundaries = []
        
    def load_audio(self, file_path: str) -> np.ndarray:
        # load audio file and resample to target sample rate
        # args:
        #   file_path: path to the audio file
        # returns:
        #   audio data as numpy array
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio
    
    def slice_equal(self, audio: np.ndarray, num_slices: int = 16) -> List[np.ndarray]:
        # slice audio into equal segments
        # args:
        #   audio: audio data as numpy array
        #   num_slices: number of slices to create
        # returns:
        #   list of audio slices
        
        slice_length = len(audio) // num_slices
        slices = []
        boundaries = []
        
        for i in range(num_slices):
            start = i * slice_length
            end = start + slice_length if i < num_slices - 1 else len(audio)
            
            slice_audio = audio[start:end]
            slices.append(slice_audio)
            boundaries.append((start, end))
            
        self.slices = slices
        self.slice_boundaries = boundaries

        return slices