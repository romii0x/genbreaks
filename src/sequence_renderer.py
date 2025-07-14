"""
sequence renderer module for genbreaks

this module handles rendering sequences of audio slices to wav files.
"""

import numpy as np
import soundfile as sf
from typing import List, Optional


class SequenceRenderer:
    # handles rendering of slice sequences to wav files
    
    def __init__(self, sample_rate: int = 44100):
        # initialize the sequence renderer
        # args:
        #   sample_rate: audio sample rate
        self.sample_rate = sample_rate
        self.slices = []
        self.volume = 1.0
        
    def set_slices(self, slices: List[np.ndarray]):
        # set the audio slices to use for rendering
        # args:
        #   slices: list of audio slices as numpy arrays
        self.slices = slices
    
    def set_volume(self, volume: float):
        # set the volume
        # args:
        #   volume: volume level (0.0 to 1.0)
        self.volume = max(0.0, min(1.0, volume))
    
    def render_sequence(self, sequence: List[int], output_path: Optional[str] = None) -> np.ndarray:
        # render a sequence to audio without playing it
        # args:
        #   sequence: list of slice numbers (1-16)
        #   output_path: optional path to save the rendered audio
        # returns:
        #   rendered audio as numpy array
        if not self.slices:
            raise ValueError("No slices available. Set slices first.")
        
        # validate sequence
        for slice_num in sequence:
            if slice_num < 1 or slice_num > len(self.slices):
                raise ValueError(f"Invalid slice number: {slice_num}. Must be 1-{len(self.slices)}")
        

        
        # concatenate slices with their natural timing
        rendered_audio = []
        
        for slice_num in sequence:
            slice_audio = self.slices[slice_num - 1].copy()
            
            # apply volume
            slice_audio *= self.volume
            
            rendered_audio.append(slice_audio)
        
        # concatenate all slices
        final_audio = np.concatenate(rendered_audio)
        
        # save if output path provided
        if output_path:
            sf.write(output_path, final_audio, self.sample_rate)
        
        return final_audio