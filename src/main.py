"""
                        __                     __          
     .-----.-----.-----|  |--.----.-----.---.-|  |--.-----.
     |  _  |  -__|     |  _  |   _|  -__|  _  |    <|__ --|
     |___  |_____|__|__|_____|__| |_____|___._|__|__|_____|
     |_____|                                               

"""

from typing import List, Optional
from audio_slicer import AudioSlicer
from sequence_renderer import SequenceRenderer
from sequence_generator import SequenceGenerator
from break_generator import BreakbeatGenerator, find_breakbeat_files


class GenBreaks:
    def __init__(self, mode: str = "audio", sample_length: int = 44100):
        # initialize the genbreaks application
        # args:
        #   mode: "audio" for neural audio generation, "sequence" for sequence generation
        #   sample_length: length of generated samples in samples (audio mode only)
        self.mode = mode
        self.sample_length = sample_length
        
        if mode == "audio":
            self.audio_generator = BreakbeatGenerator(sample_length)
            print("genbreaks initialized in audio generation mode!")
            print(f"sample length: {sample_length} samples ({sample_length/44100:.2f} seconds)")
        elif mode == "sequence":
            # initialize sequence mode components directly
            self.slicer = AudioSlicer()
            self.renderer = SequenceRenderer()
            self.sequence_generator = SequenceGenerator()
            self.slices = []
            print("genbreaks initialized in sequence generation mode!")
        else:
            raise ValueError("Mode must be 'audio' or 'sequence'")
    


    # audio generation methods
    def train_on_directory(self, directory: str, epochs: int = 500, 
                          batch_size: int = 16, learning_rate: float = 1e-4):
        # train the audio model on all wav files in a directory
        if self.mode != "audio":
            raise ValueError("This method is only available in audio mode")
        
        audio_files = find_breakbeat_files(directory)
        
        if not audio_files:
            raise ValueError(f"No wav files found in {directory}")
        
        print(f"found {len(audio_files)} audio files in {directory}")
        self.audio_generator.train(audio_files, epochs, batch_size, learning_rate)
    
    def generate_sample(self, output_path: Optional[str] = None):
        # generate a single breakbeat sample (audio mode)
        if self.mode != "audio":
            raise ValueError("This method is only available in audio mode")
        
        return self.audio_generator.generate_sample(output_path)
    
    def generate_multiple_samples(self, num_samples: int = 5, 
                                output_dir: str = "generated_breaks"):
        # generate multiple breakbeat samples (audio mode)
        if self.mode != "audio":
            raise ValueError("This method is only available in audio mode")
        
        return self.audio_generator.generate_multiple_samples(num_samples, output_dir)
    

    
    # sequence generation methods
    def load_breakbeat(self, file_path: str, num_slices: int = 16):
        # load and slice a breakbeat file (sequence mode)
        if self.mode != "sequence":
            raise ValueError("This method is only available in sequence mode")
        
        audio = self.slicer.load_audio(file_path)
        self.slices = self.slicer.slice_equal(audio, num_slices)   
        self.renderer.set_slices(self.slices)
        # store the requested num_slices and use it for the generator
        self.num_slices = num_slices
        self.sequence_generator = SequenceGenerator(num_slices)
    
    def generate_sequence(self, length: Optional[int] = None, temperature: float = 1.0) -> List[int]:
        # generate a sequence (sequence mode)
        if self.mode != "sequence":
            raise ValueError("This method is only available in sequence mode")
        
        # use the requested num_slices as default length if not specified
        if length is None:
            length = self.num_slices
        
        return self.sequence_generator.generate_sequence(length, temperature)
    

    
    def render_sequence(self, sequence: List[int], output_path: str, volume: float = 0.8):
        # render a sequence to wav file (sequence mode)
        if self.mode != "sequence":
            raise ValueError("This method is only available in sequence mode")
        
        self.renderer.set_volume(volume)
        return self.renderer.render_sequence(sequence, output_path)
    
    # common methods
    def save_model(self, filepath: str):
        # save the trained model
        if self.mode == "audio":
            self.audio_generator.save_model(filepath)
        else:
            raise ValueError("Model saving is only available in audio mode")
    
    def load_model(self, filepath: str):
        # load a trained model
        if self.mode == "audio":
            self.audio_generator.load_model(filepath)
        else:
            raise ValueError("Model loading is only available in audio mode")