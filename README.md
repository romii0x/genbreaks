```
                                __                     __          
             .-----.-----.-----|  |--.----.-----.---.-|  |--.-----.
             |  _  |  -__|     |  _  |   _|  -__|  _  |    <|__ --|
             |___  |_____|__|__|_____|__| |_____|___._|__|__|_____|
             |_____|                                               
```
breakbeat generation system with neural audio synthesis and lstm-based sequence generation.

## Architecture

### audio generation mode
- autoencoder neural network trained on breakbeat samples
- generates raw audio from scratch using pytorch
- output: `generated_breaks/` directory

### sequence generation mode  
- slices breakbeats into equal segments
- lstm neural network learns musical patterns from training sequences
- generates new sequences that follow learned breakbeat patterns
- output: `sliced_breaks/` directory

## Installation

```bash
./install.sh
source .venv/bin/activate
```

## Usage

### audio generation
```python
from src.main import GenBreaks

# initialize audio mode
genbreaks = GenBreaks(mode="audio", sample_length=44100)

# train on breakbeat samples
genbreaks.train_on_directory("breaks", epochs=2500, batch_size=4)

# generate samples
genbreaks.generate_multiple_samples(5, "generated_breaks")

# save/load model
genbreaks.save_model("model.pth")
genbreaks.load_model("model.pth")
```

### sequence generation
```python
from src.main import GenBreaks

# initialize sequence mode
genbreaks = GenBreaks(mode="sequence")

# load and slice breakbeat (16 slices default)
genbreaks.load_breakbeat("breaks/amen.wav", num_slices)  # choose any number of slices

# generate sequences
sequence = genbreaks.generate_sequence(length=16, temperature=1.0)
genbreaks.render_sequence(sequence, "sliced_breaks/output.wav")
```

## API reference

### GenBreaks Class

#### audio mode methods
- `train_on_directory(directory, epochs, batch_size, learning_rate)`
- `generate_sample(output_path=None)`
- `generate_multiple_samples(num_samples, output_dir)`
- `save_model(filepath)`
- `load_model(filepath)`

#### sequence mode methods
- `load_breakbeat(file_path, num_slices)`
- `generate_sequence(length, temperature)`
- `render_sequence(sequence, output_path, volume)`

## Demo

```bash
python demo.py
```

choose between audio generation, sequence generation, or both.

## Dependencies

- PyTorch
- librosa
- soundfile
- sounddevice
- numpy

## Files

```
genbreaks/
├── breaks/                    # Input breakbeat samples
├── generated_breaks/          # Neural audio generation output
├── sliced_breaks/             # Sequence generation output
├── src/
│   ├── main.py                # Dual-mode interface
│   ├── break_generator.py     # Neural audio generation
│   ├── sequence_generator.py  # LSTM sequence generation
│   ├── audio_slicer.py        # Audio slicing
│   └── sequence_renderer.py   # Sequence rendering
└── demo.py                    # Demo script
```


## Training Data

the sequence generator comes with built-in training data 

TBF

## Finding Breakbeats

### free sources
- add free breakbeat resources WITHOUT LOGIN here

## License

MIT License - see LICENSE file for details.