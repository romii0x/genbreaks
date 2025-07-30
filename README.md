# genbreaks
 
variational autoencoder for generating rhythm-based audio. this is a personal research project to help me learn more about ML and PyTorch. 

## Overview

This project trains a VAE on audio samples to generate new breakbeat patterns. It processes audio files by splitting them into chunks, converting to spectrograms, and learning to reconstruct them.

## Installation

```bash
make install
source .venv/bin/activate
```

## Usage

Train on your audio samples:
```bash
python cli/train.py data_dir
```
- `data_dir`: directory containing WAV files to train on
- `--epochs`: number of training epochs
- `--batch-size`: batch size
- `--learning-rate`: learning rate
- `--sample-length`: audio sample length in samples

Generate new samples:
```bash
python cli/generate.py
```
- `--model-path`: path to trained model
- `--num-samples`: number of samples to generate
- `--output-dir`: output directory
- `--sample-length`: generated audio length in samples

## Configuration

Edit `config/config.yaml` to adjust model parameters, training settings, and generation options.

## Architecture

- convolutional VAE with encoder/decoder
- mel spectrogram processing
- griffin-Lim audio reconstruction

## License

[MIT](LICENSE)
