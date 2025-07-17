#!/usr/bin/env python3
import argparse
import sys
import os
import yaml
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vae import BreakbeatVAE
from src.training.trainer import VAETrainer
from src.utils.files import find_breakbeat_files

def load_config():
    with open("config/config.yaml", 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--sample-length', type=int, default=None)
    args = parser.parse_args()

    config = load_config()

    latent_dim = config['model']['latent_dim']
    hidden_dims = config['model']['hidden_dims']
    epochs = args.epochs or config['training']['epochs']
    batch_size = args.batch_size or config['training']['batch_size']
    learning_rate = args.learning_rate or config['training']['learning_rate']
    sample_length = args.sample_length or config['generation']['sample_length']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BreakbeatVAE(latent_dim=latent_dim, hidden_dims=hidden_dims).to(device)
    trainer = VAETrainer(model, device)
    
    audio_files = find_breakbeat_files(args.data_dir)
    trainer.train(audio_files, epochs, batch_size, learning_rate, sample_length=sample_length)
    print("Training completed. Best model saved as models/audio_vae_best.pth")

if __name__ == '__main__':
    main() 