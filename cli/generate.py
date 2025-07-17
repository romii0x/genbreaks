#!/usr/bin/env python3
import argparse
import sys
import os
import yaml
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vae import BreakbeatVAE
from src.generation.generator import BreakbeatGenerator

def load_config():
    with open("config/config.yaml", 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--sample-length', type=int, default=None)
    args = parser.parse_args()
    
    config = load_config()
    
    latent_dim = config['model']['latent_dim']
    hidden_dims = config['model']['hidden_dims']
    sample_length = args.sample_length or config['generation']['sample_length']

    model_path = args.model_path or config['generation']['model_path']
    num_samples = args.num_samples or config['generation']['num_samples']
    output_dir = args.output_dir or config['generation']['output_dir']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BreakbeatVAE(latent_dim=latent_dim, hidden_dims=hidden_dims).to(device)
    generator = BreakbeatGenerator(model, device, sample_length=sample_length)
    generator.model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    generator.generate_samples(num_samples, output_dir)

if __name__ == '__main__':
    main() 