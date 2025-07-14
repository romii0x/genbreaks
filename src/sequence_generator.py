"""
sequence generator module for genbreaks

this module handles generating musical sequences using a lstm trained on breakbeat patterns.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import random


class SequenceDataset(Dataset):
    # dataset for training on sequence patterns
    
    def __init__(self, sequences: List[List[int]], sequence_length: int = 16):
        # initialize the dataset
        # args:
        #   sequences: list of training sequences
        #   sequence_length: length of each sequence
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.processed_sequences = []
        
        # process sequences into training data
        for seq in sequences:
            if len(seq) >= sequence_length:
                # create sliding windows
                for i in range(len(seq) - sequence_length + 1):
                    window = seq[i:i + sequence_length]
                    self.processed_sequences.append(window)
    
    def __len__(self):
        return len(self.processed_sequences)
    
    def __getitem__(self, idx):
        sequence = self.processed_sequences[idx]
        # 1-16 -> 0-15
        sequence = [x - 1 for x in sequence]
        return torch.tensor(sequence, dtype=torch.long)


class SequenceModel(nn.Module):
    # neural network for generating musical sequences
    
    def __init__(self, num_slices: int = 16, hidden_size: int = 128, num_layers: int = 2):
        # initialize the sequence generator
        # args:
        #   num_slices: number of slices available
        #   hidden_size: size of hidden layers
        #   num_layers: number of lstm layers
        super().__init__()
        self.num_slices = num_slices
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # embedding layer
        self.embedding = nn.Embedding(num_slices, hidden_size)
        
        # lstm layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True
        )
        
        # output layer
        self.fc = nn.Linear(hidden_size, num_slices)
        
    def forward(self, x, hidden=None):
        # forward pass
        # args:
        #   x: input sequence (batch_size, seq_len)
        #   hidden: optional hidden state
        # returns:
        #   output: logits for next slice
        #   hidden: hidden state
        
        # embed input
        embedded = self.embedding(x)
        
        # lstm forward pass
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # output layer
        output = self.fc(lstm_out)
        
        return output, hidden
    
    def generate_sequence(self, length: int = 16, temperature: float = 1.0, 
                         start_sequence: Optional[List[int]] = None) -> List[int]:
        # generate a new sequence
        # args:
        #   length: length of sequence to generate
        #   temperature: randomness factor (higher = more random)
        #   start_sequence: optional starting sequence
        # returns:
        #   generated sequence
        self.eval()
        
        with torch.no_grad():
            device = next(self.parameters()).device
            
            # initialize sequence
            if start_sequence:
                sequence = [x - 1 for x in start_sequence]  # convert to 0-based
                current_length = len(sequence)
            else:
                sequence = [random.randint(0, self.num_slices - 1)]
                current_length = 1
            
            # initialize hidden state
            hidden = None
            
            # generate sequence
            while current_length < length:
                # prepare input
                x = torch.tensor([sequence[-1]], dtype=torch.long).unsqueeze(0).to(device)
                
                # forward pass
                output, hidden = self.forward(x, hidden)
                
                # get logits for next slice
                logits = output[0, -1] / temperature
                
                # sample next slice
                probs = torch.softmax(logits, dim=0)
                next_slice = torch.multinomial(probs, 1).item()
                
                sequence.append(next_slice)
                current_length += 1
            
            # convert back to 1-based indexing
            return [x + 1 for x in sequence]


class SequenceGenerator:
    # generates slice sequences from breakbeat style slice sequences using the lstm
    
    def __init__(self, num_slices: int = 16):
        # initialize the slice generator
        # args:
        #   num_slices: number of slices available for generation
        self.num_slices = num_slices
        self.training_sequences = []
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def create_training_data(self) -> List[List[int]]:
        # create training sequences data for the generator
        # returns:
        #   list of training sequences
        sequences = []
        
        # simple patterns that respect the slice count
        sequences.append(list(range(1, self.num_slices + 1)))  # straight through
        
        # reverse pattern
        sequences.append(list(range(self.num_slices, 0, -1)))
        
        # simple alternating (first half, then second half)
        if self.num_slices >= 2:
            mid = self.num_slices // 2
            first_half = list(range(1, mid + 1))
            second_half = list(range(mid + 1, self.num_slices + 1))
            sequences.append(first_half + second_half)
        
        # ensure all sequences are exactly num_slices long
        for i, seq in enumerate(sequences):
            if len(seq) != self.num_slices:
                # pad or truncate to exact length
                if len(seq) < self.num_slices:
                    # repeat the sequence to fill
                    while len(seq) < self.num_slices:
                        seq.extend(seq[:min(len(seq), self.num_slices - len(seq))])
                sequences[i] = seq[:self.num_slices]
        
        # simple repeating pattern
        if self.num_slices >= 4:
            pattern = [1, 2, 1, 2] * (self.num_slices // 4)
            sequences.append(pattern[:self.num_slices])
        
        # balanced patterns that use all slices
        import random
        for _ in range(5):
            # create pattern that uses all slices at least once
            pattern = list(range(1, self.num_slices + 1))
            random.shuffle(pattern)
            sequences.append(pattern)
        
        # patterns that repeat shorter sequences to fill the length
        if self.num_slices >= 4:
            for _ in range(3):
                # create a shorter pattern and repeat it
                short_pattern = list(range(1, min(5, self.num_slices + 1)))
                random.shuffle(short_pattern)
                # repeat to fill the length
                pattern = []
                while len(pattern) < self.num_slices:
                    pattern.extend(short_pattern)
                sequences.append(pattern[:self.num_slices])
        
        self.training_sequences = sequences
        return sequences
    
    def train(self, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        # train the neural network on training sequences
        # args:
        #   epochs: number of training epochs
        #   batch_size: batch size for training
        #   learning_rate: learning rate for optimizer
        if not self.training_sequences:
            self.create_training_data()
        
        # create model
        self.model = SequenceModel(self.num_slices)
        self.model.to(self.device)
        
        # create dataset and dataloader
        # use num_slices for sequence length
        dataset = SequenceDataset(self.training_sequences, sequence_length=self.num_slices)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print(f"training sequence generator on {len(dataset)} sequences for {epochs} epochs")
        
        # training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # prepare input and target
                input_seq = batch[:, :-1]
                target_seq = batch[:, 1:]
                
                # forward pass
                optimizer.zero_grad()
                output, _ = self.model(input_seq)
                
                # calculate loss
                loss = criterion(output.reshape(-1, self.num_slices), target_seq.reshape(-1))
                
                # backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # print progress
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"epoch {epoch + 1}/{epochs}, loss: {avg_loss:.6f}")
        
        print("sequence generator training completed")
    
    def generate_sequence(self, length: int = 16, temperature: float = 1.0) -> List[int]:
        # generate a new beat sequence using trained neural network
        # args:
        #   length: length of sequence to generate
        #   temperature: higher = more random
        # returns:
        #   list of slice numbers
        if self.model is None:
            # fallback to training if model not trained
            self.train(epochs=50)
        
        sequence = self.model.generate_sequence(length, temperature)
        print(f"generated sequence length: {len(sequence)}, expected: {length}")
        return sequence
    
    def save_model(self, filepath: str):
        # save the trained model
        if self.model is not None:
            torch.save(self.model.state_dict(), filepath)
            print(f"sequence generator model saved to {filepath}")
    
    def load_model(self, filepath: str):
        # load a trained model
        self.model = SequenceModel(self.num_slices)
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)
        print(f"sequence generator model loaded from {filepath}")