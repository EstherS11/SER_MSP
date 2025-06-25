#!/usr/bin/env python3
"""
Lightweight test script - suitable for local Mac debugging
CPU only, small batch size, limited data
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')
import time
from tqdm import tqdm

# Simplified configuration
class Config:
    # Data paths
    data_root = '/Users/esthersun/Desktop/SER/SER_MSP'
    
    # Model parameters - use smaller model
    use_small_model = True  # Use wav2vec2-base instead of WavLM-large
    hidden_size = 256  # Reduce hidden layer size
    
    # Training parameters - small scale test
    batch_size = 4  # Mac-friendly batch size
    num_epochs = 3  # Only run a few epochs for testing
    learning_rate = 1e-3
    
    # Data parameters
    max_length = 3.0  # Only use 3 seconds of audio to reduce memory
    sample_rate = 16000
    
    # Others
    device = 'cpu'  # Use CPU on Mac
    num_workers = 0  # Set to 0 on Mac to avoid issues
    use_subset = True  # Only use a subset of data
    subset_size = 100  # Only use 100 samples per set

# Ultra-simplified dataset class
class MinimalMSPDataset(Dataset):
    def __init__(self, json_path, root_dir, config, is_train=False):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # If using subset, only take first N samples
        if config.use_subset:
            keys = list(self.data.keys())[:config.subset_size]
            self.data = {k: self.data[k] for k in keys}
        
        self.keys = list(self.data.keys())
        self.root_dir = root_dir
        self.config = config
        self.is_train = is_train
        
        self.emotion_map = {
            'N': 0, 'H': 1, 'S': 2, 'A': 3, 'F': 4,
            'D': 5, 'U': 6, 'C': 7, 'O': 8, 'X': 9
        }
        
        self.max_samples = int(config.max_length * config.sample_rate)
        print(f"Dataset initialized with {len(self.keys)} samples")
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data[key]
        
        # Load audio
        wav_path = os.path.join(self.root_dir, item['wav'])
        
        try:
            waveform, sr = torchaudio.load(wav_path)
            
            # Resample if needed
            if sr != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Crop or pad
            if waveform.shape[1] > self.max_samples:
                # Random crop (training) or center crop (testing)
                if self.is_train:
                    start = np.random.randint(0, waveform.shape[1] - self.max_samples)
                    waveform = waveform[:, start:start + self.max_samples]
                else:
                    waveform = waveform[:, :self.max_samples]
            else:
                # Pad
                pad_length = self.max_samples - waveform.shape[1]
                waveform = F.pad(waveform, (0, pad_length))
            
            # Simple data augmentation (training only)
            if self.is_train and np.random.random() < 0.3:
                # Add small noise
                noise = torch.randn_like(waveform) * 0.003
                waveform = waveform + noise
            
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            # Return silence
            waveform = torch.zeros(1, self.max_samples)
        
        # Get label
        label = self.emotion_map[item['emo']]
        
        return waveform.squeeze(0), label

# Minimal model - using smaller architecture
class MinimalSERModel(nn.Module):
    def __init__(self, config):
        super(MinimalSERModel, self).__init__()
        
        if config.use_small_model:
            # Option 1: Use wav2vec2-base (smaller)
            from transformers import Wav2Vec2Model
            self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            encoder_dim = 768
            
            # Freeze most layers, only fine-tune last few
            for param in self.encoder.parameters():
                param.requires_grad = False
            # Only unfreeze last 2 layers
            for param in self.encoder.encoder.layers[-2:].parameters():
                param.requires_grad = True
        else:
            # Option 2: Simple CNN (smallest)
            self.encoder = None
            self.conv_layers = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            encoder_dim = 256
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_size, 10)
        )
    
    def forward(self, x):
        if self.encoder is not None:
            # Use pretrained encoder
            outputs = self.encoder(x)
            # Average pooling
            features = torch.mean(outputs.last_hidden_state, dim=1)
        else:
            # Use CNN
            x = x.unsqueeze(1)  # Add channel dimension
            features = self.conv_layers(x).squeeze(-1)
        
        logits = self.classifier(features)
        return logits

# Simplified Focal Loss
class SimpleFocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(SimpleFocalLoss, self).__init__()
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Quick training function
def quick_train():
    print("=== Lightweight MSP-PODCAST Test ===")
    
    # Configuration
    config = Config()
    
    # Set device
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_json = os.path.join(config.data_root, 'msp_train_10class.json')
    valid_json = os.path.join(config.data_root, 'msp_valid_10class.json')
    
    train_dataset = MinimalMSPDataset(train_json, config.data_root, config, is_train=True)
    valid_dataset = MinimalMSPDataset(valid_json, config.data_root, config, is_train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers
    )
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(valid_dataset)}")
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(valid_loader)}")
    
    # Create model
    print("\nInitializing model...")
    model = MinimalSERModel(config).to(device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    
    # Loss function and optimizer
    criterion = SimpleFocalLoss(gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    print("\nStarting training...")
    best_valid_f1 = 0
    
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs} [Train]')
        for batch in train_pbar:
            waveforms, labels = batch
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        
        # Validation phase
        model.eval()
        valid_loss = 0
        valid_preds = []
        valid_labels = []
        
        with torch.no_grad():
            valid_pbar = tqdm(valid_loader, desc=f'Epoch {epoch+1}/{config.num_epochs} [Valid]')
            for batch in valid_pbar:
                waveforms, labels = batch
                waveforms = waveforms.to(device)
                labels = labels.to(device)
                
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                valid_preds.extend(preds.cpu().numpy())
                valid_labels.extend(labels.cpu().numpy())
                
                valid_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_valid_loss = valid_loss / len(valid_loader)
        valid_f1 = f1_score(valid_labels, valid_preds, average='macro')
        
        # Print results
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Macro F1: {train_f1:.4f}")
        print(f"  Valid - Loss: {avg_valid_loss:.4f}, Macro F1: {valid_f1:.4f}")
        
        # Save best model
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_f1': best_valid_f1,
            }, 'minimal_best_model.pth')
            print(f"  Saved best model (F1: {best_valid_f1:.4f})")
    
    print(f"\nTraining completed! Best validation F1: {best_valid_f1:.4f}")
    print("Model saved to: minimal_best_model.pth")

# Test data loading
def test_data_loading():
    print("=== Testing Data Loading ===")
    config = Config()
    
    train_json = os.path.join(config.data_root, 'msp_train_10class.json')
    dataset = MinimalMSPDataset(train_json, config.data_root, config)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a few samples
    for i in range(min(5, len(dataset))):
        try:
            waveform, label = dataset[i]
            print(f"Sample {i}: waveform shape={waveform.shape}, label={label}")
        except Exception as e:
            print(f"Sample {i} loading failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Only test data loading
        test_data_loading()
    else:
        # Run full training
        quick_train()