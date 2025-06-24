#!/usr/bin/env python3
"""
Pure PyTorch implementation - WavLM + ECAPA-TDNN for emotion classification
No SpeechBrain dependency!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import WavLMModel
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, classification_report
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple ECAPA-TDNN implementation
class ECAPA_TDNN(nn.Module):
    def __init__(self, input_size=1024, lin_neurons=192):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 512, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(512, 1536, kernel_size=1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1536)
        
        self.fc = nn.Linear(1536, lin_neurons)
        
    def forward(self, x):
        # x shape: (batch, time, features) -> (batch, features, time)
        x = x.transpose(1, 2)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global pooling
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        
        return x

# Simple Emotion Model
class EmotionModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Load WavLM
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
        self.wavlm.freeze_feature_encoder()
        
        # Feature projection
        self.projection = nn.Linear(1024, 1024)
        
        # ECAPA-TDNN
        self.ecapa = ECAPA_TDNN(input_size=1024, lin_neurons=192)
        
        # Classifier
        self.classifier = nn.Linear(192, num_classes)
        
    def forward(self, wavs):
        # Extract WavLM features
        with torch.no_grad():
            outputs = self.wavlm(wavs, output_hidden_states=True)
            features = outputs.hidden_states[-1]  # Last layer features
        
        # Project features
        features = self.projection(features)
        
        # ECAPA-TDNN
        embeddings = self.ecapa(features)
        
        # Classification
        logits = self.classifier(embeddings)
        
        return logits

# Simple Dataset
class MSPDataset(Dataset):
    def __init__(self, json_file, data_root, max_length=5*16000):  # 5 seconds
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.items = list(self.data.items())
        self.data_root = data_root
        self.max_length = max_length
        
        # Emotion mapping
        self.emo_map = {
            'N': 0, 'H': 1, 'X': 2, 'A': 3, 'S': 4,
            'U': 5, 'C': 6, 'O': 7, 'D': 8, 'F': 9
        }
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        key, item = self.items[idx]
        
        # Load audio
        wav_path = item['wav'].replace('{data_root}', self.data_root)
        try:
            wav, sr = torchaudio.load(wav_path)
            
            # Convert to mono
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                wav = resampler(wav)
            
            wav = wav.squeeze(0)
            
            # Pad or truncate
            if wav.shape[0] < self.max_length:
                wav = F.pad(wav, (0, self.max_length - wav.shape[0]))
            else:
                wav = wav[:self.max_length]
                
        except Exception as e:
            logger.error(f"Error loading {wav_path}: {e}")
            wav = torch.zeros(self.max_length)
        
        # Get label
        emo = item['emo']
        label = self.emo_map.get(emo, 0)
        
        return wav, label

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (wavs, labels) in enumerate(tqdm(dataloader, desc="Training")):
        wavs = wavs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(wavs)
        loss = F.cross_entropy(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Predictions
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if batch_idx % 100 == 0:
            logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Calculate metrics
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return total_loss / len(dataloader), macro_f1

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for wavs, labels in tqdm(dataloader, desc="Evaluating"):
            wavs = wavs.to(device)
            labels = labels.to(device)
            
            logits = model(wavs)
            loss = F.cross_entropy(logits, labels)
            
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Print classification report
    emotion_names = ['Neutral', 'Happy', 'Excited', 'Angry', 'Sad',
                     'Surprise', 'Contempt', 'Other', 'Disgust', 'Fear']
    report = classification_report(all_labels, all_preds, 
                                 target_names=emotion_names[:len(np.unique(all_labels))],
                                 digits=3)
    
    return total_loss / len(dataloader), macro_f1, report

def main():
    # Configuration
    data_folder = "/data/user_data/esthers/DATA"
    batch_size = 8
    num_epochs = 1
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = MSPDataset("msp_train_10class.json", data_folder)
    valid_dataset = MSPDataset("msp_valid_10class.json", data_folder)
    test_dataset = MSPDataset("msp_test_10class.json", data_folder)
    
    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Valid size: {len(valid_dataset)}")
    logger.info(f"Test size: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=2, pin_memory=True)
    
    # Create model
    model = EmotionModel(num_classes=10).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, device)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Macro-F1: {train_f1:.4f}")
        
        # Validate
        valid_loss, valid_f1, _ = evaluate(model, valid_loader, device)
        logger.info(f"Valid Loss: {valid_loss:.4f}, Valid Macro-F1: {valid_f1:.4f}")
    
    # Test
    logger.info("\nFinal Test Evaluation:")
    test_loss, test_f1, report = evaluate(model, test_loader, device)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Macro-F1: {test_f1:.4f}")
    logger.info(f"\nClassification Report:\n{report}")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'test_f1': test_f1
    }, 'emotion_model.pth')
    logger.info("Model saved!")

if __name__ == "__main__":
    main()