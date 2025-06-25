#!/usr/bin/env python3
"""
Minimal training script that works with W&B Sweep
This is a simplified version for testing sweep functionality
"""

import os
import wandb
import torch
import numpy as np
from sklearn.metrics import f1_score

# Import your existing components
from minimal_test import MinimalMSPDataset, MinimalSERModel, SimpleFocalLoss, Config

def train_with_config():
    """Training function that uses W&B config"""
    # Initialize W&B run (will be called by sweep agent)
    run = wandb.init()
    config = wandb.config  # Get hyperparameters from sweep
    
    # Override default config with sweep parameters
    train_config = Config()
    
    # Apply sweep parameters
    if hasattr(config, 'batch_size'):
        train_config.batch_size = config.batch_size
    if hasattr(config, 'learning_rate'):
        train_config.learning_rate = config.learning_rate
    if hasattr(config, 'hidden_size'):
        train_config.hidden_size = config.hidden_size
    if hasattr(config, 'dropout'):
        train_config.dropout = config.dropout
    if hasattr(config, 'focal_gamma'):
        focal_gamma = config.focal_gamma
    else:
        focal_gamma = 2.0
    
    # Log configuration
    wandb.log({"config": dict(config)})
    
    # Device
    device = torch.device('cpu')  # For Mac testing
    
    # Load data
    train_json = os.path.join(train_config.data_root, 'msp_train_10class.json')
    valid_json = os.path.join(train_config.data_root, 'msp_valid_10class.json')
    
    train_dataset = MinimalMSPDataset(train_json, train_config.data_root, train_config, is_train=True)
    valid_dataset = MinimalMSPDataset(valid_json, train_config.data_root, train_config, is_train=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_config.batch_size, 
        shuffle=True, 
        num_workers=0
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=train_config.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    # Model
    model = MinimalSERModel(train_config).to(device)
    
    # Loss and optimizer
    criterion = SimpleFocalLoss(gamma=focal_gamma)
    
    if hasattr(config, 'optimizer'):
        if config.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
        elif config.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=train_config.learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    
    # Training loop (simplified for sweep)
    best_valid_f1 = 0
    num_epochs = 5  # Quick test
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in train_loader:
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
        
        avg_train_loss = train_loss / len(train_loader)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        
        # Validate
        model.eval()
        valid_loss = 0
        valid_preds = []
        valid_labels = []
        
        with torch.no_grad():
            for batch in valid_loader:
                waveforms, labels = batch
                waveforms = waveforms.to(device)
                labels = labels.to(device)
                
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                valid_preds.extend(preds.cpu().numpy())
                valid_labels.extend(labels.cpu().numpy())
        
        avg_valid_loss = valid_loss / len(valid_loader)
        valid_f1 = f1_score(valid_labels, valid_preds, average='macro')
        
        # Log to W&B
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_f1': train_f1,
            'valid_loss': avg_valid_loss,
            'valid_f1': valid_f1,
        })
        
        # Update best F1
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
        
        print(f"Epoch {epoch+1}: Train F1={train_f1:.3f}, Valid F1={valid_f1:.3f}")
    
    # Log final best F1
    wandb.log({'best_valid_f1': best_valid_f1})
    
    return best_valid_f1

if __name__ == "__main__":
    train_with_config()