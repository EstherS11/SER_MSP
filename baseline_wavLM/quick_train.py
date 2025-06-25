#!/usr/bin/env python3
"""
Enhanced quick training script with W&B Sweep support and test set evaluation
"""

import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, classification_report
import wandb
from datetime import datetime
import logging
from pathlib import Path

# Import from main training script
from train_msp_podcast import (
    MSPPodcastDataset, 
    WavLMECAPAClassifier, 
    FocalLoss,
    set_seed
)

# Configure logging
def setup_logging(log_dir, exp_name):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{exp_name}.log"
    
    # Create logger
    logger = logging.getLogger('SER_Training')
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler (detailed logs)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler (simple logs)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced training script with W&B Sweep support')
    
    # Data
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model
    parser.add_argument('--freeze_wavlm', action='store_true', default=True)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--wavlm_model', type=str, default='microsoft/wavlm-large')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='adamw', 
                        choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'onecycle', 'none'])
    
    # Data augmentation
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--augment_prob', type=float, default=0.5)
    
    # Loss
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--use_class_weights', action='store_true', default=True)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_sweep', action='store_true',
                        help='Running as part of W&B sweep')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='experiments')
    parser.add_argument('--log_interval', type=int, default=10)
    
    return parser.parse_args()

def get_optimizer(model, args):
    """Get optimizer based on configuration"""
    if args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), 
                               lr=args.learning_rate, 
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), 
                                lr=args.learning_rate, 
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), 
                              lr=args.learning_rate, 
                              weight_decay=args.weight_decay,
                              momentum=0.9)

def get_scheduler(optimizer, args, steps_per_epoch):
    """Get learning rate scheduler based on configuration"""
    if args.scheduler == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
    elif args.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    elif args.scheduler == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.learning_rate, 
            epochs=args.epochs, steps_per_epoch=steps_per_epoch
        )
    else:
        return None

def train_one_epoch(model, dataloader, criterion, optimizer, device, logger):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (waveforms, labels, _) in enumerate(dataloader):
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if batch_idx % 10 == 0:
            logger.debug(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, macro_f1

def evaluate(model, dataloader, criterion, device, logger, return_predictions=False):
    """Evaluate model on a dataset"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for waveforms, labels, _ in dataloader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    if return_predictions:
        return avg_loss, macro_f1, all_preds, all_labels
    return avg_loss, macro_f1

def main():
    args = parse_args()
    
    # If running as W&B sweep, initialize with sweep config
    if args.wandb_sweep:
        wandb.init()
        # Override args with sweep parameters
        for key, value in wandb.config.items():
            setattr(args, key, value)
    
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Auto-detect data root
    if args.data_root is None:
        if os.path.exists('/data/user_data/esthers/SER_MSP'):
            args.data_root = '/data/user_data/esthers/SER_MSP'
        else:
            args.data_root = '/Users/esthersun/Desktop/SER/SER_MSP'
    
    # Create experiment directory
    if args.exp_name is None:
        args.exp_name = f"msp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    exp_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(exp_dir, args.exp_name)
    logger.info(f"Starting experiment: {args.exp_name}")
    logger.info(f"Using device: {device}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Save config
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Initialize W&B if not in sweep mode
    if args.use_wandb and not args.wandb_sweep:
        wandb.init(
            project="msp-podcast-ser",
            name=args.exp_name,
            config=args
        )
    
    # Load data
    logger.info("Loading datasets...")
    train_dataset = MSPPodcastDataset(
        os.path.join(args.data_root, 'msp_train_10class.json'),
        args.data_root,
        augment=args.augment
    )
    valid_dataset = MSPPodcastDataset(
        os.path.join(args.data_root, 'msp_valid_10class.json'),
        args.data_root,
        augment=False
    )
    test_dataset = MSPPodcastDataset(
        os.path.join(args.data_root, 'msp_test_10class.json'),
        args.data_root,
        augment=False
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
    
    # Calculate class weights
    if args.use_class_weights:
        class_counts = train_dataset.class_counts
        total_samples = sum(class_counts.values())
        class_weights = torch.tensor([
            total_samples / (10 * count) for count in class_counts.values()
        ])
        class_weights = class_weights / class_weights.sum() * 10
        logger.info(f"Using class weights: {class_weights}")
    else:
        class_weights = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = WavLMECAPAClassifier(
        num_classes=10,
        wavlm_name=args.wavlm_model,
        freeze_wavlm=args.freeze_wavlm
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params/1e6:.2f}M")
    logger.info(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    
    # Loss and optimizer
    criterion = FocalLoss(
        alpha=class_weights.to(device) if class_weights is not None else None,
        gamma=args.focal_gamma
    )
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args, len(train_loader))
    
    # Training loop
    best_valid_f1 = 0
    patience_counter = 0
    
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )
        
        # Validate
        valid_loss, valid_f1 = evaluate(
            model, valid_loader, criterion, device, logger
        )
        
        # Update scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(valid_f1)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log results
        logger.info(f"Train - Loss: {train_loss:.4f}, Macro F1: {train_f1:.4f}")
        logger.info(f"Valid - Loss: {valid_loss:.4f}, Macro F1: {valid_f1:.4f}")
        logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # Log to W&B
        if args.use_wandb or args.wandb_sweep:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_f1': train_f1,
                'valid_loss': valid_loss,
                'valid_f1': valid_f1,
                'lr': current_lr
            })
        
        # Save best model
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_f1': best_valid_f1,
                'args': args
            }
            torch.save(checkpoint, os.path.join(exp_dir, 'best_model.pth'))
            logger.info(f"Saved best model with Valid F1: {best_valid_f1:.4f}")
            
            # Update W&B summary
            if args.use_wandb or args.wandb_sweep:
                wandb.run.summary['best_valid_f1'] = best_valid_f1
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model and evaluate on test set
    logger.info("\n=== Final Test Set Evaluation ===")
    checkpoint = torch.load(os.path.join(exp_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, logger, return_predictions=True
    )
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Macro F1: {test_f1:.4f}")
    
    # Detailed classification report
    emotion_names = [
        "neutral", "happy", "sad", "angry", "fear",
        "disgust", "surprise", "contempt", "other", "unknown"
    ]
    report = classification_report(test_labels, test_preds, 
                                 target_names=emotion_names)
    logger.info(f"\nClassification Report:\n{report}")
    
    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_f1': test_f1,
        'classification_report': report,
        'predictions': test_preds,
        'labels': test_labels
    }
    
    with open(os.path.join(exp_dir, 'test_results.json'), 'w') as f:
        json.dump({k: v for k, v in test_results.items() 
                  if k not in ['predictions', 'labels']}, f, indent=4)
    
    # Log to W&B
    if args.use_wandb or args.wandb_sweep:
        wandb.run.summary['test_f1'] = test_f1
        wandb.run.summary['test_loss'] = test_loss
        wandb.log({
            'test_f1': test_f1,
            'test_loss': test_loss
        })
    
    logger.info(f"\nTraining completed! Best Valid F1: {best_valid_f1:.4f}, Test F1: {test_f1:.4f}")
    logger.info(f"All results saved in: {exp_dir}")
    
    if args.use_wandb or args.wandb_sweep:
        wandb.finish()

if __name__ == "__main__":
    main()