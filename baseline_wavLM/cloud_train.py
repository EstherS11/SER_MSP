#!/usr/bin/env python3
"""
cloud_train.py - Full training script for cloud GPU
"""

import argparse
import os
import json
import torch
import wandb
from datetime import datetime

# Import from your main training script
from train_msp_podcast import (
    MSPPodcastDataset, 
    WavLMECAPAClassifier, 
    FocalLoss,
    set_seed,
    train_epoch,
    validate
)

def main():
    parser = argparse.ArgumentParser(description='Cloud training for MSP-PODCAST')
    
    # Essential arguments
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (32 recommended for GPU)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma')
    
    # Optional arguments
    parser.add_argument('--freeze_wavlm', action='store_true', default=True)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project', type=str, default='MSP-PODCAST-Cloud')
    
    args = parser.parse_args()
    
    # Set experiment name
    if args.exp_name is None:
        args.exp_name = f"cloud_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Data paths (cloud)
    root_dir = '/data/user_data/esthers/SER_MSP'
    
    # Initialize W&B
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.exp_name,
            config=args
        )
    
    # Create save directory
    save_dir = os.path.join('experiments', args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Load datasets (full data, not subset)
    print("Loading full datasets...")
    train_dataset = MSPPodcastDataset(
        os.path.join(root_dir, 'msp_train_10class.json'),
        root_dir,
        augment=True  # Enable augmentation for full training
    )
    valid_dataset = MSPPodcastDataset(
        os.path.join(root_dir, 'msp_valid_10class.json'),
        root_dir,
        augment=False
    )
    test_dataset = MSPPodcastDataset(
        os.path.join(root_dir, 'msp_test_10class.json'),
        root_dir,
        augment=False
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model
    print("Initializing WavLM-ECAPA model...")
    model = WavLMECAPAClassifier(
        num_classes=10,
        freeze_wavlm=args.freeze_wavlm
    ).to(device)
    
    # Calculate class weights
    class_counts = train_dataset.class_counts
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor([
        total_samples / (10 * count) for count in class_counts.values()
    ])
    class_weights = class_weights / class_weights.sum() * 10
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=args.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    best_valid_f1 = 0
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        valid_loss, valid_f1, _, _ = validate(model, valid_loader, criterion, device)
        
        # Scheduler step
        scheduler.step(valid_f1)
        
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Valid F1: {valid_f1:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Log to W&B
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_f1': train_f1,
                'valid_loss': valid_loss,
                'valid_f1': valid_f1,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_f1': best_valid_f1,
                'args': args
            }, os.path.join(save_dir, 'best_model.pth'))
            
            print(f"Saved best model with Valid F1: {best_valid_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Test evaluation
    print("\n=== Test Set Evaluation ===")
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_f1, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    
    # Save results
    results = {
        'best_valid_f1': best_valid_f1,
        'test_f1': test_f1,
        'test_loss': test_loss,
        'args': vars(args)
    }
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    if args.use_wandb:
        wandb.run.summary['best_valid_f1'] = best_valid_f1
        wandb.run.summary['test_f1'] = test_f1
        wandb.finish()
    
    print(f"\nTraining completed! Results saved in {save_dir}")

if __name__ == "__main__":
    main()