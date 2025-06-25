import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import numpy as np
from sklearn.metrics import f1_score, classification_report
from transformers import WavLMModel
import warnings
import time
import sys
from datetime import datetime
from tqdm import tqdm
import psutil
import logging
warnings.filterwarnings('ignore')

# ============= 设置详细日志 =============
def setup_logging(log_file):
    """设置详细的日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_system_info():
    """获取系统资源信息"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_info = f"GPU Memory: {gpu_memory_used:.1f}/{gpu_memory_total:.1f}GB"
    
    return f"CPU: {cpu_percent}% | RAM: {memory.percent}% | {gpu_info}"

# ============= Focal Loss Implementation =============
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            focal_loss = at * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ECAPA-TDNN Components
class SE_Res2Block(nn.Module):
    """Squeeze-Excitation Res2Block used in ECAPA-TDNN"""
    def __init__(self, in_channels, out_channels, res2_scale=8, se_channels=128, kernel_size=3):
        super(SE_Res2Block, self).__init__()
        
        self.res2_scale = res2_scale
        self.out_channels = out_channels
        
        # Res2 convolutions
        self.conv_blocks = nn.ModuleList()
        for i in range(res2_scale):
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels // res2_scale, out_channels // res2_scale, 
                             kernel_size=kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(out_channels // res2_scale),
                    nn.ReLU(inplace=True)
                )
            )
        
        # SE block
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, se_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(se_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
            
    def forward(self, x):
        residual = self.residual(x)
        
        # Split input for Res2
        xs = torch.chunk(x, self.res2_scale, dim=1)
        
        # Progressive convolutions
        outs = []
        for i, conv_block in enumerate(self.conv_blocks):
            if i == 0:
                y = conv_block(xs[i])
            else:
                y = conv_block(xs[i] + outs[-1])
            outs.append(y)
        
        # Concatenate outputs
        out = torch.cat(outs, dim=1)
        
        # SE attention
        se_weight = self.se_block(out)
        out = out * se_weight
        
        # Add residual
        out = out + residual
        
        return out

class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling layer"""
    def __init__(self, channels, attention_channels=128):
        super(AttentiveStatisticsPooling, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, channels, kernel_size=1),
            nn.Softmax(dim=2),
        )
        
    def forward(self, x):
        # x: [B, C, T]
        # Calculate attention weights
        w = self.attention(x)
        
        # Weighted statistics
        mu = torch.sum(x * w, dim=2)
        rh = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        
        # Concatenate mean and std
        pooled = torch.cat([mu, rh], dim=1)
        
        return pooled

# ============= Hierarchical Model Architecture =============
class HierarchicalWavLMECAPAClassifier(nn.Module):
    """
    Serial/Hierarchical Architecture: WavLM -> ECAPA-TDNN -> Classifier
    WavLM extracts frame-level features, ECAPA performs intelligent pooling
    """
    def __init__(self, num_classes=10, wavlm_name="microsoft/wavlm-large", freeze_wavlm=True):
        super(HierarchicalWavLMECAPAClassifier, self).__init__()
        
        # WavLM as feature extractor
        self.wavlm = WavLMModel.from_pretrained(wavlm_name)
        if freeze_wavlm:
            for param in self.wavlm.parameters():
                param.requires_grad = False
        
        # Get WavLM output dimension
        wavlm_dim = 1024  # WavLM-large output dimension
        ecapa_dim = 512   # The internal dimension for ECAPA blocks
        
        # 输入投影层，将WavLM的1024维映射到ECAPA的512维
        self.input_projection = nn.Sequential(
            nn.Conv1d(wavlm_dim, ecapa_dim, kernel_size=1),
            nn.BatchNorm1d(ecapa_dim),
            nn.ReLU()
        )
        
        # ECAPA-TDNN components
        self.ecapa_encoder = nn.ModuleDict({
            'layer1': SE_Res2Block(ecapa_dim, ecapa_dim, res2_scale=4, se_channels=128, kernel_size=3),
            'layer2': SE_Res2Block(ecapa_dim, ecapa_dim, res2_scale=4, se_channels=128, kernel_size=3),
            'layer3': SE_Res2Block(ecapa_dim, ecapa_dim, res2_scale=4, se_channels=128, kernel_size=3),
            'layer4': nn.Conv1d(ecapa_dim, 1536, kernel_size=1),
        })
        
        # Attentive Statistics Pooling
        self.asp = AttentiveStatisticsPooling(1536, attention_channels=128)
        
        # Final embedding layer
        self.fc_emb = nn.Linear(3072, 192)  # 1536*2 from ASP
        self.bn_emb = nn.BatchNorm1d(192)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, waveform):
        # 1. WavLM feature extraction
        wavlm_features = self.wavlm(waveform).last_hidden_state
        
        # 2. Transpose for CNN processing: [Batch, 1024, Time]
        features = wavlm_features.transpose(1, 2)
        
        # 3. 应用输入投影层
        projected_features = self.input_projection(features)
        
        # 4. ECAPA encoding
        x = self.ecapa_encoder['layer1'](projected_features)
        x = self.ecapa_encoder['layer2'](x)
        x = self.ecapa_encoder['layer3'](x)
        x = self.ecapa_encoder['layer4'](x)
        
        # 5. Attentive Statistics Pooling
        pooled = self.asp(x)
        
        # 6. Final embedding
        emb = self.fc_emb(pooled)
        emb = self.bn_emb(emb)
        
        # 7. Classification
        logits = self.classifier(emb)
        
        return logits

# ============= Enhanced Data Augmentation =============
class AudioAugmentationPipeline:
    """Professional audio augmentation pipeline"""
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.augmentations = {
            'noise': self.add_gaussian_noise,
            'pitch': self.pitch_shift,
            'speed': self.time_stretch,
            'reverb': self.add_reverb,
            'volume': self.random_volume,
        }
        
    def add_gaussian_noise(self, waveform, min_snr_db=5, max_snr_db=40):
        """Add Gaussian noise with random SNR"""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        snr_db = np.random.uniform(min_snr_db, max_snr_db)
        signal_power = waveform.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        
        return (waveform + noise).squeeze(0)
    
    def pitch_shift(self, waveform, min_semitones=-3, max_semitones=3):
        """Pitch shifting"""
        semitones = np.random.uniform(min_semitones, max_semitones)
        return torchaudio.functional.pitch_shift(waveform, self.sample_rate, semitones)
    
    def time_stretch(self, waveform, min_rate=0.8, max_rate=1.25):
        """Time stretching"""
        rate = np.random.uniform(min_rate, max_rate)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Use phase vocoder for time stretching
        stretched = torchaudio.functional.phase_vocoder(
            torch.stft(waveform, n_fft=1024, hop_length=256, return_complex=True),
            rate=rate,
            phase_advance=torch.linspace(0, np.pi * 256 / 1024, 513)[..., None]
        )
        return torch.istft(stretched, n_fft=1024, hop_length=256).squeeze(0)
    
    def add_reverb(self, waveform, reverb_prob=0.5):
        """Simple reverb effect using convolution"""
        if np.random.random() > reverb_prob:
            return waveform
            
        # Simple reverb using delay and decay
        delay_ms = np.random.uniform(20, 50)
        decay = np.random.uniform(0.1, 0.3)
        
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        reverb = torch.zeros_like(waveform)
        reverb[delay_samples:] = waveform[:-delay_samples] * decay
        
        return waveform + reverb
    
    def random_volume(self, waveform, min_gain=0.5, max_gain=1.5):
        """Random volume adjustment"""
        gain = np.random.uniform(min_gain, max_gain)
        return waveform * gain
    
    def __call__(self, waveform, augmentation_prob=0.8, max_augmentations=3):
        """Apply random augmentations"""
        if np.random.random() > augmentation_prob:
            return waveform
            
        # Randomly select augmentations to apply
        num_augmentations = np.random.randint(1, max_augmentations + 1)
        selected_augs = np.random.choice(list(self.augmentations.keys()), 
                                       size=num_augmentations, replace=False)
        
        # Apply augmentations sequentially
        for aug_name in selected_augs:
            waveform = self.augmentations[aug_name](waveform)
            
        return waveform

# ============= Efficient Dataset with Dynamic Padding =============
class EfficientMSPDataset(Dataset):
    """Dataset with efficient loading and augmentation"""
    def __init__(self, json_path, root_dir, augment=False, sample_rate=16000):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.root_dir = root_dir
        self.keys = list(self.data.keys())
        self.sample_rate = sample_rate
        self.augment = augment
        
        if augment:
            self.augmentation_pipeline = AudioAugmentationPipeline(sample_rate)
            
        self.emotion_map = {
            'N': 0, 'H': 1, 'S': 2, 'A': 3, 'F': 4,
            'D': 5, 'U': 6, 'C': 7, 'O': 8, 'X': 9
        }
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data[key]
        
        # Load audio
        wav_path = os.path.join(self.root_dir, item['wav'])
        waveform, sr = torchaudio.load(wav_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        waveform = waveform.squeeze(0)
        
        # Apply augmentation
        if self.augment:
            waveform = self.augmentation_pipeline(waveform)
        
        # Get label
        label = self.emotion_map[item['emo']]
        
        # Return waveform, label, and original length
        return waveform, label, waveform.shape[0]

def collate_fn_padd(batch):
    """
    Custom collate function for dynamic padding within batch
    """
    # Separate waveforms, labels, and lengths
    waveforms, labels, lengths = zip(*batch)
    
    # Convert to tensors
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    # Pad waveforms to the maximum length in this batch
    waveforms_padded = pad_sequence([w for w in waveforms], 
                                   batch_first=True, 
                                   padding_value=0.0)
    
    return waveforms_padded, labels, lengths

# ============= Training Script with Enhanced Monitoring =============
def train_improved_model():
    # 设置详细日志
    log_file = f"logs/detailed_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file)
    
    logger.info("🚀 Starting training with enhanced monitoring...")
    logger.info(f"🕐 Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"🔧 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    batch_size = 16  # 减小batch size以获得更频繁的更新
    learning_rate = 1e-4
    num_epochs = 50
    
    # Data paths
    if os.path.exists('/data/user_data/esthers/SER_MSP'):
        root_dir = '/data/user_data/esthers/SER_MSP'
    else:
        root_dir = '/Users/esthersun/Desktop/SER/SER_MSP'
    
    logger.info(f"📁 Data directory: {root_dir}")
    
    # Create datasets with efficient loading
    logger.info("🔄 Loading training dataset...")
    train_dataset = EfficientMSPDataset(
        os.path.join(root_dir, 'msp_train_10class.json'),
        root_dir,
        augment=True
    )
    
    logger.info("🔄 Loading validation dataset...")
    valid_dataset = EfficientMSPDataset(
        os.path.join(root_dir, 'msp_valid_10class.json'),
        root_dir,
        augment=False
    )
    
    logger.info(f"📊 Training samples: {len(train_dataset):,}")
    logger.info(f"📊 Validation samples: {len(valid_dataset):,}")
    logger.info(f"📊 Batches per epoch: {len(train_dataset) // batch_size:,}")
    
    # Create dataloaders with custom collate function
    logger.info("🔄 Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_padd,
        num_workers=4,  # 可以根据需要调整
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_padd,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info("✅ Data loaders created successfully")
    
    # Initialize model
    logger.info("🔄 Initializing HierarchicalWavLMECAPAClassifier...")
    model_start_time = time.time()
    model = HierarchicalWavLMECAPAClassifier(num_classes=10).to(device)
    model_init_time = time.time() - model_start_time
    logger.info(f"✅ Model initialized in {model_init_time:.2f} seconds")
    
    # Print model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Total parameters: {total_params:,}")
    logger.info(f"📊 Trainable parameters: {trainable_params:,}")
    logger.info(f"📊 Frozen parameters: {total_params - trainable_params:,}")
    
    # Calculate class weights for focal loss
    logger.info("🔄 Calculating class weights...")
    emotion_counts = {}
    for item in train_dataset.data.values():
        emo = train_dataset.emotion_map[item['emo']]
        emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
    
    total_samples = sum(emotion_counts.values())
    class_weights = torch.tensor([
        total_samples / (10 * emotion_counts.get(i, 1)) 
        for i in range(10)
    ]).to(device)
    
    logger.info("📊 Class distribution:")
    emotion_names = ['N', 'H', 'S', 'A', 'F', 'D', 'U', 'C', 'O', 'X']
    for i, (name, count) in enumerate(zip(emotion_names, [emotion_counts.get(i, 0) for i in range(10)])):
        logger.info(f"  {name}: {count:,} samples (weight: {class_weights[i]:.3f})")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    logger.info(f"🎯 Starting training for {num_epochs} epochs...")
    logger.info("=" * 80)
    
    # Training loop
    best_valid_f1 = 0
    total_training_time = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        logger.info(f"🔄 Epoch {epoch+1}/{num_epochs} started")
        logger.info(f"📊 {get_system_info()}")
        
        # Train
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        # 使用tqdm显示训练进度
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", 
                         leave=False, file=sys.stdout)
        
        batch_times = []
        for batch_idx, (waveforms, labels, lengths) in enumerate(train_pbar):
            batch_start_time = time.time()
            
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # 更新进度条
            avg_batch_time = np.mean(batch_times[-50:])  # 最近50个batch的平均时间
            eta_seconds = avg_batch_time * (len(train_loader) - batch_idx - 1)
            eta_minutes = eta_seconds / 60
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Batch_time': f'{batch_time:.2f}s',
                'ETA': f'{eta_minutes:.1f}min'
            })
            
            # 每50个batch详细记录一次
            if (batch_idx + 1) % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(f"  📍 Batch {batch_idx+1}/{len(train_loader)} | "
                          f"Loss: {loss.item():.4f} | "
                          f"LR: {current_lr:.2e} | "
                          f"Batch time: {batch_time:.2f}s | "
                          f"ETA: {eta_minutes:.1f}min | "
                          f"{get_system_info()}")
        
        train_pbar.close()
        
        # Validation
        logger.info("🔄 Starting validation...")
        model.eval()
        valid_loss = 0
        valid_preds = []
        valid_labels = []
        
        valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1} Validation", 
                         leave=False, file=sys.stdout)
        
        with torch.no_grad():
            for batch_idx, (waveforms, labels, lengths) in enumerate(valid_pbar):
                waveforms = waveforms.to(device)
                labels = labels.to(device)
                
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                valid_preds.extend(preds.cpu().numpy())
                valid_labels.extend(labels.cpu().numpy())
                
                valid_pbar.set_postfix({'Val_Loss': f'{loss.item():.4f}'})
        
        valid_pbar.close()
        
        # Calculate metrics
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        valid_f1 = f1_score(valid_labels, valid_preds, average='macro')
        
        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time
        
        # 详细的epoch总结
        logger.info(f"\n🎯 Epoch {epoch+1}/{num_epochs} Results:")
        logger.info(f"  ⏱️ Epoch time: {epoch_time/60:.2f} minutes")
        logger.info(f"  📊 Train Loss: {train_loss/len(train_loader):.4f} | Train F1: {train_f1:.4f}")
        logger.info(f"  📊 Valid Loss: {valid_loss/len(valid_loader):.4f} | Valid F1: {valid_f1:.4f}")
        logger.info(f"  📊 Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        logger.info(f"  📊 Total training time: {total_training_time/3600:.2f} hours")
        logger.info(f"  📊 {get_system_info()}")
        
        # Save best model
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_valid_f1': best_valid_f1,
                'class_weights': class_weights,
                'train_f1': train_f1,
                'valid_f1': valid_f1,
                'total_training_time': total_training_time,
            }
            torch.save(checkpoint, 'hierarchical_best_model.pth')
            logger.info(f"  ✅ Saved best model with Valid F1: {best_valid_f1:.4f}")
        
        logger.info("-" * 80)
    
    logger.info("🎉 Training completed!")
    logger.info(f"🏆 Best validation F1 score: {best_valid_f1:.4f}")
    logger.info(f"⏱️ Total training time: {total_training_time/3600:.2f} hours")
    
    # Final evaluation with detailed classification report
    logger.info("🔄 Final evaluation...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for waveforms, labels, lengths in tqdm(valid_loader, desc="Final Evaluation"):
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            outputs = model(waveforms)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    logger.info("\n📊 Final Classification Report:")
    report = classification_report(all_labels, all_preds, 
                                 target_names=emotion_names, 
                                 digits=4)
    logger.info(f"\n{report}")

if __name__ == "__main__":
    train_improved_model()