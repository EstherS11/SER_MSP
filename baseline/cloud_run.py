#!/usr/bin/env python3
# completely_fixed_cloud_run.py - 完全修复版云端运行脚本

import sys
import os
import json
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import logging

# ============================================================================
# 云端集群配置
# ============================================================================

DATA_ROOT = "/data/user_data/esthers/SER_MSP"
BASELINE_DIR = "/data/user_data/esthers/SER_MSP/baseline"
AUDIO_DIR = "/data/user_data/esthers/SER_MSP/DATA/Audios"

JSON_FILES = {
    'train': f"{DATA_ROOT}/msp_train_10class.json",
    'valid': f"{DATA_ROOT}/msp_valid_10class.json", 
    'test': f"{DATA_ROOT}/msp_test_10class.json"
}

# ============================================================================
# 修复1：解决音频文件路径问题
# ============================================================================

def fix_audio_paths(json_file, audio_dir):
    """修复JSON文件中的音频路径"""
    print(f"🔧 Fixing audio paths in {json_file}...")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    audio_path = Path(audio_dir)
    fixed_count = 0
    missing_count = 0
    
    for utt_id, info in data.items():
        original_path = info['wav']
        
        # 如果原路径不存在，尝试在AUDIO_DIR中查找
        if not os.path.exists(original_path):
            # 从原路径提取文件名
            filename = os.path.basename(original_path)
            new_path = audio_path / filename
            
            if new_path.exists():
                info['wav'] = str(new_path)
                fixed_count += 1
            else:
                missing_count += 1
        else:
            fixed_count += 1
    
    print(f"✅ Fixed {fixed_count} paths, {missing_count} still missing")
    return data, fixed_count, missing_count

def prepare_fixed_data(output_dir="data"):
    """准备修复后的数据"""
    print("🔧 Preparing data with fixed paths...")
    
    emotion_map = {
        'N': 0, 'H': 1, 'S': 2, 'A': 3, 'F': 4,
        'D': 5, 'U': 6, 'C': 7, 'O': 8, 'X': 9
    }
    
    emotion_names = [
        "neutral", "happy", "sad", "angry", "fear",
        "disgust", "surprise", "contempt", "other", "unknown"
    ]
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    total_stats = {}
    
    for split, json_file in JSON_FILES.items():
        print(f"\n🔧 Processing {split} set...")
        
        if not Path(json_file).exists():
            print(f"❌ {json_file} not found")
            continue
        
        # 修复音频路径
        data, fixed_count, missing_count = fix_audio_paths(json_file, AUDIO_DIR)
        
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        valid_samples = []
        emotion_counts = {}
        
        # 收集有效样本
        for utt_id, info in data.items():
            wav_path = info['wav']
            emotion = info['emo']
            
            # 检查文件存在且标签有效
            if os.path.exists(wav_path) and emotion in emotion_map:
                valid_samples.append((utt_id, wav_path, emotion_map[emotion]))
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"✅ {split}: {len(valid_samples)} valid samples from {len(data)} total")
        
        # 写入文件
        with open(split_dir / "speech.scp", 'w') as scp_f, \
             open(split_dir / "emotion.txt", 'w') as emo_f, \
             open(split_dir / "utt2spk", 'w') as spk_f:
            
            for utt_id, wav_path, emotion_idx in valid_samples:
                scp_f.write(f"{utt_id} {wav_path}\n")
                emo_f.write(f"{utt_id} {emotion_idx}\n")
                
                # 简单的speaker ID
                speaker_id = '_'.join(utt_id.split('_')[:2])
                spk_f.write(f"{utt_id} {speaker_id}\n")
        
        total_stats[split] = {
            'total_samples': len(data),
            'valid_samples': len(valid_samples),
            'missing_files': len(data) - len(valid_samples),
            'emotion_distribution': emotion_counts
        }
    
    # 保存统计信息
    with open(output_path / "stats.json", 'w') as f:
        json.dump(total_stats, f, indent=2)
    
    print(f"\n✅ Data prepared in {output_dir}/")
    print("📊 Statistics:")
    for split, stats in total_stats.items():
        print(f"  {split}: {stats['valid_samples']}/{stats['total_samples']} valid samples")
    
    return total_stats

# ============================================================================
# 修复2：简化的ESP-net配置
# ============================================================================

def create_fixed_espnet_config():
    """创建修复后的ESP-net配置"""
    
    config = """# Fixed ESP-net ASR configuration
batch_type: numel
batch_size: 16
max_epoch: 30
patience: 5
seed: 42
num_workers: 4
log_interval: 100
grad_clip: 5.0
accum_grad: 1

# Optimizer
optim: adamw
optim_conf:
  lr: 0.0001
  weight_decay: 0.001

# Scheduler
scheduler: warmuplr
scheduler_conf:
  warmup_steps: 1000

# Model configuration
model: espnet
model_conf:
  ctc_weight: 0.3
  lsm_weight: 0.1
  length_normalized_loss: false
  
# Encoder
encoder: conformer
encoder_conf:
  output_size: 256
  attention_heads: 4
  linear_units: 1024
  num_blocks: 6
  dropout_rate: 0.1
  attention_dropout_rate: 0.0
  input_layer: conv2d
  normalize_before: true
  
# Decoder
decoder: transformer
decoder_conf:
  attention_heads: 4
  linear_units: 1024
  num_blocks: 3
  dropout_rate: 0.1

# Frontend
frontend: default
frontend_conf:
  n_fft: 512
  win_length: 400
  hop_length: 160

# SpecAugment
specaug: specaug
specaug_conf:
  apply_time_warp: true
  time_warp_window: 5
  apply_freq_mask: true
  freq_mask_width_range: [0, 30]
  num_freq_mask: 2
  apply_time_mask: true
  time_mask_width_range: [0, 40]
  num_time_mask: 2

# Best model criterion
best_model_criterion:
  - ["valid", "acc", "max"]
  - ["valid", "loss", "min"]

# Other settings
resume: true
keep_nbest_models: 3
use_tensorboard: true
"""
    
    with open("fixed_espnet_config.yaml", 'w') as f:
        f.write(config)
    
    print("✅ Created fixed_espnet_config.yaml")

# ============================================================================
# 修复3：完整的PyTorch SER实现
# ============================================================================

class SimpleSERDataset(Dataset):
    """修复后的SER数据集"""
    def __init__(self, data_dir, split='train', max_length=16000*5):
        self.max_length = max_length
        self.data = []
        
        # 读取数据
        speech_file = Path(data_dir) / split / "speech.scp"
        emotion_file = Path(data_dir) / split / "emotion.txt"
        
        if not speech_file.exists() or not emotion_file.exists():
            raise FileNotFoundError(f"Data files not found in {data_dir}/{split}")
        
        speech_data = {}
        with open(speech_file, 'r') as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    utt_id, path = parts
                    speech_data[utt_id] = path
        
        with open(emotion_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    utt_id, emotion = parts
                    if utt_id in speech_data:
                        self.data.append((speech_data[utt_id], int(emotion)))
        
        print(f"📊 {split} dataset: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        wav_path, emotion = self.data[idx]
        
        try:
            # 加载音频
            waveform, sr = torchaudio.load(wav_path)
            
            # 重采样到16kHz
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            # 转为单声道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # 处理长度
            if waveform.shape[1] > self.max_length:
                waveform = waveform[:, :self.max_length]
            else:
                pad_length = self.max_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, pad_length))
                
            return waveform.squeeze(0), emotion
            
        except Exception as e:
            print(f"⚠️  Error loading {wav_path}: {e}")
            return torch.zeros(self.max_length), 0

class SimpleSERModel(nn.Module):
    """改进的SER模型"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        # CNN特征提取器
        self.features = nn.Sequential(
            # 第一层：粗粒度特征
            nn.Conv1d(1, 64, kernel_size=80, stride=16),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            
            # 第二层：中等粒度特征
            nn.Conv1d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            
            # 第三层：细粒度特征
            nn.Conv1d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            
            # 全局平均池化
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, time)
        x = x.unsqueeze(1)  # (batch, 1, time)
        x = self.features(x)  # (batch, 256, 1)
        x = x.squeeze(-1)  # (batch, 256)
        x = self.classifier(x)  # (batch, num_classes)
        return x

def train_pytorch_ser():
    """训练PyTorch SER模型"""
    print("🚀 Training PyTorch SER Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # 检查数据是否准备好
    if not Path("data/train/speech.scp").exists():
        print("❌ Data not prepared. Running data preparation first...")
        stats = prepare_fixed_data()
        
        # 检查是否有有效数据
        if all(stats[split]['valid_samples'] == 0 for split in ['train', 'valid']):
            print("❌ No valid samples found after data preparation!")
            print("🔍 Please check:")
            print(f"  1. Audio files are in: {AUDIO_DIR}")
            print(f"  2. JSON files contain correct paths")
            return False
    
    # 数据加载
    try:
        train_dataset = SimpleSERDataset("data", "train")
        valid_dataset = SimpleSERDataset("data", "valid")
        
        if len(train_dataset) == 0 or len(valid_dataset) == 0:
            print("❌ Empty datasets!")
            return False
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=8, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True if device.type == 'cuda' else False
        )
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=8, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        print(f"📊 Dataset sizes - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # 模型设置
    model = SimpleSERModel(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_macro_f1 = 0.0
    patience_counter = 0
    max_patience = 5
    
    # 训练循环
    for epoch in range(20):
        print(f"\n📈 Epoch {epoch+1}/20")
        
        # === 训练阶段 ===
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (audio, labels) in enumerate(train_pbar):
            audio, labels = audio.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # === 验证阶段 ===
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(valid_loader, desc="Validation", leave=False)
            for audio, labels in val_pbar:
                audio, labels = audio.to(device), labels.to(device)
                outputs = model(audio)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # 计算指标
        val_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        avg_val_loss = val_loss / len(valid_loader)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # 学习率调度
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 打印结果
        print(f"🏋️  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"📊 Valid Loss: {avg_val_loss:.4f}, Valid Acc: {val_acc:.2f}%")
        print(f"🎯 Macro-F1: {macro_f1:.4f}, LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            patience_counter = 0
            
            # 保存模型
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_macro_f1': best_macro_f1,
                'val_acc': val_acc,
                'val_loss': avg_val_loss
            }
            torch.save(checkpoint, "best_ser_model.pth")
            print(f"💾 New best model saved! Macro-F1: {best_macro_f1:.4f}")
            
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{max_patience}")
            
            if patience_counter >= max_patience:
                print("🛑 Early stopping triggered!")
                break
    
    # 最终评估
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best Macro-F1: {best_macro_f1:.4f}")
    
    # 生成详细报告
    if len(set(all_labels)) > 1:
        emotion_names = [
            "Neutral", "Happy", "Sad", "Angry", "Fear",
            "Disgust", "Surprise", "Contempt", "Other", "Unknown"
        ]
        try:
            report = classification_report(
                all_labels, 
                all_preds,
                target_names=emotion_names[:len(set(all_labels))],
                zero_division=0,
                digits=4
            )
            print(f"\n📋 Final Classification Report:\n{report}")
        except:
            print("⚠️  Could not generate detailed classification report")
    
    return True

# ============================================================================
# 主程序
# ============================================================================

def check_environment():
    """检查环境"""
    print("🔍 Checking environment...")
    
    # 检查数据路径
    checks = [
        (DATA_ROOT, "Data root"),
        (AUDIO_DIR, "Audio directory"),
    ]
    
    for path, name in checks:
        if Path(path).exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name} not found: {path}")
            return False
    
    # 检查JSON文件
    for split, json_path in JSON_FILES.items():
        if Path(json_path).exists():
            print(f"✅ {split} JSON: {json_path}")
        else:
            print(f"❌ {split} JSON not found: {json_path}")
            return False
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"✅ CUDA: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  No CUDA, using CPU")
    
    return True

def main():
    print("🌐 Completely Fixed Cloud SER Training")
    print("=" * 60)
    print(f"📁 Data root: {DATA_ROOT}")
    print(f"🎵 Audio dir: {AUDIO_DIR}")
    print("=" * 60)
    
    # 1. 环境检查
    if not check_environment():
        print("❌ Environment check failed")
        return
    
    # 2. 数据准备
    print("\n🔧 Step 1: Data Preparation")
    try:
        stats = prepare_fixed_data()
        
        # 检查是否有足够的数据
        train_samples = stats.get('train', {}).get('valid_samples', 0)
        valid_samples = stats.get('valid', {}).get('valid_samples', 0)
        
        if train_samples == 0 or valid_samples == 0:
            print("❌ Insufficient valid data found!")
            print("🔍 Troubleshooting suggestions:")
            print(f"  1. Check if audio files exist in: {AUDIO_DIR}")
            print(f"  2. Verify JSON file paths are correct")
            print(f"  3. Ensure audio file extensions match (.wav)")
            return
        
        print(f"✅ Found {train_samples} training samples, {valid_samples} validation samples")
        
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return
    
    # 3. 创建配置文件
    print("\n🔧 Step 2: Configuration")
    create_fixed_espnet_config()
    
    # 4. 训练模型
    print("\n🚀 Step 3: Model Training")
    
    try:
        if train_pytorch_ser():
            print("\n🎉 Training completed successfully!")
            print("\n📊 Results:")
            print("  - Model: best_ser_model.pth")
            print("  - Logs: Training progress shown above")
            print("\n🎯 Next steps:")
            print("  - Test the model on test set")
            print("  - Analyze per-class performance")
            print("  - Consider hyperparameter tuning")
        else:
            print("\n❌ Training failed")
            
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()