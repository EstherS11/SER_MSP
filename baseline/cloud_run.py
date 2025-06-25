#!/usr/bin/env python3
# fixed_cloud_run.py - 修复版云端集群运行脚本

import sys
import os
import json
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import Counter
import logging
from tqdm import tqdm

# ============================================================================
# 云端集群配置 - 根据你的实际路径
# ============================================================================

# 🔧 你的实际数据路径
DATA_ROOT = "/data/user_data/esthers/SER_MSP"
BASELINE_DIR = "/data/user_data/esthers/SER_MSP/baseline"
AUDIO_DIR = "/data/user_data/esthers/SER_MSP/DATA/Audios"

# JSON文件路径
JSON_FILES = {
    'train': f"{DATA_ROOT}/msp_train_10class.json",
    'valid': f"{DATA_ROOT}/msp_valid_10class.json", 
    'test': f"{DATA_ROOT}/msp_test_10class.json"
}

# ============================================================================
# 修复：直接使用ASR任务进行SER（ESP-net推荐方式）
# ============================================================================

def create_fixed_data_prep():
    """创建修复版数据准备脚本 - 使用ASR多任务格式"""
    
    data_prep_code = '''#!/usr/bin/env python3
# fixed_cloud_data_prep.py - 修复版云端数据准备脚本

import json
import os
import numpy as np
from pathlib import Path
from collections import Counter

def prepare_asr_multitask_data(output_dir="data"):
    """
    准备ASR多任务格式数据 - ESP-net推荐的SER实现方式
    同时预测文本转录和情感标签
    """
    
    # 云端路径配置
    JSON_FILES = {
        'train': "/data/user_data/esthers/SER_MSP/msp_train_10class.json",
        'valid': "/data/user_data/esthers/SER_MSP/msp_valid_10class.json", 
        'test': "/data/user_data/esthers/SER_MSP/msp_test_10class.json"
    }
    
    # MSP-PODCAST情感标签映射
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
    
    print("🔧 Preparing ASR multi-task format data for SER...")
    
    stats = {}
    
    # 处理每个数据集
    for split, json_file in JSON_FILES.items():
        print(f"\\nProcessing {split} set...")
        
        if not Path(json_file).exists():
            print(f"❌ {json_file} not found")
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        valid_count = 0
        durations = []
        emotion_counts = {}
        missing_files = []
        
        # 创建ASR多任务格式文件
        with open(split_dir / "speech.scp", 'w') as scp_f, \\
             open(split_dir / "text", 'w') as text_f, \\
             open(split_dir / "emotion.txt", 'w') as emo_f, \\
             open(split_dir / "utt2spk", 'w') as spk_f:
            
            for utt_id, info in data.items():
                wav_path = info['wav']
                emotion = info['emo']
                # 从转录文本获取，如果没有就用占位符
                text = info.get('transcript', '<unk>')
                duration = info.get('length', 0)
                
                # 检查文件存在且标签有效
                if not os.path.exists(wav_path):
                    missing_files.append(wav_path)
                    continue
                    
                if emotion not in emotion_map:
                    print(f"⚠️  Unknown emotion '{emotion}' for {utt_id}")
                    continue
                
                # 写入ASR多任务格式文件
                scp_f.write(f"{utt_id} {wav_path}\\n")
                text_f.write(f"{utt_id} {text}\\n")  # ASR目标
                emo_f.write(f"{utt_id} {emotion_map[emotion]}\\n")  # SER目标
                
                # 简单的speaker ID
                speaker_id = '_'.join(utt_id.split('_')[:2])
                spk_f.write(f"{utt_id} {speaker_id}\\n")
                
                valid_count += 1
                durations.append(duration)
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # 统计信息
        stats[split] = {
            'count': valid_count,
            'missing_files': len(missing_files),
            'duration_mean': np.mean(durations) if durations else 0,
            'duration_std': np.std(durations) if durations else 0,
            'emotion_dist': emotion_counts
        }
        
        print(f"✅ {split}: {valid_count} valid samples")
        if missing_files:
            print(f"⚠️  {split}: {len(missing_files)} missing files")
    
    # 保存数据集信息
    dataset_info = {
        "dataset_name": "MSP-PODCAST-ASR-MultiTask",
        "task": "ASR + 10-class emotion recognition",
        "num_classes": 10,
        "emotion_names": emotion_names,
        "emotion_mapping": emotion_map,
        "stats": stats,
        "format": "ASR multi-task (text + emotion)",
        "data_source": "Cloud cluster: /data/user_data/esthers/SER_MSP"
    }
    
    with open(output_path / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"\\n✅ ASR multi-task data prepared in {output_dir}/")
    print("📝 Format: speech.scp + text + emotion.txt (ASR + SER)")
    
    return dataset_info

if __name__ == "__main__":
    prepare_asr_multitask_data()
'''
    
    with open("fixed_cloud_data_prep.py", 'w') as f:
        f.write(data_prep_code)
    
    print("✅ Created fixed_cloud_data_prep.py")

def create_asr_multitask_config():
    """创建ASR多任务配置文件"""
    
    config = '''# asr_multitask_config.yaml - ASR多任务配置（文本+情感）
batch_type: numel
batch_size: 16
max_epoch: 50
patience: 10
seed: 42
num_workers: 4
log_interval_steps: 100
grad_clip: 5.0
accum_grad: 1

# 优化器配置
optim: adamw
optim_conf:
  lr: 0.0001
  weight_decay: 0.001

# 学习率调度器
scheduler: warmuplr
scheduler_conf:
  warmup_steps: 1000

# 模型配置 - 使用Conformer进行多任务学习
model: espnet
model_conf:
  # 编码器
  encoder: conformer
  encoder_conf:
    output_size: 256
    attention_heads: 4
    linear_units: 1024
    num_blocks: 8
    dropout_rate: 0.1
    attention_dropout_rate: 0.0
    input_layer: conv2d
    normalize_before: true
    
  # ASR解码器
  decoder: transformer
  decoder_conf:
    attention_heads: 4
    linear_units: 1024
    num_blocks: 4
    dropout_rate: 0.1
    
  # 多任务配置 - 关键部分
  ctc_weight: 0.3
  lsm_weight: 0.1
  
  # 添加情感分类头
  aux_ctc_tasks:
    - name: emotion
      ctc_weight: 0.3
      ctc_conf:
        dropout_rate: 0.0
        
# 预处理器配置
frontend: default
frontend_conf:
  # 特征提取
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

# 最佳模型选择
best_model_criterion:
  - ["valid", "acc", "max"]
  - ["valid", "loss", "min"]

# 其他配置
resume: true
keep_nbest_models: 3
use_tensorboard: true
'''
    
    with open("asr_multitask_config.yaml", 'w') as f:
        f.write(config)
    
    print("✅ Created asr_multitask_config.yaml")

def create_simple_pytorch_ser():
    """创建简单的PyTorch SER基线（备选方案）"""
    
    ser_code = '''#!/usr/bin/env python3
# simple_pytorch_ser.py - 简单PyTorch SER基线

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
from pathlib import Path
import logging
from tqdm import tqdm

class SimpleSERDataset(Dataset):
    """简单SER数据集"""
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
                utt_id, path = line.strip().split(None, 1)
                speech_data[utt_id] = path
        
        with open(emotion_file, 'r') as f:
            for line in f:
                utt_id, emotion = line.strip().split()
                if utt_id in speech_data:
                    self.data.append((speech_data[utt_id], int(emotion)))
    
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
            print(f"Error loading {wav_path}: {e}")
            return torch.zeros(self.max_length), 0

class SimpleSERModel(nn.Module):
    """简单的SER模型"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 简单的CNN特征提取器
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=80, stride=16),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, time)
        x = self.features(x)  # (batch, 256, 1)
        x = x.squeeze(-1)  # (batch, 256)
        x = self.classifier(x)  # (batch, num_classes)
        return x

def train_simple_ser():
    """训练简单SER模型"""
    print("🚀 Training Simple PyTorch SER Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据加载
    try:
        train_dataset = SimpleSERDataset("data", "train")
        valid_dataset = SimpleSERDataset("data", "valid")
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
        valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=2)
        
        print(f"Dataset sizes - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # 模型
    model = SimpleSERModel(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_macro_f1 = 0.0
    
    # 训练循环
    for epoch in range(10):  # 简化为10个epoch
        print(f"\\nEpoch {epoch+1}/10")
        
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (audio, labels) in enumerate(tqdm(train_loader, desc="Training")):
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
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for audio, labels in tqdm(valid_loader, desc="Validation"):
                audio, labels = audio.to(device), labels.to(device)
                outputs = model(audio)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        val_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        avg_val_loss = val_loss / len(valid_loader)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Macro-F1: {macro_f1:.4f}")
        
        # 保存最佳模型
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), "best_simple_ser_model.pth")
            print(f"🎯 New best Macro-F1: {best_macro_f1:.4f}")
    
    print(f"\\n✅ Training completed! Best Macro-F1: {best_macro_f1:.4f}")
    return True

if __name__ == "__main__":
    train_simple_ser()
'''
    
    with open("simple_pytorch_ser.py", 'w') as f:
        f.write(ser_code)
    
    print("✅ Created simple_pytorch_ser.py")

def check_cloud_environment():
    """检查云端环境"""
    print("🔍 Checking cloud environment...")
    
    # 检查数据路径
    if not Path(DATA_ROOT).exists():
        print(f"❌ Data root not found: {DATA_ROOT}")
        return False
    
    if not Path(AUDIO_DIR).exists():
        print(f"❌ Audio directory not found: {AUDIO_DIR}")
        return False
    
    # 检查JSON文件
    for split, json_path in JSON_FILES.items():
        if not Path(json_path).exists():
            print(f"❌ Missing JSON file: {json_path}")
            return False
        print(f"✅ Found {split}: {json_path}")
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  No CUDA, using CPU")
    
    return True

def install_dependencies():
    """安装必需的依赖"""
    print("📦 Installing dependencies...")
    
    packages = [
        "torch",
        "torchaudio", 
        "transformers",
        "scikit-learn",
        "numpy",
        "tqdm"
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"🔧 Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)

def run_asr_multitask_training():
    """运行ASR多任务训练（如果ESP-net可用）"""
    print("🚀 Attempting ASR multi-task training...")
    
    # 数据准备
    try:
        from fixed_cloud_data_prep import prepare_asr_multitask_data
        prepare_asr_multitask_data()
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return False
    
    # ASR训练命令
    exp_dir = Path("exp/asr_multitask_ser")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    train_cmd = [
        sys.executable, "-m", "espnet2.bin.asr_train",
        "--train_data_path_and_name_and_type", "data/train/speech.scp,speech,sound",
        "--train_data_path_and_name_and_type", "data/train/text,text,text",
        "--train_data_path_and_name_and_type", "data/train/emotion.txt,emotion,text_int",
        "--valid_data_path_and_name_and_type", "data/valid/speech.scp,speech,sound",
        "--valid_data_path_and_name_and_type", "data/valid/text,text,text", 
        "--valid_data_path_and_name_and_type", "data/valid/emotion.txt,emotion,text_int",
        "--output_dir", str(exp_dir),
        "--config", "asr_multitask_config.yaml",
        "--ngpu", "1" if torch.cuda.is_available() else "0",
        "--num_workers", "4",
    ]
    
    print("ASR Multi-task training command:")
    print(" ".join(train_cmd))
    
    try:
        subprocess.run(train_cmd, check=True)
        print("✅ ASR multi-task training completed!")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ ASR training failed: {e}")
        print("🔄 Falling back to simple PyTorch SER...")
        return False

def run_simple_pytorch_training():
    """运行简单PyTorch训练（备选方案）"""
    print("🔄 Running simple PyTorch SER training...")
    
    try:
        # 导入并运行
        exec(open("simple_pytorch_ser.py").read())
        return True
    except Exception as e:
        print(f"❌ Simple PyTorch training failed: {e}")
        return False

def main():
    print("🌐 Fixed Cloud Cluster SER Training")
    print("=" * 60)
    print(f"📁 Data root: {DATA_ROOT}")
    print(f"🎵 Audio dir: {AUDIO_DIR}")
    print("=" * 60)
    
    # 1. 环境检查
    if not check_cloud_environment():
        print("❌ Environment check failed")
        return
    
    # 2. 安装依赖
    try:
        install_dependencies()
    except Exception as e:
        print(f"⚠️  Dependency installation issue: {e}")
    
    # 3. 创建必需文件
    print("\\n🔧 Creating configuration files...")
    create_fixed_data_prep()
    create_asr_multitask_config()
    create_simple_pytorch_ser()
    
    # 4. 尝试训练
    print("\\n🚀 Starting training...")
    
    # 方案1: ASR多任务（ESP-net推荐）
    if run_asr_multitask_training():
        print("\\n🎉 ASR multi-task training successful!")
        print("📊 This approach predicts both text and emotion simultaneously")
    else:
        # 方案2: 简单PyTorch SER（备选）
        print("\\n🔄 Trying simple PyTorch approach...")
        if run_simple_pytorch_training():
            print("\\n🎉 Simple PyTorch SER training successful!")
        else:
            print("\\n❌ All training approaches failed")
            return
    
    print("\\n" + "=" * 60)
    print("🎯 Training completed!")
    print("📁 Check results in:")
    print("  - exp/asr_multitask_ser/ (if ASR multi-task worked)")
    print("  - best_simple_ser_model.pth (if PyTorch worked)")
    print("=" * 60)

if __name__ == "__main__":
    main()