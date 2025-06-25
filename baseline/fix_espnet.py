#!/usr/bin/env python3
# fix_espnet.py - ESP-net诊断和修复脚本

import subprocess
import sys
import os
from pathlib import Path

def check_python_env():
    """检查Python环境"""
    print("🐍 Python Environment Check:")
    print(f"  Python path: {sys.executable}")
    print(f"  Python version: {sys.version}")
    print(f"  Virtual env: {os.environ.get('VIRTUAL_ENV', 'None')}")
    print(f"  Conda env: {os.environ.get('CONDA_DEFAULT_ENV', 'None')}")
    print()

def check_espnet_installation():
    """检查ESP-net安装状态"""
    print("🔍 ESP-net Installation Check:")
    
    # 检查是否能导入espnet
    try:
        import espnet
        print(f"✅ ESP-net imported successfully")
        print(f"  Version: {espnet.__version__}")
        print(f"  Location: {espnet.__file__}")
    except ImportError as e:
        print(f"❌ Cannot import espnet: {e}")
        return False
    
    # 检查espnet2
    try:
        import espnet2
        print(f"✅ ESP-net2 imported successfully")
        print(f"  Location: {espnet2.__file__}")
    except ImportError as e:
        print(f"❌ Cannot import espnet2: {e}")
        return False
    
    # 检查具体的SER模块
    try:
        from espnet2.bin import ser_train
        print(f"✅ SER training module found")
    except ImportError as e:
        print(f"❌ Cannot import SER training module: {e}")
        return False
    
    # 检查其他关键模块
    try:
        from espnet2.tasks.ser import SERTask
        print(f"✅ SER task module found")
    except ImportError as e:
        print(f"⚠️  SER task module not found: {e}")
        print(f"   This might be a version issue")
    
    return True

def fix_espnet_installation():
    """修复ESP-net安装"""
    print("\n🔧 Fixing ESP-net Installation:")
    
    # 卸载旧版本
    print("1. Uninstalling old ESP-net versions...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "espnet", "-y"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "espnet2", "-y"], check=False)
    except:
        pass
    
    # 安装最新版本
    print("2. Installing latest ESP-net...")
    try:
        # 安装预依赖
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchaudio", "transformers", "librosa", "soundfile"
        ], check=True)
        
        # 安装ESP-net
        subprocess.run([
            sys.executable, "-m", "pip", "install", "espnet"
        ], check=True)
        
        print("✅ ESP-net installation completed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False

def install_from_source():
    """从源码安装ESP-net"""
    print("\n🛠️  Installing ESP-net from source:")
    
    try:
        # 克隆仓库
        print("1. Cloning ESP-net repository...")
        subprocess.run([
            "git", "clone", "https://github.com/espnet/espnet.git", "/tmp/espnet"
        ], check=True)
        
        # 切换到目录
        os.chdir("/tmp/espnet")
        
        # 安装
        print("2. Installing from source...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        
        print("✅ Source installation completed")
        return True
        
    except Exception as e:
        print(f"❌ Source installation failed: {e}")
        return False

def create_simple_baseline():
    """创建一个不依赖ESP-net的简单baseline"""
    
    print("\n🎯 Creating Simple PyTorch Baseline (No ESP-net):")
    
    simple_baseline = '''#!/usr/bin/env python3
# simple_pytorch_baseline.py - 不依赖ESP-net的简单baseline

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import json
import numpy as np
from sklearn.metrics import f1_score, classification_report
from pathlib import Path
import random
from tqdm import tqdm

class SimpleDataset(Dataset):
    def __init__(self, data_dir, split='train', max_length=16000*5):
        self.max_length = max_length
        self.data = []
        
        # 读取数据
        speech_file = Path(data_dir) / split / "speech.scp"
        emotion_file = Path(data_dir) / split / "emotion.txt"
        
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
        
        # 加载音频
        try:
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

class SimpleEmoModel(nn.Module):
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
        # x: (batch, time)
        x = x.unsqueeze(1)  # (batch, 1, time)
        x = self.features(x)  # (batch, 256, 1)
        x = x.squeeze(-1)  # (batch, 256)
        x = self.classifier(x)  # (batch, num_classes)
        return x

def train_simple_model():
    print("🚀 Training Simple PyTorch Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据加载
    train_dataset = SimpleDataset("tiny_data", "train")
    valid_dataset = SimpleDataset("tiny_data", "valid")
    test_dataset = SimpleDataset("tiny_data", "test")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
    
    # 模型
    model = SimpleEmoModel(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练1个epoch
    model.train()
    train_loss = 0
    
    for batch_idx, (audio, labels) in enumerate(train_loader):
        audio, labels = audio.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(audio)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    # 验证
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for audio, labels in valid_loader:
            audio, labels = audio.to(device), labels.to(device)
            outputs = model(audio)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"\\n🎯 Results:")
    print(f"Training Loss: {train_loss/len(train_loader):.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Macro-F1: {macro_f1:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), "simple_emotion_model.pth")
    print("✅ Model saved to simple_emotion_model.pth")
    
    return True

if __name__ == "__main__":
    train_simple_model()
'''
    
    with open("simple_pytorch_baseline.py", 'w') as f:
        f.write(simple_baseline)
    
    print("✅ Created simple_pytorch_baseline.py")
    return True

def main():
    print("🔧 ESP-net Diagnosis and Fix Tool")
    print("=" * 40)
    
    # 1. 检查Python环境
    check_python_env()
    
    # 2. 检查ESP-net安装
    if check_espnet_installation():
        print("✅ ESP-net is properly installed!")
        print("🤔 The issue might be with the specific SER command")
        print("💡 Try running: python -m espnet2.bin.ser_train --help")
        return
    
    # 3. 尝试修复安装
    print("🔧 ESP-net not found or incomplete. Attempting to fix...")
    
    choice = input("\\nChoose repair method:\\n1. Reinstall from pip\\n2. Install from source\\n3. Create simple PyTorch baseline\\nEnter choice (1/2/3): ")
    
    if choice == '1':
        if fix_espnet_installation():
            print("🎉 ESP-net fixed! Try running minimal_test.py again")
        else:
            print("❌ Repair failed. Try option 2 or 3")
    
    elif choice == '2':
        if install_from_source():
            print("🎉 ESP-net installed from source! Try running minimal_test.py again")
        else:
            print("❌ Source installation failed. Try option 3")
    
    elif choice == '3':
        if create_simple_baseline():
            print("🎉 Simple baseline created!")
            print("Run: python simple_pytorch_baseline.py")
        else:
            print("❌ Failed to create simple baseline")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()