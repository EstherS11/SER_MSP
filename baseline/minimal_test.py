#!/usr/bin/env python3
# minimal_test.py - 超级简单的测试脚本

import json
import os
import random
import subprocess
import sys
from pathlib import Path

# 你的数据路径
DATA_ROOT = "/data/user_data/esthers/SER_MSP"

def ensure_scripts_exist():
    """确保必要的脚本存在"""
    ser_train_script = Path("ser_train.py")
    register_script = Path("register_model.py")
    
    if not ser_train_script.exists():
        print("🔧 SER training script not found, creating...")
        if register_script.exists():
            try:
                result = subprocess.run([sys.executable, "register_model.py"], 
                                      check=True, capture_output=True, text=True)
                print("✅ Scripts created successfully")
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to create scripts: {e}")
                print(f"stdout: {e.stdout}")
                print(f"stderr: {e.stderr}")
                return False
        else:
            print("❌ register_model.py not found!")
            print("Please make sure you have all the required files:")
            print("  - espnet_ser_model.py")
            print("  - register_model.py")
            return False
    else:
        print("✅ SER training script found")
        return True

def create_tiny_dataset():
    """创建超小数据集：每个集合只有10个样本"""
    
    print("🎯 Creating tiny test dataset...")
    
    emotion_map = {'N': 0, 'H': 1, 'S': 2, 'A': 3, 'F': 4, 'D': 5, 'U': 6, 'C': 7, 'O': 8, 'X': 9}
    
    # 创建输出目录
    output_dir = Path("tiny_data")
    output_dir.mkdir(exist_ok=True)
    
    # 处理每个split，每个只取10个样本
    total_samples = 0
    for split in ['train', 'valid', 'test']:
        json_file = f"{DATA_ROOT}/msp_{split}_10class.json"
        
        print(f"Processing {split}...")
        
        if not Path(json_file).exists():
            print(f"❌ JSON file not found: {json_file}")
            continue
        
        with open(json_file, 'r') as f:
            full_data = json.load(f)
        
        # 随机选择10个样本
        all_items = list(full_data.items())
        selected = random.sample(all_items, min(10, len(all_items)))
        
        # 创建目录
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)
        
        valid_count = 0
        # 写入文件
        with open(split_dir / "speech.scp", 'w') as scp_f, \
             open(split_dir / "emotion.txt", 'w') as emo_f:
            
            for utt_id, info in selected:
                wav_path = info['wav']
                emotion = info['emo']
                
                # 检查文件存在且标签有效
                if os.path.exists(wav_path) and emotion in emotion_map:
                    scp_f.write(f"{utt_id} {wav_path}\n")
                    emo_f.write(f"{utt_id} {emotion_map[emotion]}\n")
                    valid_count += 1
        
        print(f"✅ {split}: {valid_count} valid samples")
        total_samples += valid_count
    
    if total_samples == 0:
        print("❌ No valid samples found!")
        return False
    
    print(f"✅ Tiny dataset created! Total: {total_samples} samples")
    return True

def create_minimal_config():
    """创建最简配置：1个epoch，最小参数"""
    
    config = '''# minimal_config.yaml - 极简测试配置
batch_type: numel
batch_size: 2
max_epoch: 1
patience: 1
seed: 42
num_workers: 1
log_interval_steps: 1
grad_clip: 5.0
accum_grad: 1

# 优化器配置
optim: adam
optim_conf:
  lr: 0.01

# 学习率调度器
scheduler: constantlr
scheduler_conf: {}

# 模型配置 (通过命令行参数传递)
num_class: 10
wavlm_model_name: "microsoft/wavlm-base"  # 使用base版本更快
wavlm_freeze: true
ecapa_channels: [256, 256]  # 减少参数
ecapa_kernels: [3, 3]
ecapa_dilations: [1, 2]
context_dim: 512           # 减少参数
embedding_dim: 128         # 减少参数
loss_type: "cross_entropy"
label_smoothing: 0.0
save_macro_f1: true

# 预处理器配置
preprocessor: default
preprocessor_conf:
  # 关闭SpecAugment以加速
  spec_augment: false

# 最佳模型选择
best_model_criterion:
  - ["valid", "acc", "max"]
  - ["valid", "loss", "min"]

# 其他配置
resume: false
keep_nbest_models: 1
use_tensorboard: false
'''
    
    with open("minimal_config.yaml", 'w') as f:
        f.write(config)
    
    print("✅ Minimal config created!")

def check_data_files():
    """检查数据文件是否正确创建"""
    print("🔍 Checking data files...")
    
    data_dir = Path("tiny_data")
    if not data_dir.exists():
        print("❌ tiny_data directory not found")
        return False
    
    for split in ['train', 'valid']:
        speech_scp = data_dir / split / "speech.scp"
        emotion_txt = data_dir / split / "emotion.txt"
        
        if not speech_scp.exists():
            print(f"❌ {speech_scp} not found")
            return False
        
        if not emotion_txt.exists():
            print(f"❌ {emotion_txt} not found")
            return False
        
        # 检查文件内容
        with open(speech_scp, 'r') as f:
            speech_lines = f.readlines()
        
        with open(emotion_txt, 'r') as f:
            emotion_lines = f.readlines()
        
        if len(speech_lines) == 0 or len(emotion_lines) == 0:
            print(f"❌ {split} data files are empty")
            return False
        
        if len(speech_lines) != len(emotion_lines):
            print(f"❌ {split} speech and emotion files have different lengths")
            return False
        
        print(f"✅ {split}: {len(speech_lines)} samples")
    
    return True

def run_minimal_test():
    """运行最小测试"""
    
    print("🚀 Running minimal test...")
    
    # 使用我们自己的训练脚本
    cmd = [
        sys.executable, "ser_train.py",
        "--config", "minimal_config.yaml",
        "--train_data_path_and_name_and_type", "tiny_data/train/speech.scp,speech,sound",
        "--train_data_path_and_name_and_type", "tiny_data/train/emotion.txt,emotion,text",
        "--valid_data_path_and_name_and_type", "tiny_data/valid/speech.scp,speech,sound",
        "--valid_data_path_and_name_and_type", "tiny_data/valid/emotion.txt,emotion,text",
        "--output_dir", "exp/minimal_test",
        "--ngpu", "1",
        "--use_preprocessor", "true",
    ]
    
    print("Command:", " ".join(cmd))
    print("Expected time: 2-5 minutes")
    print()
    
    try:
        # 创建输出目录
        Path("exp/minimal_test").mkdir(parents=True, exist_ok=True)
        
        # 运行训练
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ Minimal test PASSED!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed with exit code: {e.returncode}")
        return False
    except FileNotFoundError as e:
        print(f"\n❌ Script not found: {e}")
        print("Make sure ser_train.py exists")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def main():
    print("⚡ Super Minimal ESP-net SER Test")
    print("="*40)
    print("📊 Data: ~10 samples per split")
    print("⏱️  Time: ~2-5 minutes")
    print("🎯 Goal: Check if ESP-net SER works")
    print("🏗️  Model: WavLM-base + mini ECAPA-TDNN")
    print("="*40)
    
    random.seed(42)
    
    # 0. 确保脚本存在
    print("🔧 Step 0: Checking scripts...")
    if not ensure_scripts_exist():
        print("\n❌ Script setup failed!")
        print("Please run: python register_model.py")
        return
    
    # 1. 创建超小数据集
    print("\n📊 Step 1: Creating tiny dataset...")
    if not create_tiny_dataset():
        print("\n❌ Data preparation failed!")
        return
    
    # 2. 创建最简配置
    print("\n⚙️  Step 2: Creating minimal config...")
    create_minimal_config()
    
    # 3. 检查数据文件
    print("\n🔍 Step 3: Checking data files...")
    if not check_data_files():
        print("\n❌ Data validation failed!")
        return
    
    # 4. 运行测试
    print("\n🚀 Step 4: Running minimal test...")
    if run_minimal_test():
        print("\n" + "="*40)
        print("🎉 SUCCESS! ESP-net SER pipeline works!")
        print("🚀 You can now run the full training")
        print("\n📁 Check results:")
        print("  - Model: exp/minimal_test/")
        print("  - Logs: exp/minimal_test/train.log")
        print("="*40)
    else:
        print("\n" + "="*40)
        print("❌ Something is wrong with the setup")
        print("🔧 Troubleshooting:")
        print("  1. Check if all required files exist")
        print("  2. Verify data paths are correct")
        print("  3. Check ESP-net installation")
        print("  4. Look at error messages above")
        print("="*40)

if __name__ == "__main__":
    main()