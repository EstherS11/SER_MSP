#!/usr/bin/env python3
# minimal_test.py - 超级简单的测试脚本

import json
import os
import random
from pathlib import Path

# 你的数据路径
DATA_ROOT = "/data/user_data/esthers/SER_MSP"

def create_tiny_dataset():
    """创建超小数据集：每个集合只有10个样本"""
    
    print("🎯 Creating tiny test dataset...")
    
    emotion_map = {'N': 0, 'H': 1, 'S': 2, 'A': 3, 'F': 4, 'D': 5, 'U': 6, 'C': 7, 'O': 8, 'X': 9}
    
    # 创建输出目录
    output_dir = Path("tiny_data")
    output_dir.mkdir(exist_ok=True)
    
    # 处理每个split，每个只取10个样本
    for split in ['train', 'valid', 'test']:
        json_file = f"{DATA_ROOT}/msp_{split}_10class.json"
        
        print(f"Processing {split}...")
        
        with open(json_file, 'r') as f:
            full_data = json.load(f)
        
        # 随机选择10个样本
        all_items = list(full_data.items())
        selected = random.sample(all_items, min(10, len(all_items)))
        
        # 创建目录
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)
        
        # 写入文件
        with open(split_dir / "speech.scp", 'w') as scp_f, \
             open(split_dir / "emotion.txt", 'w') as emo_f:
            
            for utt_id, info in selected:
                wav_path = info['wav']
                emotion = info['emo']
                
                if os.path.exists(wav_path) and emotion in emotion_map:
                    scp_f.write(f"{utt_id} {wav_path}\\n")
                    emo_f.write(f"{utt_id} {emotion_map[emotion]}\\n")
        
        print(f"✅ {split}: 10 samples")
    
    print("✅ Tiny dataset created!")

def create_minimal_config():
    """创建最简配置：1个epoch，最小参数"""
    
    config = '''# minimal_config.yaml
batch_type: numel
batch_size: 2
max_epoch: 1
patience: 1
seed: 42
num_workers: 1
log_interval_steps: 1

optim: adam
optim_conf:
  lr: 0.01

model: espnet_ser
model_conf:
  num_class: 10

preprocessor: default
preprocessor_conf: {}

best_model_criterion:
  - ["valid", "acc", "max"]

resume: false
keep_nbest_models: 1
use_tensorboard: false
'''
    
    with open("minimal_config.yaml", 'w') as f:
        f.write(config)
    
    print("✅ Minimal config created!")

def run_minimal_test():
    """运行最小测试"""
    
    import subprocess
    import sys
    
    print("🚀 Running minimal test...")
    
    cmd = [
        sys.executable, "-m", "espnet2.bin.ser_train",
        "--use_preprocessor", "true",
        "--train_data_path_and_name_and_type", "tiny_data/train/speech.scp,speech,sound",
        "--train_data_path_and_name_and_type", "tiny_data/train/emotion.txt,emotion,text",
        "--valid_data_path_and_name_and_type", "tiny_data/valid/speech.scp,speech,sound",
        "--valid_data_path_and_name_and_type", "tiny_data/valid/emotion.txt,emotion,text",
        "--output_dir", "exp/minimal_test",
        "--config", "minimal_config.yaml",
        "--ngpu", "1",
    ]
    
    print("Command:", " ".join(cmd))
    print("Expected time: 2-5 minutes")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Minimal test PASSED!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("⚡ Super Minimal ESP-net Test")
    print("=" * 30)
    print("📊 Data: 10 samples per split")
    print("⏱️  Time: ~2-5 minutes")
    print("🎯 Goal: Check if ESP-net works")
    print("=" * 30)
    
    random.seed(42)
    
    # 1. 创建超小数据集
    create_tiny_dataset()
    
    # 2. 创建最简配置
    create_minimal_config()
    
    # 3. 运行测试
    if run_minimal_test():
        print("\\n🎉 SUCCESS! ESP-net pipeline works!")
        print("🚀 You can now run the full training")
    else:
        print("\\n❌ Something is wrong with the setup")
        print("🔧 Check the error messages above")

if __name__ == "__main__":
    main()