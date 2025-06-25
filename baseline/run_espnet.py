# ============================================================================
# 文件5: run_espnet.py - ESP-net运行脚本
# ============================================================================

#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
from pathlib import Path

# 确保模型已注册
from register_model import register_wavlm_ecapa_model, test_model_creation

def prepare_espnet_data(baseline_dir, output_dir="data"):
    """准备ESP-net格式数据"""
    # 复用之前的data_prep.py逻辑
    from data_prep import prepare_msp_data
    return prepare_msp_data(baseline_dir, output_dir)

def run_espnet_training():
    """运行ESP-net训练"""
    
    data_dir = Path("data")
    exp_dir = Path("exp/wavlm_ecapa_baseline")
    config_file = "train_config.yaml"
    
    # 检查数据
    for split in ["train", "valid"]:
        speech_scp = data_dir / split / "speech.scp"
        emotion_txt = data_dir / split / "emotion.txt"
        
        if not speech_scp.exists() or not emotion_txt.exists():
            print(f"❌ {split} data files not found")
            print("Please run data preparation first")
            return False
    
    # ESP-net训练命令
    train_cmd = [
        "python", "-m", "espnet2.bin.ser_train",
        "--use_preprocessor", "true",
        "--train_data_path_and_name_and_type", f"{data_dir}/train/speech.scp,speech,sound",
        "--train_data_path_and_name_and_type", f"{data_dir}/train/emotion.txt,emotion,text",
        "--valid_data_path_and_name_and_type", f"{data_dir}/valid/speech.scp,speech,sound", 
        "--valid_data_path_and_name_and_type", f"{data_dir}/valid/emotion.txt,emotion,text",
        "--output_dir", str(exp_dir),
        "--config", config_file,
        "--ngpu", "1",
        "--num_workers", "4",
    ]
    
    print("🚀 Starting ESP-net training...")
    print("Command:", " ".join(train_cmd))
    
    try:
        result = subprocess.run(train_cmd, check=True, capture_output=False)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        return False

def run_espnet_evaluation():
    """运行ESP-net评估"""
    
    data_dir = Path("data")
    exp_dir = Path("exp/wavlm_ecapa_baseline")
    
    # 找最佳模型
    model_files = [
        exp_dir / "valid.macro_f1.best.pth",
        exp_dir / "valid.acc.best.pth", 
        exp_dir / "valid.loss.best.pth"
    ]
    
    model_file = None
    for mf in model_files:
        if mf.exists():
            model_file = mf
            break
    
    if model_file is None:
        print("❌ No trained model found")
        return False
    
    print(f"📊 Using model: {model_file}")
    
    # ESP-net推理命令
    eval_cmd = [
        "python", "-m", "espnet2.bin.ser_inference",
        "--train_data_path_and_name_and_type", f"{data_dir}/train/speech.scp,speech,sound",
        "--train_data_path_and_name_and_type", f"{data_dir}/train/emotion.txt,emotion,text",
        "--valid_data_path_and_name_and_type", f"{data_dir}/test/speech.scp,speech,sound",
        "--valid_data_path_and_name_and_type", f"{data_dir}/test/emotion.txt,emotion,text", 
        "--output_dir", str(exp_dir / "results"),
        "--model_file", str(model_file),
        "--ngpu", "1",
    ]
    
    print("📊 Starting ESP-net evaluation...")
    
    try:
        result = subprocess.run(eval_cmd, check=True, capture_output=False)
        print("✅ Evaluation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description='ESP-net WavLM + ECAPA-TDNN Baseline')
    parser.add_argument('--baseline_dir', type=str, required=True,
                       help='Path to MSP-PODCAST baseline directory')
    parser.add_argument('--stage', type=str, default='all',
                       choices=['data', 'train', 'eval', 'all'],
                       help='Which stage to run')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🎵 ESP-net WavLM + ECAPA-TDNN Baseline")
    print("🎯 Optimized for Macro-F1")
    print("="*60)
    
    # 1. 模型注册
    print("🔧 Step 1: Model Registration")
    if not test_model_creation():
        sys.exit(1)
    if not register_wavlm_ecapa_model():
        sys.exit(1)
    
    # 2. 数据准备
    if args.stage in ['data', 'all']:
        print("\n🔧 Step 2: Data Preparation")
        prepare_espnet_data(args.baseline_dir)
    
    # 3. 模型训练
    if args.stage in ['train', 'all']:
        print("\n🚀 Step 3: Model Training")
        if not run_espnet_training():
            sys.exit(1)
    
    # 4. 模型评估
    if args.stage in ['eval', 'all']:
        print("\n📊 Step 4: Model Evaluation")
        if not run_espnet_evaluation():
            sys.exit(1)
    
    print("\n" + "="*60)
    print("🎉 ESP-net WavLM + ECAPA-TDNN Baseline Completed!")
    print("📁 Check results in: exp/wavlm_ecapa_baseline/")
    print("🎯 Focus metric: Macro-F1")
    print("="*60)

if __name__ == "__main__":
    main()