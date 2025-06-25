#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
from pathlib import Path

def prepare_espnet_data(baseline_dir, output_dir="data"):
    """准备ESP-net格式数据"""
    # 检查是否有data_prep.py文件
    data_prep_file = Path("data_prep.py")
    if data_prep_file.exists():
        from data_prep import prepare_msp_data
        return prepare_msp_data(baseline_dir, output_dir)
    else:
        # 如果没有，创建一个简单的数据准备函数
        print("⚠️  data_prep.py not found, creating basic data preparation...")
        create_basic_data_prep()
        from data_prep import prepare_msp_data
        return prepare_msp_data(baseline_dir, output_dir)

def create_basic_data_prep():
    """创建基本的数据准备脚本"""
    basic_data_prep = '''#!/usr/bin/env python3
# data_prep.py - 基本数据准备脚本

import json
import os
from pathlib import Path

def prepare_msp_data(baseline_dir, output_dir="data"):
    """准备MSP-PODCAST数据"""
    print(f"🔧 Preparing data from {baseline_dir} to {output_dir}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 这里需要根据实际数据结构实现
    # 现在只创建目录结构
    for split in ["train", "valid", "test"]:
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        # 创建空文件作为占位符
        (split_dir / "speech.scp").touch()
        (split_dir / "emotion.txt").touch()
        (split_dir / "utt2spk").touch()
    
    print(f"✅ Basic data structure created in {output_dir}")
    return True

if __name__ == "__main__":
    prepare_msp_data("./baseline")
'''
    
    with open("data_prep.py", 'w') as f:
        f.write(basic_data_prep)
    
    print("✅ Created basic data_prep.py")

def ensure_scripts_exist():
    """确保必要的脚本存在"""
    # 确保注册脚本已运行
    register_script = Path("register_model.py")
    ser_train_script = Path("ser_train.py")
    
    if not ser_train_script.exists():
        print("🔧 Running model registration...")
        if register_script.exists():
            subprocess.run([sys.executable, "register_model.py"], check=True)
        else:
            print("❌ register_model.py not found!")
            return False
    
    return True

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
        
        # 检查文件是否为空
        if speech_scp.stat().st_size == 0 or emotion_txt.stat().st_size == 0:
            print(f"⚠️  {split} data files are empty")
            print("Please check your data preparation")
    
    # 确保脚本存在
    if not ensure_scripts_exist():
        return False
    
    # 创建实验目录
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # ESP-net训练命令 - 使用我们自己的训练脚本
    train_cmd = [
        sys.executable, "ser_train.py",
        "--config", config_file,
        "--train_data_path_and_name_and_type", f"{data_dir}/train/speech.scp,speech,sound",
        "--train_data_path_and_name_and_type", f"{data_dir}/train/emotion.txt,emotion,text",
        "--valid_data_path_and_name_and_type", f"{data_dir}/valid/speech.scp,speech,sound", 
        "--valid_data_path_and_name_and_type", f"{data_dir}/valid/emotion.txt,emotion,text",
        "--output_dir", str(exp_dir),
        "--ngpu", "1",
        "--num_workers", "4",
        "--use_preprocessor", "true",
    ]
    
    print("🚀 Starting ESP-net training...")
    print("Command:", " ".join(train_cmd))
    print()
    
    try:
        result = subprocess.run(train_cmd, check=True, capture_output=False)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        return False
    except FileNotFoundError as e:
        print(f"❌ Script not found: {e}")
        print("Please make sure ser_train.py exists and is executable")
        return False

def run_espnet_evaluation():
    """运行ESP-net评估"""
    
    data_dir = Path("data")
    exp_dir = Path("exp/wavlm_ecapa_baseline")
    
    # 找最佳模型
    model_files = [
        exp_dir / "valid.macro_f1.best.pth",
        exp_dir / "valid.acc.best.pth", 
        exp_dir / "valid.loss.best.pth",
        exp_dir / "checkpoint.pth",
    ]
    
    model_file = None
    for mf in model_files:
        if mf.exists():
            model_file = mf
            break
    
    if model_file is None:
        print("❌ No trained model found")
        print("Available files in exp directory:")
        if exp_dir.exists():
            for f in exp_dir.iterdir():
                if f.suffix == '.pth':
                    print(f"  - {f}")
        return False
    
    print(f"📊 Using model: {model_file}")
    
    # 检查是否有测试数据
    test_speech = data_dir / "test" / "speech.scp"
    test_emotion = data_dir / "test" / "emotion.txt"
    
    if not test_speech.exists() or not test_emotion.exists():
        print("⚠️  Test data not found, using validation data for evaluation")
        test_speech = data_dir / "valid" / "speech.scp"
        test_emotion = data_dir / "valid" / "emotion.txt"
    
    # ESP-net推理命令
    eval_cmd = [
        sys.executable, "ser_inference.py",
        "--model_file", str(model_file),
        "--valid_data_path_and_name_and_type", f"{test_speech},speech,sound",
        "--valid_data_path_and_name_and_type", f"{test_emotion},emotion,text", 
        "--output_dir", str(exp_dir / "results"),
        "--ngpu", "1",
        "--batch_size", "32",
    ]
    
    print("📊 Starting ESP-net evaluation...")
    print("Command:", " ".join(eval_cmd))
    print()
    
    try:
        result = subprocess.run(eval_cmd, check=True, capture_output=False)
        print("✅ Evaluation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed with exit code {e.returncode}")
        return False
    except FileNotFoundError as e:
        print(f"❌ Inference script not found: {e}")
        return False

def check_dependencies():
    """检查依赖"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        "torch",
        "torchaudio", 
        "transformers",
        "sklearn",
        "numpy",
        "yaml"
    ]
    
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"✅ {pkg}")
        except ImportError:
            missing.append(pkg)
            print(f"❌ {pkg}")
    
    if missing:
        print(f"\n🔧 Installing missing packages: {missing}")
        subprocess.run([sys.executable, "-m", "pip", "install"] + missing)
    
    # 检查ESP-net
    try:
        import espnet2
        print("✅ espnet2")
    except ImportError:
        print("❌ espnet2 - Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "espnet"])

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
    
    # 0. 检查依赖
    print("📦 Step 0: Checking Dependencies")
    check_dependencies()
    
    # 1. 脚本准备
    print("\n🔧 Step 1: Script Preparation")
    if not ensure_scripts_exist():
        print("❌ Script preparation failed")
        sys.exit(1)
    
    # 2. 数据准备
    if args.stage in ['data', 'all']:
        print("\n🔧 Step 2: Data Preparation")
        try:
            prepare_espnet_data(args.baseline_dir)
        except Exception as e:
            print(f"❌ Data preparation failed: {e}")
            if args.stage == 'data':
                sys.exit(1)
    
    # 3. 模型训练
    if args.stage in ['train', 'all']:
        print("\n🚀 Step 3: Model Training")
        if not run_espnet_training():
            sys.exit(1)
    
    # 4. 模型评估
    if args.stage in ['eval', 'all']:
        print("\n📊 Step 4: Model Evaluation")
        if not run_espnet_evaluation():
            print("⚠️  Evaluation failed, but training may have succeeded")
    
    print("\n" + "="*60)
    print("🎉 ESP-net WavLM + ECAPA-TDNN Baseline Completed!")
    print("📁 Check results in: exp/wavlm_ecapa_baseline/")
    print("🎯 Focus metric: Macro-F1")
    print("\n📊 Generated files:")
    print("  - Model checkpoints: exp/wavlm_ecapa_baseline/*.pth")
    print("  - Training logs: exp/wavlm_ecapa_baseline/train.log")
    print("  - TensorBoard logs: exp/wavlm_ecapa_baseline/tensorboard/")
    print("="*60)

if __name__ == "__main__":
    main()