#!/usr/bin/env python3
# cloud_run.py - 云端集群专用运行脚本

import sys
import os
import json
import subprocess
from pathlib import Path

# ============================================================================
# 云端集群配置 - 根据你的实际路径
# ============================================================================

# 🔧 你的实际数据路径
DATA_ROOT = "/data/user_data/esthers/SER_MSP"
BASELINE_DIR = "/data/user_data/esthers/SER_MSP/baseline"  # 这个目录可能不存在，我们会创建
AUDIO_DIR = "/data/user_data/esthers/SER_MSP/DATA/Audios"

# JSON文件路径（直接在SER_MSP目录下）
JSON_FILES = {
    'train': f"{DATA_ROOT}/msp_train_10class.json",
    'valid': f"{DATA_ROOT}/msp_valid_10class.json", 
    'test': f"{DATA_ROOT}/msp_test_10class.json"
}

# ============================================================================
# 云端环境检查和设置
# ============================================================================

def setup_cloud_environment():
    """设置云端环境"""
    project_root = Path(__file__).parent.absolute()
    
    # 添加项目根目录到Python路径
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 创建__init__.py文件
    init_file = project_root / "__init__.py"
    if not init_file.exists():
        init_file.touch()
        print("✅ Created __init__.py")
    
    # 创建baseline目录（如果不存在）
    baseline_path = Path(BASELINE_DIR)
    baseline_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Baseline directory: {BASELINE_DIR}")
    
    return project_root

def check_cloud_data():
    """检查云端数据结构"""
    print("🔍 Checking cloud data structure...")
    
    # 检查音频目录
    audio_path = Path(AUDIO_DIR)
    if not audio_path.exists():
        print(f"❌ Audio directory not found: {AUDIO_DIR}")
        return False
    
    audio_files = list(audio_path.glob("*.wav"))
    print(f"✅ Found {len(audio_files)} audio files in {AUDIO_DIR}")
    
    # 检查JSON文件
    for split, json_path in JSON_FILES.items():
        if not Path(json_path).exists():
            print(f"❌ Missing JSON file: {json_path}")
            return False
        
        # 检查JSON格式
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            print(f"✅ {split}: {len(data)} samples in {json_path}")
            
            # 验证数据格式
            first_key = next(iter(data))
            first_item = data[first_key]
            if 'wav' not in first_item or 'emo' not in first_item:
                print(f"❌ Invalid JSON format in {json_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error reading {json_path}: {e}")
            return False
    
    return True

def create_cloud_data_prep():
    """创建适配云端数据结构的数据准备脚本"""
    
    data_prep_code = '''#!/usr/bin/env python3
# cloud_data_prep.py - 云端数据准备脚本

import json
import os
import numpy as np
from pathlib import Path
from collections import Counter

def prepare_cloud_msp_data(output_dir="data"):
    """准备云端MSP-PODCAST数据"""
    
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
    
    print("🔧 Preparing ESP-net format data for cloud MSP-PODCAST...")
    
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
        
        # 创建ESP-net标准格式文件
        with open(split_dir / "speech.scp", 'w') as scp_f, \\
             open(split_dir / "emotion.txt", 'w') as emo_f, \\
             open(split_dir / "utt2spk", 'w') as spk_f:
            
            for utt_id, info in data.items():
                wav_path = info['wav']
                emotion = info['emo']
                duration = info.get('length', 0)
                
                # 检查文件存在且标签有效
                if not os.path.exists(wav_path):
                    missing_files.append(wav_path)
                    continue
                    
                if emotion not in emotion_map:
                    print(f"⚠️  Unknown emotion '{emotion}' for {utt_id}")
                    continue
                
                # 写入ESP-net格式文件
                scp_f.write(f"{utt_id} {wav_path}\\n")
                emo_f.write(f"{utt_id} {emotion_map[emotion]}\\n")
                
                # 简单的speaker ID (从utterance ID提取)
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
            'duration_min': np.min(durations) if durations else 0,
            'duration_max': np.max(durations) if durations else 0,
            'emotion_dist': emotion_counts
        }
        
        print(f"✅ {split}: {valid_count} valid samples")
        if missing_files:
            print(f"⚠️  {split}: {len(missing_files)} missing files")
            # 显示前几个缺失文件作为示例
            for i, missing in enumerate(missing_files[:3]):
                print(f"     Missing: {missing}")
            if len(missing_files) > 3:
                print(f"     ... and {len(missing_files) - 3} more")
    
    # 保存数据集信息
    dataset_info = {
        "dataset_name": "MSP-PODCAST",
        "task": "10-class emotion recognition",
        "num_classes": 10,
        "emotion_names": emotion_names,
        "emotion_mapping": emotion_map,
        "stats": stats,
        "data_source": "Cloud cluster: /data/user_data/esthers/SER_MSP"
    }
    
    with open(output_path / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"\\n✅ ESP-net data prepared in {output_dir}/")
    
    # 显示详细统计
    print("\\n📊 Dataset Statistics:")
    print(f"Dataset: {dataset_info['dataset_name']}")
    print(f"Task: {dataset_info['task']}")
    print(f"Classes: {dataset_info['num_classes']}")
    
    for split in ['train', 'valid', 'test']:
        if split in stats:
            print(f"\\n{split.upper()} Set:")
            print(f"  Samples: {stats[split]['count']:,}")
            print(f"  Missing files: {stats[split]['missing_files']}")
            print(f"  Duration: {stats[split]['duration_mean']:.2f}±{stats[split]['duration_std']:.2f}s")
            print(f"  Range: [{stats[split]['duration_min']:.2f}, {stats[split]['duration_max']:.2f}]s")
            
            print(f"  Emotion distribution:")
            total = stats[split]['count']
            for emotion, count in stats[split]['emotion_dist'].items():
                pct = count / total * 100 if total > 0 else 0
                emotion_name = emotion_names[emotion_map[emotion]]
                print(f"    {emotion} ({emotion_name}): {count:,} ({pct:.1f}%)")
    
    # 类别平衡分析
    if 'train' in stats:
        print(f"\\n⚖️  Class Balance Analysis (Training Set):")
        train_dist = stats['train']['emotion_dist']
        counts = list(train_dist.values())
        if counts:
            max_count = max(counts)
            min_count = min(counts)
            balance_ratio = min_count / max_count
            print(f"  Balance ratio: {balance_ratio:.3f} (1.0 = perfect)")
            
            if balance_ratio < 0.5:
                print(f"  ⚠️  Severe class imbalance detected!")
                print(f"     Consider using class weights or data resampling")
    
    return dataset_info

if __name__ == "__main__":
    prepare_cloud_msp_data()
'''
    
    with open("cloud_data_prep.py", 'w') as f:
        f.write(data_prep_code)
    
    print("✅ Created cloud_data_prep.py")

def create_cloud_train_config():
    """创建云端训练配置"""
    
    config = '''# cloud_train_config.yaml - 云端集群训练配置
batch_type: numel
batch_size: 12                      # 云端可能有更好的GPU
max_epoch: 50
patience: 10
seed: 42
num_workers: 4                      # 云端并行处理
log_interval_steps: 100
grad_clip: 5.0
accum_grad: 1

# 优化器配置
optim: adamw
optim_conf:
  lr: 0.0001
  weight_decay: 0.001
  betas: [0.9, 0.999]
  eps: 1.0e-8

# 学习率调度器
scheduler: cosineannealinglr
scheduler_conf:
  T_max: 50
  eta_min: 1.0e-6

# 模型配置 (通过命令行参数传递)
num_class: 10
wavlm_model_name: "microsoft/wavlm-large"
wavlm_freeze: true
ecapa_channels: [512, 512, 512]
ecapa_kernels: [5, 3, 3]
ecapa_dilations: [1, 2, 3]
context_dim: 1536
embedding_dim: 256
loss_type: "cross_entropy"
focal_gamma: 2.0
label_smoothing: 0.1
save_macro_f1: true

# 预处理器配置
preprocessor: default
preprocessor_conf:
  # SpecAugment
  spec_augment: true
  spec_augment_conf:
    apply_time_warp: true
    time_warp_window: 5
    apply_freq_mask: true
    freq_mask_width_range: [0, 30]
    num_freq_mask: 2
    apply_time_mask: true
    time_mask_width_range: [0, 40]
    num_time_mask: 2

# 最佳模型选择（重点关注macro-F1）
best_model_criterion:
  - ["valid", "macro_f1", "max"]
  - ["valid", "acc", "max"]
  - ["valid", "loss", "min"]

# 其他配置
resume: true
keep_nbest_models: 5
use_tensorboard: true
'''
    
    with open("cloud_train_config.yaml", 'w') as f:
        f.write(config)
    
    print("✅ Created cloud_train_config.yaml")

def check_cloud_dependencies():
    """检查云端依赖"""
    print("📦 Checking cloud dependencies...")
    
    required_packages = {
        'espnet': 'espnet',
        'transformers': 'transformers', 
        'sklearn': 'scikit-learn',
        'torch': 'torch',
        'torchaudio': 'torchaudio',
        'yaml': 'pyyaml',
        'numpy': 'numpy'
    }
    
    missing = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            missing.append(package_name)
            print(f"❌ {package_name}")
    
    if missing:
        print(f"\n🔧 Installing missing packages...")
        install_cmd = [sys.executable, "-m", "pip", "install"] + missing
        print(f"Command: {' '.join(install_cmd)}")
        
        try:
            subprocess.run(install_cmd, check=True)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    return True

def ensure_cloud_scripts():
    """确保云端脚本存在"""
    # 确保注册脚本已运行
    register_script = Path("register_model.py")
    ser_train_script = Path("ser_train.py")
    
    if not ser_train_script.exists():
        print("🔧 Creating necessary scripts...")
        if register_script.exists():
            try:
                subprocess.run([sys.executable, "register_model.py"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to run register_model.py: {e}")
                return False
        else:
            print("❌ register_model.py not found!")
            return False
    
    return True

def run_cloud_training():
    """在云端运行训练"""
    print("\n🚀 Starting cloud training...")
    
    # 1. 数据准备
    print("📊 Step 1: Data preparation...")
    try:
        from cloud_data_prep import prepare_cloud_msp_data
        dataset_info = prepare_cloud_msp_data()
        
        # 检查数据准备结果
        data_dir = Path("data")
        total_samples = 0
        for split in ["train", "valid", "test"]:
            speech_scp = data_dir / split / "speech.scp"
            emotion_txt = data_dir / split / "emotion.txt"
            
            if speech_scp.exists() and emotion_txt.exists():
                with open(speech_scp, 'r') as f:
                    samples = len(f.readlines())
                total_samples += samples
                print(f"  {split}: {samples} samples")
        
        if total_samples == 0:
            print("❌ No valid samples found after data preparation")
            return False
        
        print(f"✅ Total samples prepared: {total_samples}")
        
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return False
    
    # 2. 检查脚本
    print("\n🔧 Step 2: Ensuring scripts...")
    if not ensure_cloud_scripts():
        print("❌ Script preparation failed")
        return False
    
    # 3. 模型训练
    print("\n🚀 Step 3: Model training...")
    
    exp_dir = Path("exp/cloud_wavlm_ecapa")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用我们自己的SER训练脚本
    train_cmd = [
        sys.executable, "ser_train.py",
        "--config", "cloud_train_config.yaml",
        "--train_data_path_and_name_and_type", "data/train/speech.scp,speech,sound",
        "--train_data_path_and_name_and_type", "data/train/emotion.txt,emotion,text",
        "--valid_data_path_and_name_and_type", "data/valid/speech.scp,speech,sound", 
        "--valid_data_path_and_name_and_type", "data/valid/emotion.txt,emotion,text",
        "--output_dir", str(exp_dir),
        "--ngpu", "1",
        "--num_workers", "4",
        "--use_preprocessor", "true",
    ]
    
    print("Training command:")
    print(" ".join(train_cmd))
    print()
    
    try:
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd()) + ':' + env.get('PYTHONPATH', '')
        
        result = subprocess.run(train_cmd, check=True, env=env)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False
    except FileNotFoundError as e:
        print(f"❌ Script not found: {e}")
        return False

def run_cloud_evaluation():
    """运行云端评估"""
    print("\n📊 Step 4: Model evaluation...")
    
    exp_dir = Path("exp/cloud_wavlm_ecapa")
    
    # 查找可用的模型文件
    model_patterns = [
        "valid.macro_f1.best.pth",
        "valid.acc.best.pth",
        "valid.loss.best.pth",
        "checkpoint.pth",
    ]
    
    model_file = None
    for pattern in model_patterns:
        candidate = exp_dir / pattern
        if candidate.exists():
            model_file = candidate
            break
    
    if model_file is None:
        print("❌ No trained model found")
        print("Available files in experiment directory:")
        if exp_dir.exists():
            for f in exp_dir.iterdir():
                if f.suffix == '.pth':
                    print(f"  - {f}")
        return False
    
    print(f"📊 Using model: {model_file}")
    
    # 检查测试数据
    test_data_dir = Path("data/test")
    if not test_data_dir.exists() or not (test_data_dir / "speech.scp").exists():
        print("⚠️  Test data not found, using validation data for evaluation")
        test_data_dir = Path("data/valid")
    
    # 运行评估
    eval_cmd = [
        sys.executable, "ser_inference.py",
        "--model_file", str(model_file),
        "--valid_data_path_and_name_and_type", f"{test_data_dir}/speech.scp,speech,sound",
        "--valid_data_path_and_name_and_type", f"{test_data_dir}/emotion.txt,emotion,text",
        "--output_dir", str(exp_dir / "evaluation"),
        "--ngpu", "1",
        "--batch_size", "32",
    ]
    
    print("Evaluation command:")
    print(" ".join(eval_cmd))
    print()
    
    try:
        result = subprocess.run(eval_cmd, check=True)
        print("✅ Evaluation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed: {e}")
        return False
    except FileNotFoundError:
        print("⚠️  Evaluation script not found, skipping evaluation")
        return True  # 不算失败

def main():
    print("🌐 Cloud Cluster ESP-net WavLM + ECAPA-TDNN")
    print("=" * 60)
    print(f"📁 Data root: {DATA_ROOT}")
    print(f"🎵 Audio dir: {AUDIO_DIR}")
    print("=" * 60)
    
    # 1. 环境设置
    print("\n🔧 Setting up cloud environment...")
    setup_cloud_environment()
    
    # 2. 检查数据
    print("\n📊 Checking cloud data...")
    if not check_cloud_data():
        print("❌ Cloud data check failed")
        print("Please check your data paths and JSON files")
        return
    
    # 3. 检查依赖
    print("\n📦 Checking dependencies...")
    if not check_cloud_dependencies():
        print("❌ Dependency check failed")
        return
    
    # 4. 创建云端配置文件
    print("\n🔧 Creating cloud configuration...")
    create_cloud_data_prep()
    create_cloud_train_config()
    
    # 5. 运行完整流程
    try:
        # 数据准备和训练
        if not run_cloud_training():
            print("\n❌ Cloud training failed")
            return
        
        # 评估
        run_cloud_evaluation()
        
        print("\n🎉 Cloud training completed successfully!")
        print("\n📊 Results location:")
        print(f"  - Model: exp/cloud_wavlm_ecapa/")
        print(f"  - Logs: exp/cloud_wavlm_ecapa/train.log")
        print(f"  - TensorBoard: exp/cloud_wavlm_ecapa/tensorboard/")
        print(f"  - Data info: data/dataset_info.json")
        
        # 显示训练结果摘要
        exp_dir = Path("exp/cloud_wavlm_ecapa")
        log_file = exp_dir / "train.log"
        if log_file.exists():
            print(f"\n📈 Check training progress:")
            print(f"  tail -f {log_file}")
        
        tensorboard_dir = exp_dir / "tensorboard"
        if tensorboard_dir.exists():
            print(f"\n📊 View TensorBoard:")
            print(f"  tensorboard --logdir {tensorboard_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()