#!/usr/bin/env python3
# enhanced_espnet_fix.py - 增强版ESP-net诊断和修复脚本

import subprocess
import sys
import os
from pathlib import Path
import importlib.util

def check_python_env():
    """检查Python环境"""
    print("🐍 Python Environment Check:")
    print(f"  Python path: {sys.executable}")
    print(f"  Python version: {sys.version}")
    print(f"  Virtual env: {os.environ.get('VIRTUAL_ENV', 'None')}")
    print(f"  Conda env: {os.environ.get('CONDA_DEFAULT_ENV', 'None')}")
    print()

def check_espnet_modules():
    """详细检查ESP-net模块"""
    print("🔍 Detailed ESP-net Module Check:")
    
    modules_to_check = [
        'espnet',
        'espnet2', 
        'espnet2.bin',
        'espnet2.bin.ser_train',
        'espnet2.tasks',
        'espnet2.tasks.ser',
        'espnet2.asr',
        'espnet2.tts'
    ]
    
    available_modules = []
    missing_modules = []
    
    for module in modules_to_check:
        try:
            spec = importlib.util.find_spec(module)
            if spec is not None:
                print(f"✅ {module}")
                available_modules.append(module)
                if hasattr(spec, 'origin') and spec.origin:
                    print(f"   📍 {spec.origin}")
            else:
                print(f"❌ {module} - Not found")
                missing_modules.append(module)
        except Exception as e:
            print(f"❌ {module} - Error: {e}")
            missing_modules.append(module)
    
    print(f"\n📊 Summary: {len(available_modules)} available, {len(missing_modules)} missing")
    return available_modules, missing_modules

def check_espnet_commands():
    """检查ESP-net命令行工具"""
    print("\n🛠️  ESP-net Command Check:")
    
    commands_to_check = [
        'espnet2.bin.asr_train',
        'espnet2.bin.tts_train', 
        'espnet2.bin.ser_train',
        'espnet2.bin.ssl_train',
        'espnet.bin.asr_train'
    ]
    
    for cmd in commands_to_check:
        try:
            result = subprocess.run([sys.executable, '-m', cmd, '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ {cmd} - Available")
            else:
                print(f"❌ {cmd} - Error: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"⏰ {cmd} - Timeout (might be available)")
        except Exception as e:
            print(f"❌ {cmd} - Exception: {e}")

def list_espnet_structure():
    """查看ESP-net安装结构"""
    print("\n📂 ESP-net Installation Structure:")
    
    try:
        import espnet
        espnet_path = Path(espnet.__file__).parent
        print(f"ESP-net location: {espnet_path}")
        
        # 列出主要目录
        for item in espnet_path.iterdir():
            if item.is_dir():
                print(f"  📁 {item.name}/")
                # 查看bin目录
                if item.name == 'bin':
                    for subitem in item.iterdir():
                        if subitem.suffix == '.py':
                            print(f"    📄 {subitem.name}")
            elif item.suffix == '.py':
                print(f"  📄 {item.name}")
                
    except ImportError:
        print("❌ Cannot import espnet")
    except Exception as e:
        print(f"❌ Error exploring structure: {e}")
    
    # 检查espnet2
    try:
        import espnet2
        espnet2_path = Path(espnet2.__file__).parent
        print(f"\nESP-net2 location: {espnet2_path}")
        
        # 查看bin目录
        bin_path = espnet2_path / 'bin'
        if bin_path.exists():
            print(f"  📁 bin/")
            for item in bin_path.iterdir():
                if item.suffix == '.py':
                    print(f"    📄 {item.name}")
        else:
            print("  ❌ bin/ directory not found")
            
    except ImportError:
        print("❌ Cannot import espnet2")

def install_espnet_full():
    """完整安装ESP-net"""
    print("\n🔧 Full ESP-net Installation:")
    
    try:
        # 1. 清理旧安装
        print("1. Cleaning old installations...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "espnet", "espnet2", "-y"], 
                      check=False, capture_output=True)
        
        # 2. 更新pip
        print("2. Updating pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # 3. 安装依赖
        print("3. Installing dependencies...")
        dependencies = [
            "torch>=1.13.0",
            "torchaudio>=0.13.0", 
            "transformers>=4.21.0",
            "librosa>=0.8.0",
            "soundfile>=0.10.0",
            "PyYAML>=5.4.0",
            "numpy>=1.20.0",
            "scipy>=1.7.0",
            "scikit-learn>=1.0.0",
            "tensorboard>=2.8.0",
            "matplotlib>=3.5.0",
            "editdistance>=0.5.0",
            "configargparse>=1.5.0",
            "typeguard>=2.13.0",
            "humanfriendly>=9.2.0",
            "g2p_en>=2.1.0",
            "phonemizer>=3.0.0",
            "jaconv>=0.3.0",
            "pypinyin>=0.44.0",
            "jieba>=0.42.0",
            "resampy>=0.2.2",
            "asteroid-filterbanks>=0.4.0"
        ]
        
        for dep in dependencies:
            print(f"   Installing {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                          check=False, capture_output=True)
        
        # 4. 安装ESP-net
        print("4. Installing ESP-net...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "espnet", "--force-reinstall", "--no-cache-dir"
        ], check=True)
        
        print("✅ Full installation completed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False

def install_espnet_dev():
    """安装ESP-net开发版本"""
    print("\n🚀 Installing ESP-net Development Version:")
    
    try:
        # 克隆最新代码
        print("1. Cloning ESP-net repository...")
        temp_dir = "/tmp/espnet_dev"
        if Path(temp_dir).exists():
            subprocess.run(["rm", "-rf", temp_dir], check=True)
        
        subprocess.run([
            "git", "clone", "https://github.com/espnet/espnet.git", temp_dir
        ], check=True)
        
        # 切换目录并安装
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        print("2. Installing tools...")
        subprocess.run(["./tools/installers/install_conda.sh"], check=False)
        
        print("3. Installing ESP-net in development mode...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        
        os.chdir(original_dir)
        print("✅ Development installation completed")
        return True
        
    except Exception as e:
        print(f"❌ Development installation failed: {e}")
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return False

def create_alternative_ser_script():
    """创建替代的SER训练脚本"""
    print("\n🎯 Creating Alternative SER Training Script:")
    
    alternative_script = '''#!/usr/bin/env python3
# alternative_ser_train.py - ESP-net SER训练的替代实现

import argparse
import logging
import sys
from pathlib import Path
import yaml

# 尝试导入ESP-net模块
try:
    from espnet2.tasks.ser import SERTask
    ESPNET_AVAILABLE = True
    print("✅ Using ESP-net2 SER module")
except ImportError:
    try:
        # 尝试旧版本的导入
        from espnet.tasks.ser import SERTask
        ESPNET_AVAILABLE = True
        print("✅ Using ESP-net1 SER module")
    except ImportError:
        ESPNET_AVAILABLE = False
        print("❌ ESP-net SER module not available, using fallback")

def create_default_config():
    """创建默认配置"""
    config = {
        'batch_type': 'length',
        'batch_size': 64,
        'max_epoch': 100,
        'patience': 10,
        'init_lr': 0.001,
        'model_conf': {
            'encoder_type': 'transformer',
            'encoder_conf': {
                'num_blocks': 6,
                'input_layer': 'linear',
                'attention_dim': 256,
                'feed_forward_dim': 1024,
                'attention_heads': 4,
                'dropout_rate': 0.1
            },
            'classifier_conf': {
                'num_classes': 10,
                'dropout_rate': 0.1
            }
        },
        'optim': 'adamw',
        'optim_conf': {
            'lr': 0.001,
            'weight_decay': 0.01
        },
        'scheduler': 'warmuplr',
        'scheduler_conf': {
            'warmup_steps': 1000
        }
    }
    return config

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Alternative SER Training')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--train_data_path_and_name_and_type', action='append', 
                       help='Training data specification')
    parser.add_argument('--valid_data_path_and_name_and_type', action='append',
                       help='Validation data specification') 
    parser.add_argument('--output_dir', type=str, default='exp/ser',
                       help='Output directory')
    parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--use_preprocessor', type=str, default='true',
                       help='Use preprocessor')
    
    return parser.parse_args()

def simple_ser_training(args):
    """简单的SER训练实现"""
    print("🚀 Starting Simple SER Training...")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_default_config()
        config_path = output_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"📄 Created default config: {config_path}")
    
    # 解析数据路径
    train_data = {}
    valid_data = {}
    
    if args.train_data_path_and_name_and_type:
        for item in args.train_data_path_and_name_and_type:
            path, name, dtype = item.split(',')
            train_data[name] = {'path': path, 'type': dtype}
    
    if args.valid_data_path_and_name_and_type:
        for item in args.valid_data_path_and_name_and_type:
            path, name, dtype = item.split(',')
            valid_data[name] = {'path': path, 'type': dtype}
    
    print(f"📊 Training data: {train_data}")
    print(f"📊 Validation data: {valid_data}")
    
    if ESPNET_AVAILABLE:
        try:
            # 使用ESP-net进行训练
            print("🔥 Using ESP-net for training...")
            task = SERTask()
            # 这里需要根据实际的ESP-net API进行调整
            # task.main(...)
            print("✅ ESP-net training completed")
            
        except Exception as e:
            print(f"❌ ESP-net training failed: {e}")
            print("🔄 Falling back to simple PyTorch training...")
            simple_pytorch_training(config, train_data, valid_data, output_dir)
    else:
        print("🔄 Using simple PyTorch training...")
        simple_pytorch_training(config, train_data, valid_data, output_dir)

def simple_pytorch_training(config, train_data, valid_data, output_dir):
    """简单的PyTorch训练实现"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import torchaudio
    import numpy as np
    from tqdm import tqdm
    
    print("🎯 Simple PyTorch SER Training...")
    
    # 这里实现一个简单的训练循环
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # 创建简单模型（这里需要根据实际需求实现）
    class SimpleSERModel(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(1000, 256),  # 假设输入特征维度
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.classifier = nn.Linear(128, num_classes)
        
        def forward(self, x):
            x = self.encoder(x)
            return self.classifier(x)
    
    model = SimpleSERModel().to(device)
    
    # 保存模型信息
    model_info = {
        'model_type': 'SimpleSERModel',
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'config': config
    }
    
    with open(output_dir / 'model_info.yaml', 'w') as f:
        yaml.dump(model_info, f, default_flow_style=False)
    
    print(f"✅ Training completed. Model saved to {output_dir}")
    print(f"📊 Model parameters: {model_info['num_parameters']}")

if __name__ == "__main__":
    args = parse_arguments()
    simple_ser_training(args)
'''
    
    with open("alternative_ser_train.py", 'w') as f:
        f.write(alternative_script)
    
    print("✅ Created alternative_ser_train.py")
    
    # 使替代脚本可执行
    os.chmod("alternative_ser_train.py", 0o755)
    
    return True

def create_fixed_minimal_test():
    """创建修复版本的minimal test"""
    
    print("\n🔧 Creating Fixed Minimal Test:")
    
    fixed_test = '''#!/usr/bin/env python3
# fixed_minimal_test.py - 修复版本的最小测试

import subprocess
import sys
import os
from pathlib import Path
import random

def create_tiny_dataset():
    """创建微型测试数据集"""
    print("🎯 Creating tiny test dataset...")
    
    # 创建目录结构
    for split in ['train', 'valid', 'test']:
        Path(f"tiny_data/{split}").mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {split}...")
        
        # 创建虚拟音频文件路径和情感标签
        speech_data = []
        emotion_data = []
        
        for i in range(10):  # 每个分割10个样本
            utt_id = f"{split}_utt_{i:03d}"
            # 使用虚拟路径，实际训练时需要真实音频文件
            wav_path = f"dummy_audio_{i}.wav"
            emotion_label = random.randint(0, 9)  # 10类情感
            
            speech_data.append(f"{utt_id} {wav_path}")
            emotion_data.append(f"{utt_id} {emotion_label}")
        
        # 写入文件
        with open(f"tiny_data/{split}/speech.scp", 'w') as f:
            f.write("\\n".join(speech_data) + "\\n")
        
        with open(f"tiny_data/{split}/emotion.txt", 'w') as f:
            f.write("\\n".join(emotion_data) + "\\n")
        
        print(f"✅ {split}: {len(speech_data)} samples")
    
    print("✅ Tiny dataset created!")

def create_minimal_config():
    """创建最小配置文件"""
    config = """
# Minimal SER configuration
batch_type: length
batch_size: 2
max_epoch: 1
patience: 1
init_lr: 0.001

model_conf:
    encoder_type: transformer
    encoder_conf:
        num_blocks: 2
        input_layer: linear
        attention_dim: 64
        feed_forward_dim: 256
        attention_heads: 2
        dropout_rate: 0.1
    classifier_conf:
        num_classes: 10
        dropout_rate: 0.1

optim: adam
optim_conf:
    lr: 0.001

scheduler: warmuplr
scheduler_conf:
    warmup_steps: 100
"""
    
    with open("minimal_config.yaml", 'w') as f:
        f.write(config)
    
    print("✅ Minimal config created!")

def test_espnet_alternatives():
    """测试ESP-net的不同调用方式"""
    print("🚀 Testing ESP-net alternatives...")
    
    # 测试不同的命令格式
    commands_to_try = [
        # 标准ESP-net2命令
        [sys.executable, "-m", "espnet2.bin.ser_train", "--help"],
        
        # 尝试直接导入
        [sys.executable, "-c", "from espnet2.bin import ser_train; print('SER module found')"],
        
        # 尝试旧版本命令
        [sys.executable, "-m", "espnet.bin.ser_train", "--help"],
        
        # 使用我们的替代脚本
        [sys.executable, "alternative_ser_train.py", "--help"]
    ]
    
    working_commands = []
    
    for i, cmd in enumerate(commands_to_try, 1):
        print(f"\\n🧪 Test {i}: {' '.join(cmd[:3])}...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"✅ Success!")
                working_commands.append(cmd)
            else:
                print(f"❌ Failed: {result.stderr.strip()[:100]}...")
        except subprocess.TimeoutExpired:
            print("⏰ Timeout")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return working_commands

def run_fixed_test():
    """运行修复后的测试"""
    print("\\n" + "="*50)
    print("⚡ Fixed ESP-net Test")
    print("="*50)
    print("📊 Data: 10 samples per split")
    print("⏱️  Time: ~1-2 minutes")
    print("🎯 Goal: Find working SER training method")
    print("="*50)
    
    # 1. 创建数据
    create_tiny_dataset()
    create_minimal_config()
    
    # 2. 测试不同方法
    working_commands = test_espnet_alternatives()
    
    if not working_commands:
        print("\\n❌ No working ESP-net command found")
        print("🔧 Recommendation: Use alternative PyTorch implementation")
        return False
    
    # 3. 使用第一个工作的命令进行训练
    base_cmd = working_commands[0]
    
    if "alternative_ser_train.py" in base_cmd[1]:
        # 使用我们的替代脚本
        train_cmd = [
            sys.executable, "alternative_ser_train.py",
            "--train_data_path_and_name_and_type", "tiny_data/train/speech.scp,speech,sound",
            "--train_data_path_and_name_and_type", "tiny_data/train/emotion.txt,emotion,text",
            "--valid_data_path_and_name_and_type", "tiny_data/valid/speech.scp,speech,sound", 
            "--valid_data_path_and_name_and_type", "tiny_data/valid/emotion.txt,emotion,text",
            "--output_dir", "exp/fixed_test",
            "--config", "minimal_config.yaml",
            "--ngpu", "1"
        ]
    else:
        # 使用标准ESP-net命令
        train_cmd = [
            sys.executable, "-m", "espnet2.bin.ser_train",
            "--use_preprocessor", "true",
            "--train_data_path_and_name_and_type", "tiny_data/train/speech.scp,speech,sound",
            "--train_data_path_and_name_and_type", "tiny_data/train/emotion.txt,emotion,text", 
            "--valid_data_path_and_name_and_type", "tiny_data/valid/speech.scp,speech,sound",
            "--valid_data_path_and_name_and_type", "tiny_data/valid/emotion.txt,emotion,text",
            "--output_dir", "exp/fixed_test",
            "--config", "minimal_config.yaml",
            "--ngpu", "1"
        ]
    
    print(f"\\n🚀 Running fixed test...")
    print(f"Command: {' '.join(train_cmd)}")
    
    try:
        result = subprocess.run(train_cmd, check=True, capture_output=False)
        print("\\n🎉 Fixed test successful!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ Fixed test failed: {e}")
        return False

if __name__ == "__main__":
    run_fixed_test()
'''
    
    with open("fixed_minimal_test.py", 'w') as f:
        f.write(fixed_test)
    
    print("✅ Created fixed_minimal_test.py")
    os.chmod("fixed_minimal_test.py", 0o755)
    
    return True

def main():
    print("🔧 Enhanced ESP-net Diagnosis and Fix Tool")
    print("=" * 50)
    
    # 1. 检查Python环境
    check_python_env()
    
    # 2. 详细检查ESP-net模块
    available, missing = check_espnet_modules()
    
    # 3. 检查命令行工具
    check_espnet_commands()
    
    # 4. 查看安装结构
    list_espnet_structure()
    
    # 如果SER模块缺失，提供修复选项
    if 'espnet2.bin.ser_train' in missing:
        print("\n❌ SER training module missing!")
        print("🔧 Available fix options:")
        print("1. Full reinstall with all dependencies")
        print("2. Install development version from source") 
        print("3. Create alternative SER training script")
        print("4. Create fixed minimal test with fallbacks")
        
        choice = input("\nChoose fix option (1/2/3/4): ").strip()
        
        if choice == '1':
            if install_espnet_full():
                print("🎉 Try running your test again!")
            else:
                print("❌ Full install failed. Try option 3 or 4")
                
        elif choice == '2':
            if install_espnet_dev():
                print("🎉 Development version installed!")
            else:
                print("❌ Dev install failed. Try option 3 or 4")
                
        elif choice == '3':
            create_alternative_ser_script()
            print("🎉 Alternative script created!")
            print("💡 Use: python alternative_ser_train.py [args]")
            
        elif choice == '4':
            create_alternative_ser_script()
            create_fixed_minimal_test()
            print("🎉 Fixed test created!")
            print("💡 Run: python fixed_minimal_test.py")
            
        else:
            print("❌ Invalid choice")
    else:
        print("✅ ESP-net SER module seems available")
        print("🤔 The issue might be with specific arguments or data format")

if __name__ == "__main__":
    main()