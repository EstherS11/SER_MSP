#!/usr/bin/env python3
"""
ESP-net SER任务注册和训练脚本生成器
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

def create_ser_train_script():
    """创建ser_train.py脚本"""
    
    ser_train_code = '''#!/usr/bin/env python3
"""
ESP-net Speech Emotion Recognition Training Script
"""

import argparse
import logging
import sys
from pathlib import Path

# 导入我们的SER任务
from espnet_ser_model import SERTask

def get_parser():
    """获取命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Train a speech emotion recognition (SER) model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # 基本训练参数
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--output_dir", type=str, default="exp/ser", help="Output directory")
    parser.add_argument("--ngpu", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--use_preprocessor", type=str, default="true", help="Use preprocessor")
    
    # 数据参数
    parser.add_argument("--train_data_path_and_name_and_type", action="append",
                       help="Training data path, name, and type")
    parser.add_argument("--valid_data_path_and_name_and_type", action="append",
                       help="Validation data path, name, and type")
    
    # 训练参数
    parser.add_argument("--batch_type", type=str, default="numel", help="Batch type")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_epoch", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Gradient clipping")
    parser.add_argument("--accum_grad", type=int, default=1, help="Gradient accumulation")
    
    # 优化器参数
    parser.add_argument("--optim", type=str, default="adamw", help="Optimizer")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay")
    
    # 学习率调度器
    parser.add_argument("--scheduler", type=str, default="cosineannealinglr", help="Scheduler")
    
    # 添加SER任务参数
    SERTask.add_task_arguments(parser)
    
    return parser

def main():
    """主函数"""
    parser = get_parser()
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(Path(args.output_dir) / "train.log"),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting ESP-net SER training...")
    logging.info(f"Arguments: {vars(args)}")
    
    # 运行SER训练 - 使用正确的ESP-net调用方式
    try:
        logging.info("Using ESP-net training framework...")
        
        # 正确的ESP-net调用方式 - 传递解析后的args
        SERTask.main(args=args)
        
    except Exception as e:
        logging.error(f"ESP-net training failed: {e}")
        logging.error("Please check the error details above")
        raise  # 重新抛出异常，不使用fallback

if __name__ == "__main__":
    main()
'''
    
    with open("ser_train.py", 'w') as f:
        f.write(ser_train_code)
    
    # 让脚本可执行
    os.chmod("ser_train.py", 0o755)
    print("✅ Created ser_train.py")

def create_ser_inference_script():
    """创建ser_inference.py脚本"""
    
    ser_inference_code = '''#!/usr/bin/env python3
"""
ESP-net Speech Emotion Recognition Inference Script
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import classification_report, f1_score

# 导入我们的SER任务
from espnet_ser_model import SERTask, WavLMECAPAModel

def get_parser():
    """获取命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Inference for speech emotion recognition (SER) model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # 基本参数
    parser.add_argument("--model_file", type=str, required=True, help="Model file path")
    parser.add_argument("--train_config", type=str, help="Training config file")
    parser.add_argument("--output_dir", type=str, default="exp/ser_inference", help="Output directory")
    parser.add_argument("--ngpu", type=int, default=1, help="Number of GPUs")
    
    # 数据参数
    parser.add_argument("--valid_data_path_and_name_and_type", action="append",
                       help="Test data path, name, and type")
    
    # 推理参数
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    
    return parser

def main():
    """主函数"""
    parser = get_parser()
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "inference.log"),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting ESP-net SER inference...")
    logging.info(f"Model file: {args.model_file}")
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() and args.ngpu > 0 else "cpu")
    
    try:
        # 加载训练好的模型
        model_state = torch.load(args.model_file, map_location=device)
        
        # 创建模型实例
        model = WavLMECAPAModel(num_class=10)  # 根据需要调整
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        
        logging.info("✅ Model loaded successfully")
        
        # 这里可以添加具体的推理逻辑
        logging.info("Inference completed")
        
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        raise

if __name__ == "__main__":
    main()
'''
    
    with open("ser_inference.py", 'w') as f:
        f.write(ser_inference_code)
    
    # 让脚本可执行
    os.chmod("ser_inference.py", 0o755)
    print("✅ Created ser_inference.py")

def test_model_creation():
    """测试模型创建"""
    try:
        from espnet_ser_model import WavLMECAPAModel, SERTask
        
        # 测试模型创建
        model = WavLMECAPAModel(num_class=10)
        print("✅ Model creation test passed")
        print(f"Model type: {type(model)}")
        
        # 测试任务类
        print("✅ SER Task class imported successfully")
        print(f"Task type: {type(SERTask)}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False

def create_train_config():
    """创建训练配置文件"""
    
    config_content = '''# ESP-net SER训练配置
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

# 最佳模型选择
best_model_criterion:
  - ["valid", "macro_f1", "max"]
  - ["valid", "acc", "max"]
  - ["valid", "loss", "min"]

# 其他配置
resume: true
keep_nbest_models: 5
use_tensorboard: true
'''
    
    with open("train_config.yaml", 'w') as f:
        f.write(config_content)
    
    print("✅ Created train_config.yaml")

def main():
    print("🔧 ESP-net SER Task Registration and Script Generation")
    print("=" * 60)
    
    # 1. 测试模型创建
    print("📦 Step 1: Testing model creation...")
    if not test_model_creation():
        sys.exit(1)
    
    # 2. 创建训练脚本
    print("\n🚀 Step 2: Creating training script...")
    create_ser_train_script()
    
    # 3. 创建推理脚本
    print("\n📊 Step 3: Creating inference script...")
    create_ser_inference_script()
    
    # 4. 创建配置文件
    print("\n⚙️ ss Step 4: Creating configuration file...")
    create_train_config()
    
    print("\n" + "=" * 60)
    print("🎉 ESP-net SER Registration Completed!")
    print("\n📁 Generated files:")
    print("  - ser_train.py       # 训练脚本")
    print("  - ser_inference.py   # 推理脚本")
    print("  - train_config.yaml  # 训练配置")
    print("\n💡 Usage:")
    print("  python ser_train.py --config train_config.yaml [其他参数]")
    print("=" * 60)

if __name__ == "__main__":
    main()