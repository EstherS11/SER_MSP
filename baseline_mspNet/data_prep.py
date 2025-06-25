#!/usr/bin/env python3
# data_prep.py - ESP-net格式数据准备

import json
import os
import numpy as np
from pathlib import Path
from collections import Counter

def prepare_msp_data(baseline_dir, output_dir="data"):
    """准备ESP-net格式的MSP-PODCAST数据"""
    
    # MSP-PODCAST情感标签映射
    emotion_map = {
        'N': 0, 'H': 1, 'S': 2, 'A': 3, 'F': 4,
        'D': 5, 'U': 6, 'C': 7, 'O': 8, 'X': 9
    }
    
    emotion_names = [
        "neutral", "happy", "sad", "angry", "fear",
        "disgust", "surprise", "contempt", "other", "unknown"
    ]
    
    # 支持两种数据路径结构
    baseline_path = Path(baseline_dir)
    
    # 尝试从两个可能的位置找JSON文件
    # 1. 在baseline_dir下 (原版)
    # 2. 直接在指定路径下 (你的情况)
    json_locations = [
        baseline_path,  # 原始位置
        Path("/data/user_data/esthers/SER_MSP"),  # 你的云端路径
        baseline_path.parent if baseline_path.name == "baseline" else baseline_path,  # 父目录
    ]
    
    # 找到JSON文件所在的目录
    json_dir = None
    for location in json_locations:
        test_file = location / "msp_train_10class.json"
        if test_file.exists():
            json_dir = location
            break
    
    if json_dir is None:
        print("❌ Could not find JSON files in any of the expected locations:")
        for location in json_locations:
            print(f"   - {location}")
        print("\nPlease check your data paths!")
        return None
    
    print(f"📁 Found JSON files in: {json_dir}")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("🔧 Preparing ESP-net format data for MSP-PODCAST...")
    
    stats = {}
    
    # 处理每个数据集
    for split in ['train', 'valid', 'test']:
        json_file = json_dir / f"msp_{split}_10class.json"
        
        print(f"\nProcessing {split} set...")
        print(f"Looking for: {json_file}")
        
        if not json_file.exists():
            print(f"❌ {json_file} not found")
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print(f"📊 Loaded {len(data)} samples from JSON")
        
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        valid_count = 0
        durations = []
        emotion_counts = {}
        missing_files = []
        
        # 创建ESP-net标准格式文件
        with open(split_dir / "speech.scp", 'w') as scp_f, \
             open(split_dir / "emotion.txt", 'w') as emo_f, \
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
                scp_f.write(f"{utt_id} {wav_path}\n")
                emo_f.write(f"{utt_id} {emotion_map[emotion]}\n")
                
                # 简单的speaker ID (从utterance ID提取)
                speaker_id = '_'.join(utt_id.split('_')[:2])
                spk_f.write(f"{utt_id} {speaker_id}\n")
                
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
            # 显示前几个缺失文件作为调试信息
            for i, missing in enumerate(missing_files[:3]):
                print(f"     Missing: {missing}")
            if len(missing_files) > 3:
                print(f"     ... and {len(missing_files) - 3} more missing files")
    
    # 保存数据集信息
    dataset_info = {
        "dataset_name": "MSP-PODCAST",
        "task": "10-class emotion recognition", 
        "num_classes": 10,
        "emotion_names": emotion_names,
        "emotion_mapping": emotion_map,
        "stats": stats,
        "json_source": str(json_dir)
    }
    
    with open(output_path / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ ESP-net data prepared in {output_dir}/")
    
    # 显示详细统计
    print("\n📊 Dataset Statistics:")
    print(f"Dataset: {dataset_info['dataset_name']}")
    print(f"Task: {dataset_info['task']}")
    print(f"Classes: {dataset_info['num_classes']}")
    print(f"JSON source: {dataset_info['json_source']}")
    
    for split in ['train', 'valid', 'test']:
        if split in stats:
            print(f"\n{split.upper()} Set:")
            print(f"  Samples: {stats[split]['count']:,}")
            print(f"  Missing files: {stats[split]['missing_files']}")
            if stats[split]['count'] > 0:
                print(f"  Duration: {stats[split]['duration_mean']:.2f}±{stats[split]['duration_std']:.2f}s")
                print(f"  Range: [{stats[split]['duration_min']:.2f}, {stats[split]['duration_max']:.2f}]s")
                
                print(f"  Emotion distribution:")
                total = stats[split]['count']
                for emotion, count in stats[split]['emotion_dist'].items():
                    pct = count / total * 100 if total > 0 else 0
                    emotion_name = emotion_names[emotion_map[emotion]]
                    print(f"    {emotion} ({emotion_name}): {count:,} ({pct:.1f}%)")
    
    # 类别平衡分析
    if 'train' in stats and stats['train']['count'] > 0:
        print(f"\n⚖️  Class Balance Analysis (Training Set):")
        train_dist = stats['train']['emotion_dist']
        counts = list(train_dist.values())
        if counts:
            max_count = max(counts)
            min_count = min(counts)
            balance_ratio = min_count / max_count if max_count > 0 else 0
            print(f"  Balance ratio: {balance_ratio:.3f} (1.0 = perfect)")
            
            if balance_ratio < 0.5:
                print(f"  ⚠️  Severe class imbalance detected!")
                print(f"     Consider using class weights or data resampling")
                
                # 建议类别权重
                print(f"  💡 Suggested class weights:")
                for emotion, count in train_dist.items():
                    if count > 0:
                        weight = max_count / count
                        emotion_name = emotion_names[emotion_map[emotion]]
                        print(f"     {emotion} ({emotion_name}): {weight:.2f}")
    
    return dataset_info

def prepare_cloud_msp_data(output_dir="data"):
    """云端版本的数据准备 - 直接使用固定路径"""
    
    # 云端固定路径
    DATA_ROOT = "/data/user_data/esthers/SER_MSP"
    
    # 检查路径是否存在
    data_root_path = Path(DATA_ROOT)
    if not data_root_path.exists():
        print(f"❌ Data root not found: {DATA_ROOT}")
        return None
    
    print(f"🌐 Using cloud data path: {DATA_ROOT}")
    return prepare_msp_data(DATA_ROOT, output_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare ESP-net format data for MSP-PODCAST')
    parser.add_argument('--baseline_dir', type=str, 
                       default="/data/user_data/esthers/SER_MSP",
                       help='Path to MSP-PODCAST baseline directory')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory for ESP-net format data')
    parser.add_argument('--cloud', action='store_true',
                       help='Use cloud-specific paths')
    
    args = parser.parse_args()
    
    if args.cloud:
        print("🌐 Using cloud mode...")
        prepare_cloud_msp_data(args.output_dir)
    else:
        prepare_msp_data(args.baseline_dir, args.output_dir)