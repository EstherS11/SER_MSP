import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import torchaudio
from tqdm import tqdm

def analyze_dataset(json_path, root_dir, dataset_name):
    """分析数据集的统计信息"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    emotion_map = {
        'N': 0, 'H': 1, 'S': 2, 'A': 3, 'F': 4,
        'D': 5, 'U': 6, 'C': 7, 'O': 8, 'X': 9
    }
    
    emotion_names = [
        "neutral", "happy", "sad", "angry", "fear",
        "disgust", "surprise", "contempt", "other", "unknown"
    ]
    
    # 统计情感分布
    emotions = [data[key]['emo'] for key in data]
    emotion_counts = Counter(emotions)
    
    # 统计音频长度
    lengths = [data[key]['length'] for key in data]
    
    print(f"\n{dataset_name} Dataset Statistics:")
    print(f"Total samples: {len(data)}")
    print(f"Average length: {np.mean(lengths):.2f}s")
    print(f"Min length: {np.min(lengths):.2f}s")
    print(f"Max length: {np.max(lengths):.2f}s")
    print(f"Std length: {np.std(lengths):.2f}s")
    
    print("\nEmotion distribution:")
    for emo_code, count in sorted(emotion_counts.items()):
        idx = emotion_map[emo_code]
        percentage = (count / len(data)) * 100
        print(f"{emotion_names[idx]}: {count} ({percentage:.2f}%)")
    
    return emotion_counts, lengths

def plot_dataset_statistics(train_json, valid_json, test_json, root_dir):
    """绘制数据集统计图表"""
    # 分析各个数据集
    train_emotions, train_lengths = analyze_dataset(train_json, root_dir, "Training")
    valid_emotions, valid_lengths = analyze_dataset(valid_json, root_dir, "Validation")
    test_emotions, test_lengths = analyze_dataset(test_json, root_dir, "Test")
    
    emotion_map = {
        'N': 0, 'H': 1, 'S': 2, 'A': 3, 'F': 4,
        'D': 5, 'U': 6, 'C': 7, 'O': 8, 'X': 9
    }
    
    emotion_names = [
        "neutral", "happy", "sad", "angry", "fear",
        "disgust", "surprise", "contempt", "other", "unknown"
    ]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 情感分布条形图
    ax1 = axes[0, 0]
    emotions = list(emotion_map.keys())
    train_counts = [train_emotions.get(e, 0) for e in emotions]
    valid_counts = [valid_emotions.get(e, 0) for e in emotions]
    test_counts = [test_emotions.get(e, 0) for e in emotions]
    
    x = np.arange(len(emotions))
    width = 0.25
    
    ax1.bar(x - width, train_counts, width, label='Train', alpha=0.8)
    ax1.bar(x, valid_counts, width, label='Valid', alpha=0.8)
    ax1.bar(x + width, test_counts, width, label='Test', alpha=0.8)
    
    ax1.set_xlabel('Emotion')
    ax1.set_ylabel('Count')
    ax1.set_title('Emotion Distribution Across Datasets')
    ax1.set_xticks(x)
    ax1.set_xticklabels([emotion_names[emotion_map[e]] for e in emotions], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 类别不平衡率
    ax2 = axes[0, 1]
    total_train = sum(train_counts)
    imbalance_ratios = [max(train_counts) / (count + 1) for count in train_counts]
    
    bars = ax2.bar(range(len(emotion_names)), imbalance_ratios, color='coral', alpha=0.7)
    ax2.set_xlabel('Emotion')
    ax2.set_ylabel('Imbalance Ratio')
    ax2.set_title('Class Imbalance Ratios (Max/Current)')
    ax2.set_xticks(range(len(emotion_names)))
    ax2.set_xticklabels(emotion_names, rotation=45)
    ax2.axhline(y=1, color='green', linestyle='--', label='Balanced')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    # 3. 音频长度分布
    ax3 = axes[1, 0]
    ax3.hist([train_lengths, valid_lengths, test_lengths], 
             bins=30, label=['Train', 'Valid', 'Test'], alpha=0.7)
    ax3.set_xlabel('Audio Length (seconds)')
    ax3.set_ylabel('Count')
    ax3.set_title('Audio Length Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 累积分布函数
    ax4 = axes[1, 1]
    for lengths, label in [(train_lengths, 'Train'), 
                          (valid_lengths, 'Valid'), 
                          (test_lengths, 'Test')]:
        sorted_lengths = np.sort(lengths)
        cum_dist = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
        ax4.plot(sorted_lengths, cum_dist, label=label, linewidth=2)
    
    ax4.set_xlabel('Audio Length (seconds)')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Distribution of Audio Lengths')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='10s cutoff')
    
    plt.tight_layout()
    plt.savefig('dataset_statistics.png', dpi=300)
    plt.show()

def calculate_class_weights(train_json):
    """计算用于处理类别不平衡的权重"""
    with open(train_json, 'r') as f:
        data = json.load(f)
    
    emotion_map = {
        'N': 0, 'H': 1, 'S': 2, 'A': 3, 'F': 4,
        'D': 5, 'U': 6, 'C': 7, 'O': 8, 'X': 9
    }
    
    # 统计各类别数量
    class_counts = {i: 0 for i in range(10)}
    for item in data.values():
        label = emotion_map[item['emo']]
        class_counts[label] += 1
    
    # 计算权重
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    # 方法1: 反比例权重
    weights_inverse = {}
    for cls, count in class_counts.items():
        weights_inverse[cls] = total_samples / (num_classes * count)
    
    # 方法2: 有效样本数权重
    beta = 0.9999
    weights_effective = {}
    for cls, count in class_counts.items():
        effective_num = 1.0 - np.power(beta, count)
        weights_effective[cls] = (1.0 - beta) / effective_num
    
    # 归一化权重
    sum_inverse = sum(weights_inverse.values())
    sum_effective = sum(weights_effective.values())
    
    for cls in range(num_classes):
        weights_inverse[cls] = weights_inverse[cls] * num_classes / sum_inverse
        weights_effective[cls] = weights_effective[cls] * num_classes / sum_effective
    
    print("\nClass Weights (Inverse Frequency):")
    for cls, weight in weights_inverse.items():
        print(f"Class {cls}: {weight:.4f}")
    
    print("\nClass Weights (Effective Number):")
    for cls, weight in weights_effective.items():
        print(f"Class {cls}: {weight:.4f}")
    
    return weights_inverse, weights_effective

def check_audio_quality(json_path, root_dir, num_samples=100):
    """检查音频质量和潜在问题"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    keys = list(data.keys())[:num_samples]
    
    issues = {
        'missing_files': [],
        'corrupted_files': [],
        'sample_rate_mismatch': [],
        'multi_channel': []
    }
    
    sample_rates = []
    
    print(f"Checking {num_samples} audio files...")
    for key in tqdm(keys):
        wav_path = os.path.join(root_dir, data[key]['wav'])
        
        # 检查文件是否存在
        if not os.path.exists(wav_path):
            issues['missing_files'].append(wav_path)
            continue
        
        try:
            # 尝试加载音频
            waveform, sr = torchaudio.load(wav_path)
            sample_rates.append(sr)
            
            # 检查采样率
            if sr != 16000:
                issues['sample_rate_mismatch'].append((wav_path, sr))
            
            # 检查通道数
            if waveform.shape[0] > 1:
                issues['multi_channel'].append((wav_path, waveform.shape[0]))
                
        except Exception as e:
            issues['corrupted_files'].append((wav_path, str(e)))
    
    # 报告结果
    print("\nAudio Quality Check Results:")
    print(f"Missing files: {len(issues['missing_files'])}")
    print(f"Corrupted files: {len(issues['corrupted_files'])}")
    print(f"Sample rate mismatches: {len(issues['sample_rate_mismatch'])}")
    print(f"Multi-channel files: {len(issues['multi_channel'])}")
    
    if sample_rates:
        print(f"\nSample rates found: {set(sample_rates)}")
    
    return issues

if __name__ == "__main__":
    # 设置路径
    if os.path.exists('/data/user_data/esthers/SER_MSP'):
        root_dir = '/data/user_data/esthers/SER_MSP'
    else:
        root_dir = '/Users/esthersun/Desktop/SER/SER_MSP'
    
    train_json = os.path.join(root_dir, 'msp_train_10class.json')
    valid_json = os.path.join(root_dir, 'msp_valid_10class.json')
    test_json = os.path.join(root_dir, 'msp_test_10class.json')
    
    # 绘制统计图表
    plot_dataset_statistics(train_json, valid_json, test_json, root_dir)
    
    # 计算类别权重
    weights_inverse, weights_effective = calculate_class_weights(train_json)
    
    # 检查音频质量（可选）
    # issues = check_audio_quality(train_json, root_dir, num_samples=100)