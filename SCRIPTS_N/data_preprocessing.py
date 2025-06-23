"""
Updated MSP-PODCAST dataset processor for 10-class emotion recognition
Supports all MSP-PODCAST emotion classes with detailed class distribution analysis
"""

import os
import json
import random
from speechbrain.dataio.dataio import read_audio
from collections import Counter
import numpy as np

SAMPLERATE = 16000

def create_json(wav_list, json_file, output_format='minimal'):
    """
    Create JSON manifest file with flexible output format
    
    Args:
        wav_list: List of audio data entries
        json_file: Output JSON file path
        output_format: 'minimal' (IEMOCAP style) or 'extended' (full metadata)
    """
    json_dict = {}
    
    for obj in wav_list:
        wav_file = obj[0]
        emo = obj[1]
        
        # Read audio to get duration
        try:
            signal = read_audio(wav_file)
            duration = signal.shape[0] / SAMPLERATE
        except:
            print(f"Warning: Cannot read audio file {wav_file}")
            continue
            
        uttid = os.path.basename(wav_file)[:-4]  # Remove .wav extension
        
        # Basic fields (minimal format - IEMOCAP compatible)
        entry = {
            "wav": wav_file,
            "length": duration,
            "emo": emo
        }
        
        # Extended fields (full MSP-PODCAST metadata)
        if output_format == 'extended' and len(obj) > 2:
            entry.update({
                "gender": obj[2] if len(obj) > 2 else 'Unknown',
                "spkr_id": obj[3] if len(obj) > 3 else 'Unknown',
                "emo_act": obj[4] if len(obj) > 4 else 0.0,
                "emo_dom": obj[5] if len(obj) > 5 else 0.0,
                "emo_val": obj[6] if len(obj) > 6 else 0.0
            })
        
        json_dict[uttid] = entry
    
    # Write to file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)
    
    print(f"{json_file} created successfully! Format: {output_format}")


def analyze_class_distribution(labels_data, emotion_mapping):
    """
    Analyze and report class distribution for MSP-PODCAST 10-class setup
    """
    print("\n📊 MSP-PODCAST 10-Class Emotion Distribution Analysis")
    print("=" * 60)
    
    # Count emotions
    emotion_counts = Counter()
    split_counts = {'Train': Counter(), 'Development': Counter(), 'Test1': Counter(), 'Test2': Counter()}
    
    total_samples = 0
    for audio_file, info in labels_data.items():
        emo_class = info['EmoClass']
        split_set = info.get('Split_Set', 'Unknown')
        
        if emo_class in emotion_mapping:
            mapped_emo = emotion_mapping[emo_class]
            emotion_counts[mapped_emo] += 1
            
            if split_set in split_counts:
                split_counts[split_set][mapped_emo] += 1
            
            total_samples += 1
    
    # Overall distribution
    print(f"Total samples: {total_samples}")
    print(f"Number of classes: {len(emotion_counts)}")
    print("\n=== Overall Emotion Distribution ===")
    
    emotion_info = []
    for emo in sorted(emotion_counts.keys()):
        count = emotion_counts[emo]
        percentage = (count / total_samples) * 100
        emotion_info.append((emo, count, percentage))
        print(f"{emo:2s}: {count:6d} samples ({percentage:5.1f}%)")
    
    # Calculate class imbalance
    counts = [info[1] for info in emotion_info]
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count
    
    print(f"\nClass imbalance ratio: {imbalance_ratio:.1f}:1")
    print(f"Most frequent class: {max(emotion_counts, key=emotion_counts.get)} ({max_count} samples)")
    print(f"Least frequent class: {min(emotion_counts, key=emotion_counts.get)} ({min_count} samples)")
    
    # Split-wise distribution
    print(f"\n=== Split-wise Distribution ===")
    for split_name, split_counter in split_counts.items():
        if split_counter:
            total_split = sum(split_counter.values())
            print(f"\n{split_name} ({total_split} samples):")
            for emo in sorted(split_counter.keys()):
                count = split_counter[emo]
                percentage = (count / total_split) * 100
                print(f"  {emo:2s}: {count:5d} ({percentage:4.1f}%)")
    
    # Calculate class weights for imbalanced learning
    print(f"\n=== Recommended Class Weights ===")
    print("For handling class imbalance in training:")
    
    # Inverse frequency weighting
    inv_weights = []
    for emo in sorted(emotion_counts.keys()):
        weight = total_samples / (len(emotion_counts) * emotion_counts[emo])
        inv_weights.append(weight)
        print(f"{emo:2s}: {weight:.2f}")
    
    print(f"\nClass weights list: {inv_weights}")
    
    return emotion_counts, split_counts


def split_by_original_sets(labels_data, audio_path, emotion_mapping, test_set_choice='both'):
    """
    Use MSP-PODCAST original Train/Dev/Test split with 10-class emotions
    
    Args:
        labels_data: Loaded labels_consensus.json data
        audio_path: Path to audio files
        emotion_mapping: Dict mapping MSP emotions to standard labels
        test_set_choice: 'Test1', 'Test2', or 'both'
    
    Returns:
        dict: Contains 'train', 'valid', 'test' splits
    """
    data_split = {"train": [], "valid": [], "test": []}
    
    # Track statistics
    processed_samples = 0
    skipped_samples = 0
    missing_files = 0
    
    for audio_file, info in labels_data.items():
        # Check if emotion label is in mapping (10-class filtering)
        if info['EmoClass'] not in emotion_mapping:
            skipped_samples += 1
            continue
            
        # Build full audio path
        full_audio_path = os.path.join(audio_path, audio_file)
        if not os.path.exists(full_audio_path):
            missing_files += 1
            continue
            
        # Convert emotion label to 10-class system
        mapped_emotion = emotion_mapping[info['EmoClass']]
        
        # Build data entry with all available metadata
        data_entry = [
            full_audio_path,
            mapped_emotion,
            info.get('Gender', 'Unknown'),
            info.get('SpkrID', 'Unknown'),
            info.get('EmoAct', 0.0),
            info.get('EmoDom', 0.0),
            info.get('EmoVal', 0.0)
        ]
        
        # Assign to different sets based on original split
        split_set = info.get('Split_Set', '')
        
        if split_set == 'Train':
            data_split["train"].append(data_entry)
        elif split_set == 'Development':
            data_split["valid"].append(data_entry)
        elif split_set == 'Test1' and test_set_choice in ['Test1', 'both']:
            data_split["test"].append(data_entry)
        elif split_set == 'Test2' and test_set_choice in ['Test2', 'both']:
            data_split["test"].append(data_entry)
        
        processed_samples += 1
    
    print(f"\n📈 Data Processing Summary:")
    print(f"Total processed samples: {processed_samples}")
    print(f"Skipped samples (unsupported emotion): {skipped_samples}")
    print(f"Missing audio files: {missing_files}")
    
    return data_split


def prepare_msp_podcast_10class(
    data_original,
    labels_file,
    save_json_train,
    save_json_valid,
    save_json_test,
    emotion_mapping=None,
    use_original_split=True,
    test_set_choice='both',
    output_format='minimal',
    seed=12
):
    """
    Prepare MSP-PODCAST dataset for 10-class emotion recognition
    
    Args:
        data_original: Path to MSP-PODCAST audio folder
        labels_file: Path to labels_consensus.json
        save_json_train: Path to save training set JSON
        save_json_valid: Path to save validation set JSON  
        save_json_test: Path to save test set JSON
        emotion_mapping: Dict mapping MSP emotions to standard labels
        use_original_split: Use original Train/Dev/Test split if True
        test_set_choice: 'Test1', 'Test2', or 'both'
        output_format: 'minimal' (IEMOCAP style) or 'extended' (full metadata)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # MSP-PODCAST 10-class emotion mapping
    if emotion_mapping is None:
        emotion_mapping = {
            'N': 'N',    # Neutral
            'H': 'H',    # Happy
            'X': 'X',    # Excited (new class)
            'A': 'A',    # Angry
            'S': 'S',    # Sad
            'U': 'U',    # Surprised (new class)
            'C': 'C',    # Contempt (new class)
            'O': 'O',    # Other (new class)
            'D': 'D',    # Disgusted (new class)
            'F': 'F',    # Fear (new class)
        }
    
    print(f"🎯 MSP-PODCAST 10-Class Emotion Recognition Setup")
    print(f"Emotion classes: {list(emotion_mapping.keys())}")
    print(f"Mapped to: {list(emotion_mapping.values())}")
    
    # Check if already completed
    if all(os.path.exists(f) for f in [save_json_train, save_json_valid, save_json_test]):
        print("Data preparation already completed, skipping...")
        return
    
    # Load labels file
    with open(labels_file, 'r') as f:
        labels_data = json.load(f)
    
    print(f"Loaded {len(labels_data)} samples from {labels_file}")
    
    # Analyze class distribution
    emotion_counts, split_counts = analyze_class_distribution(labels_data, emotion_mapping)
    
    # Organize data by split strategy
    if use_original_split:
        data_split = split_by_original_sets(labels_data, data_original, emotion_mapping, test_set_choice)
        print(f"\n✅ Using MSP-PODCAST original split with test_set_choice='{test_set_choice}'")
    else:
        # Could implement speaker-independent split here if needed
        print("❌ Speaker-independent split not implemented for 10-class setup")
        print("Using original split instead...")
        data_split = split_by_original_sets(labels_data, data_original, emotion_mapping, test_set_choice)
    
    # Create JSON files
    create_json(data_split["train"], save_json_train, output_format)
    create_json(data_split["valid"], save_json_valid, output_format)
    create_json(data_split["test"], save_json_test, output_format)
    
    print(f"\n🎉 MSP-PODCAST 10-Class Data Preparation Completed!")
    print(f"Train set: {len(data_split['train'])} samples")
    print(f"Valid set: {len(data_split['valid'])} samples")
    print(f"Test set: {len(data_split['test'])} samples")
    
    # Final emotion distribution in splits
    print(f"\n=== Final Split Emotion Distribution ===")
    for split_name, split_data in data_split.items():
        emotion_count = Counter(entry[1] for entry in split_data)
        print(f"\n{split_name.upper()} SET:")
        for emo in sorted(emotion_count.keys()):
            count = emotion_count[emo]
            percentage = (count / len(split_data)) * 100
            print(f"  {emo}: {count:5d} ({percentage:5.1f}%)")


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    
    # MSP-PODCAST 10-class emotion mapping
    emotion_mapping = {
        'N': 'N',    # Neutral
        'H': 'H',    # Happy  
        'X': 'X',    # Excited
        'A': 'A',    # Angry
        'S': 'S',    # Sad
        'U': 'U',    # Surprised
        'C': 'C',    # Contempt
        'O': 'O',    # Other
        'D': 'D',    # Disgusted
        'F': 'F',    # Fear
    }
    
    print("🎯 MSP-PODCAST 10-Class Data Preparation")
    
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'DATA'))
    
    # Prepare 10-class dataset
    prepare_msp_podcast_10class(
        data_original=os.path.join(DATA_DIR, 'Audios'),
        labels_file=os.path.join(DATA_DIR, 'Labels', 'labels_consensus.json'),
        save_json_train='msp_train_10class.json',
        save_json_valid='msp_valid_10class.json',
        save_json_test='msp_test_10class.json',
        emotion_mapping=emotion_mapping,
        use_original_split=True,
        test_set_choice='both',
        output_format='minimal',  # IEMOCAP compatible
        seed=12
    )
    
    print("\n✅ Ready for 10-class emotion recognition training!")
    print("Usage: python train_discrete_SSL.py hparams/train_discrete_SSL.yaml")