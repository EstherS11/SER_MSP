#!/usr/bin/env python3
"""
Environment and data integrity check script
Run this script before training to ensure everything is working properly
"""

import os
import sys
import json
import subprocess

def check_environment():
    """Check Python environment and required packages"""
    print("=== Environment Check ===")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.executable}")
    
    # Check required packages
    required_packages = {
        'torch': 'PyTorch',
        'torchaudio': 'TorchAudio',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'tqdm': 'tqdm',
        'matplotlib': 'Matplotlib',
        'transformers': 'Transformers (optional, for pretrained models)'
    }
    
    print("\nChecking dependencies:")
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} installed")
            
            # Special check for PyTorch
            if package == 'torch':
                import torch
                print(f"  - PyTorch version: {torch.__version__}")
                print(f"  - CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"  - CUDA version: {torch.version.cuda}")
                    print(f"  - GPU count: {torch.cuda.device_count()}")
                
            elif package == 'torchaudio':
                import torchaudio
                print(f"  - TorchAudio version: {torchaudio.__version__}")
                
        except ImportError:
            print(f"✗ {name} not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please run: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_data_structure(data_root):
    """Check data directory structure"""
    print(f"\n=== Data Structure Check ===")
    print(f"Data root directory: {data_root}")
    
    if not os.path.exists(data_root):
        print(f"✗ Data directory does not exist: {data_root}")
        return False
    
    print("✓ Data directory exists")
    
    # Check required files
    required_files = [
        'msp_train_10class.json',
        'msp_valid_10class.json',
        'msp_test_10class.json'
    ]
    
    print("\nChecking JSON files:")
    for file in required_files:
        file_path = os.path.join(data_root, file)
        if os.path.exists(file_path):
            print(f"✓ {file}")
            
            # Try to load JSON to check format
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                print(f"  - Sample count: {len(data)}")
                
                # Check first sample format
                if data:
                    first_key = list(data.keys())[0]
                    sample = data[first_key]
                    print(f"  - Sample format: {list(sample.keys())}")
                    
            except Exception as e:
                print(f"  ✗ JSON load error: {e}")
        else:
            print(f"✗ {file} does not exist")
    
    # Check audio directory
    audio_dir = os.path.join(data_root, 'DATA', 'Audios')
    if os.path.exists(audio_dir):
        print(f"\n✓ Audio directory exists: {audio_dir}")
        
        # Count audio files
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        print(f"  - WAV file count: {len(audio_files)}")
        
        if audio_files:
            print(f"  - Example file: {audio_files[0]}")
    else:
        print(f"\n✗ Audio directory does not exist: {audio_dir}")
        return False
    
    return True

def check_data_integrity(data_root, num_samples=5):
    """Check data integrity"""
    print(f"\n=== Data Integrity Check ===")
    
    train_json = os.path.join(data_root, 'msp_train_10class.json')
    if not os.path.exists(train_json):
        print("✗ Training set JSON file not found")
        return False
    
    with open(train_json, 'r') as f:
        data = json.load(f)
    
    # Check emotion label distribution
    emotion_counts = {}
    for item in data.values():
        emo = item['emo']
        emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
    
    print("\nEmotion label distribution:")
    emotion_names = {
        'N': 'neutral', 'H': 'happy', 'S': 'sad', 'A': 'angry', 
        'F': 'fear', 'D': 'disgust', 'U': 'surprise', 'C': 'contempt', 
        'O': 'other', 'X': 'unknown'
    }
    
    for emo, count in sorted(emotion_counts.items()):
        name = emotion_names.get(emo, emo)
        percentage = (count / len(data)) * 100
        print(f"  {name} ({emo}): {count} ({percentage:.1f}%)")
    
    # Randomly check a few audio files
    print(f"\nRandomly checking {num_samples} audio files:")
    keys = list(data.keys())[:num_samples]
    
    missing_files = 0
    for i, key in enumerate(keys):
        item = data[key]
        audio_path = os.path.join(data_root, item['wav'])
        
        if os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path) / 1024  # KB
            print(f"  ✓ {i+1}. {key}: {file_size:.1f} KB, label={item['emo']}")
        else:
            print(f"  ✗ {i+1}. {key}: file does not exist")
            missing_files += 1
    
    if missing_files > 0:
        print(f"\nWarning: {missing_files}/{num_samples} files missing")
    
    return True

def test_audio_loading(data_root):
    """Test audio loading functionality"""
    print(f"\n=== Audio Loading Test ===")
    
    try:
        import torch
        import torchaudio
    except ImportError:
        print("✗ Requires torch and torchaudio")
        return False
    
    train_json = os.path.join(data_root, 'msp_train_10class.json')
    with open(train_json, 'r') as f:
        data = json.load(f)
    
    # Find an audio file to test
    key = list(data.keys())[0]
    item = data[key]
    audio_path = os.path.join(data_root, item['wav'])
    
    print(f"Testing loading: {audio_path}")
    
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        print(f"✓ Audio loaded successfully")
        print(f"  - Sample rate: {sample_rate} Hz")
        print(f"  - Waveform shape: {waveform.shape}")
        print(f"  - Duration: {waveform.shape[1]/sample_rate:.2f} seconds")
        print(f"  - Value range: [{waveform.min():.3f}, {waveform.max():.3f}]")
        
        # Test resampling
        if sample_rate != 16000:
            print(f"\nTesting resampling to 16000Hz...")
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            resampled = resampler(waveform)
            print(f"✓ Resampling successful, new shape: {resampled.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Audio loading failed: {e}")
        return False

def create_minimal_requirements():
    """Create minimal requirements.txt"""
    print("\n=== Creating Minimal Requirements File ===")
    
    requirements = """# Core dependencies
torch>=1.10.0
torchaudio>=0.10.0
numpy>=1.19.0
scikit-learn>=0.24.0
tqdm>=4.62.0
matplotlib>=3.3.0

# Optional dependencies (required for pretrained models)
# transformers>=4.20.0
# speechbrain>=0.5.0
"""
    
    with open('minimal_requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✓ Created minimal_requirements.txt")
    print("  Run: pip install -r minimal_requirements.txt")

def main():
    """Main function"""
    print("MSP-PODCAST Speech Emotion Recognition - Environment Check\n")
    
    # Check data path
    if os.path.exists('/Users/esthersun/Desktop/SER/SER_MSP'):
        data_root = '/Users/esthersun/Desktop/SER/SER_MSP'
    elif os.path.exists('/data/user_data/esthers/SER_MSP'):
        data_root = '/data/user_data/esthers/SER_MSP'
    else:
        print("✗ Data directory not found")
        print("  Please ensure data is in one of these locations:")
        print("  - /Users/esthersun/Desktop/SER/SER_MSP (local)")
        print("  - /data/user_data/esthers/SER_MSP (cloud)")
        return
    
    # Run checks
    checks = [
        ("Environment check", check_environment),
        ("Data structure check", lambda: check_data_structure(data_root)),
        ("Data integrity check", lambda: check_data_integrity(data_root)),
        ("Audio loading test", lambda: test_audio_loading(data_root))
    ]
    
    all_passed = True
    for name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"\nError during {name}: {e}")
            all_passed = False
    
    # Summary
    print("\n" + "="*50)
    if all_passed:
        print("✓ All checks passed! Ready to start training.")
        print("\nNext steps:")
        print("1. Run lightweight test: python minimal_test.py")
        print("2. Run full training: python train.py")
    else:
        print("✗ Some checks failed, please fix the issues above.")
        create_minimal_requirements()

if __name__ == "__main__":
    main()