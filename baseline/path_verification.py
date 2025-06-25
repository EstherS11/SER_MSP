#!/usr/bin/env python3
# path_verification.py - 验证数据路径

import json
import os
from pathlib import Path

def verify_data_structure():
    """验证数据结构和路径"""
    
    print("🔍 Verifying SER_MSP data structure...")
    
    # 基础路径
    DATA_ROOT = "/data/user_data/esthers/SER_MSP"
    BASELINE_DIR = "/data/user_data/esthers/SER_MSP/baseline"
    AUDIO_DIR = "/data/user_data/esthers/SER_MSP/DATA/Audios"
    
    # 检查目录存在
    directories = {
        "Data Root": DATA_ROOT,
        "Baseline Dir": BASELINE_DIR, 
        "Audio Dir": AUDIO_DIR
    }
    
    for name, path in directories.items():
        if Path(path).exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} - NOT FOUND")
    
    # 检查JSON文件
    print(f"\n📄 Checking JSON files...")
    json_files = [
        "msp_train_10class.json",
        "msp_valid_10class.json", 
        "msp_test_10class.json"
    ]
    
    for json_file in json_files:
        json_path = Path(DATA_ROOT) / json_file
        if json_path.exists():
            print(f"✅ {json_file}: {json_path}")
            
            # 检查JSON内容
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                print(f"   📊 Samples: {len(data)}")
                
                # 检查第一个样本的音频路径
                first_key = next(iter(data))
                first_sample = data[first_key]
                audio_path = first_sample['wav']
                
                # 处理相对路径
                if not audio_path.startswith('/'):
                    # 相对路径，转换为绝对路径
                    if audio_path.startswith('DATA/'):
                        full_audio_path = os.path.join(DATA_ROOT, audio_path)
                    else:
                        full_audio_path = os.path.join(DATA_ROOT, "DATA", "Audios", os.path.basename(audio_path))
                else:
                    full_audio_path = audio_path
                
                if os.path.exists(full_audio_path):
                    print(f"   ✅ Audio path accessible: {audio_path} -> {full_audio_path}")
                else:
                    print(f"   ❌ Audio path missing: {audio_path} -> {full_audio_path}")
                    
                print(f"   🎭 Emotion: {first_sample['emo']}")
                
            except Exception as e:
                print(f"   ❌ Error reading JSON: {e}")
        else:
            print(f"❌ {json_file}: {json_path} - NOT FOUND")
    
    # 检查音频目录内容
    print(f"\n🎵 Checking audio directory...")
    audio_path = Path(AUDIO_DIR)
    if audio_path.exists():
        wav_files = list(audio_path.glob("*.wav"))
        print(f"✅ Found {len(wav_files)} .wav files in {AUDIO_DIR}")
        if wav_files:
            print(f"   📄 Example: {wav_files[0]}")
    else:
        print(f"❌ Audio directory not accessible: {AUDIO_DIR}")
    
    # 检查工作目录中的脚本
    print(f"\n📜 Checking scripts in baseline directory...")
    baseline_path = Path(BASELINE_DIR)
    
    required_scripts = [
        "espnet_ser_model.py",
        "register_model.py", 
        "minimal_test.py",
        "data_prep.py"
    ]
    
    for script in required_scripts:
        script_path = baseline_path / script
        if script_path.exists():
            print(f"✅ {script}")
        else:
            print(f"❌ {script} - Missing")
    
    return True

def test_json_audio_mapping():
    """测试JSON和音频文件的映射关系"""
    
    print(f"\n🔗 Testing JSON-Audio mapping...")
    
    DATA_ROOT = "/data/user_data/esthers/SER_MSP"
    json_path = Path(DATA_ROOT) / "msp_train_10class.json"
    
    if not json_path.exists():
        print(f"❌ Training JSON not found: {json_path}")
        return False
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"📊 Testing {len(data)} samples...")
    
    # 测试前10个样本
    accessible_count = 0
    missing_count = 0
    
    for i, (utt_id, info) in enumerate(data.items()):
        if i >= 10:  # 只测试前10个
            break
            
        wav_path = info['wav']
        emotion = info['emo']
        
        # 处理相对路径
        if not wav_path.startswith('/'):
            if wav_path.startswith('DATA/'):
                full_wav_path = os.path.join(DATA_ROOT, wav_path)
            else:
                full_wav_path = os.path.join(DATA_ROOT, "DATA", "Audios", os.path.basename(wav_path))
        else:
            full_wav_path = wav_path
        
        if os.path.exists(full_wav_path):
            accessible_count += 1
            print(f"✅ {utt_id}: {emotion} -> {wav_path} (found at: {full_wav_path})")
        else:
            missing_count += 1
            print(f"❌ {utt_id}: {emotion} -> {wav_path} (looked at: {full_wav_path})")
    
    print(f"\n📈 Sample test results:")
    print(f"   ✅ Accessible: {accessible_count}/10")
    print(f"   ❌ Missing: {missing_count}/10")
    
    if accessible_count > 0:
        print(f"   🎉 Audio files are accessible!")
        return True
    else:
        print(f"   ⚠️  No audio files found - check paths")
        return False

def check_current_working_directory():
    """检查当前工作目录"""
    
    print(f"\n📂 Current working directory check...")
    
    cwd = Path.cwd()
    expected_baseline = Path("/data/user_data/esthers/SER_MSP/baseline")
    
    print(f"Current: {cwd}")
    print(f"Expected: {expected_baseline}")
    
    if cwd == expected_baseline:
        print(f"✅ You're in the correct directory!")
        return True
    else:
        print(f"⚠️  You should cd to: {expected_baseline}")
        print(f"Run: cd {expected_baseline}")
        return False

def fix_json_paths_inplace():
    """在验证时直接修复JSON路径问题"""
    
    print(f"\n🔧 Fixing JSON paths in place...")
    
    DATA_ROOT = "/data/user_data/esthers/SER_MSP"
    json_files = [
        "msp_train_10class.json",
        "msp_valid_10class.json", 
        "msp_test_10class.json"
    ]
    
    for json_file in json_files:
        json_path = Path(DATA_ROOT) / json_file
        
        if not json_path.exists():
            continue
        
        print(f"📄 Fixing paths in {json_file}...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        fixed_count = 0
        for utt_id, info in data.items():
            old_path = info['wav']
            
            # 如果已经是绝对路径，跳过
            if old_path.startswith('/'):
                continue
            
            # 转换为绝对路径
            if old_path.startswith('DATA/'):
                new_path = os.path.join(DATA_ROOT, old_path)
            else:
                new_path = os.path.join(DATA_ROOT, "DATA", "Audios", os.path.basename(old_path))
            
            # 只有文件存在时才更新路径
            if os.path.exists(new_path):
                info['wav'] = new_path
                fixed_count += 1
        
        if fixed_count > 0:
            # 保存修复后的文件
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ Fixed {fixed_count} paths in {json_file}")
        else:
            print(f"⚠️  No paths fixed in {json_file}")

def main():
    print("🔍 SER_MSP Data Structure Verification")
    print("=" * 50)
    
    # 1. 验证数据结构
    verify_data_structure()
    
    # 2. 测试JSON-音频映射
    mapping_ok = test_json_audio_mapping()
    
    # 3. 如果映射有问题，尝试修复
    if not mapping_ok:
        fix_json_paths_inplace()
        print(f"\n🔄 Re-testing after path fixing...")
        mapping_ok = test_json_audio_mapping()
    
    # 4. 检查工作目录
    check_current_working_directory()
    
    print("\n" + "=" * 50)
    print("🎯 Summary:")
    if mapping_ok:
        print("✅ All paths verified! You can run:")
        print("  cd /data/user_data/esthers/SER_MSP/baseline")
        print("  python register_model.py")
        print("  python minimal_test.py")
    else:
        print("❌ Path issues remain. Please check your DATA directory.")
    print("=" * 50)

if __name__ == "__main__":
    main()