#!/usr/bin/env python3
"""
Quick script to validate all audio files in the dataset.
This script checks for missing, empty, and corrupted audio files
and provides an option to automatically clean the JSON manifest files.
"""
import json
import os
import torch
import torchaudio
from tqdm import tqdm
import sys
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def validate_dataset(json_file, data_root):
    """Validate all audio files referenced in a JSON manifest."""
    logging.info(f"Validating {json_file}...")
    
    if not os.path.exists(json_file):
        logging.error(f"JSON file not found: {json_file}")
        return [], [], []

    # Load JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    problematic_files = []
    empty_files = []
    missing_files = []
    
    # Use tqdm for a nice progress bar
    for key, item in tqdm(data.items(), desc=f"Checking {os.path.basename(json_file)}"):
        # The path stored in the JSON file
        relative_wav_path = item['wav']
        # The full, absolute path to the audio file
        full_wav_path = relative_wav_path.replace('{data_root}', data_root)
        
        # 1. Check if file exists
        if not os.path.exists(full_wav_path):
            missing_files.append(full_wav_path)
            continue
        
        # 2. Try to load audio to check for corruption or emptiness
        try:
            waveform, sample_rate = torchaudio.load(full_wav_path)
            
            # 3. Check if audio is empty (no samples)
            if waveform.numel() == 0:
                empty_files.append(full_wav_path)
            
        except Exception as e:
            # This catches corrupted files that torchaudio cannot load
            problematic_files.append((full_wav_path, str(e)))
    
    # Report results
    print("-" * 50)
    logging.info(f"Validation Results for {json_file}:")
    print(f"  - Total entries in JSON: {len(data)}")
    print(f"  - Missing files: {len(missing_files)}")
    print(f"  - Empty audio files: {len(empty_files)}")
    print(f"  - Corrupted/Unreadable files: {len(problematic_files)}")
    print("-" * 50)
    
    # Display details if any bad files were found
    if missing_files:
        logging.warning("Found Missing files (first 10):")
        for f in missing_files[:10]:
            print(f"  - {f}")
    
    if empty_files:
        logging.warning("Found Empty files (first 10):")
        for f in empty_files[:10]:
            print(f"  - {f}")
    
    if problematic_files:
        logging.warning("Found Problematic files (first 10):")
        for f, err in problematic_files[:10]:
            print(f"  - {f}: {err}")
    
    return missing_files, empty_files, problematic_files

def fix_json_file(json_file, data_root, files_to_remove):
    """Remove problematic entries from a JSON file."""
    if not files_to_remove:
        logging.info(f"No files to remove from {json_file}.")
        return

    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Create a backup for safety
    backup_file = json_file.replace('.json', '_backup.json')
    with open(backup_file, 'w') as f:
        json.dump(data, f, indent=4)
    logging.info(f"Created backup: {backup_file}")
    
    # Create a set of bad files for efficient lookup
    files_to_remove_set = set(files_to_remove)
    
    initial_count = len(data)
    keys_to_remove = []
    
    # Find the keys corresponding to the bad file paths
    for key, item in data.items():
        full_wav_path = item['wav'].replace('{data_root}', data_root)
        if full_wav_path in files_to_remove_set:
            keys_to_remove.append(key)
    
    # Remove the entries
    for key in keys_to_remove:
        del data[key]
    
    # Save the cleaned version
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    logging.info(f"Removed {len(keys_to_remove)} entries from {json_file}")
    print(f"Remaining entries: {len(data)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_data.py /path/to/your/msp_podcast_data/folder")
        print("Example: python validate_data.py /datasets/MSP-PODCAST")
        sys.exit(1)
    
    data_folder = os.path.abspath(sys.argv[1])
    
    if not os.path.isdir(data_folder):
        logging.error(f"Provided data folder does not exist: {data_folder}")
        sys.exit(1)
        
    print(f"\nStarting validation for data in: {data_folder}")

    # Process all standard splits
    for split in ['train', 'valid', 'test']:
        json_filename = f'msp_{split}_10class.json'
        # Assume json is in the current directory, but you can change this path
        json_filepath = os.path.join(os.getcwd(), json_filename)
        
        missing, empty, problematic = validate_dataset(json_filepath, data_folder)
        
        # Combine all problematic file paths into a single list
        all_bad_files = missing + empty + [f[0] for f in problematic]
        
        if all_bad_files:
            try:
                response = input(f"\nDo you want to remove {len(all_bad_files)} bad entries from {json_filename}? (y/n): ")
                if response.strip().lower() == 'y':
                    fix_json_file(json_filepath, data_folder, all_bad_files)
            except (KeyboardInterrupt, EOFError):
                 print("\nOperation cancelled by user.")

    print("\nValidation process completed.")
