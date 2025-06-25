#!/usr/bin/env python3
"""
Script to fix paths in JSON files
Changes SER_MSP_PODCAST to SER_MSP in all paths
"""

import json
import os
import shutil
from datetime import datetime

def fix_json_paths(json_file_path, backup=True):
    """Fix paths in a single JSON file"""
    print(f"\nProcessing: {json_file_path}")
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"  File not found: {json_file_path}")
        return False
    
    # Create backup if requested
    if backup:
        backup_path = json_file_path + f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        shutil.copy2(json_file_path, backup_path)
        print(f"  Created backup: {backup_path}")
    
    # Load JSON
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Count changes
    changes_made = 0
    
    # Fix paths
    for key in data:
        old_path = data[key]['wav']
        
        # Replace SER_MSP_PODCAST with SER_MSP
        new_path = old_path.replace('SER_MSP_PODCAST', 'SER_MSP')
        
        if old_path != new_path:
            data[key]['wav'] = new_path
            changes_made += 1
    
    # Save updated JSON
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Fixed {changes_made} paths")
    
    # Verify a few paths
    print("  Verifying first 3 paths:")
    for i, key in enumerate(list(data.keys())[:3]):
        path = data[key]['wav']
        exists = os.path.exists(path)
        status = "✓ EXISTS" if exists else "✗ NOT FOUND"
        print(f"    {status}: {path}")
    
    return True

def main():
    print("=== JSON Path Fixer for MSP-PODCAST ===")
    
    # Determine data root
    if os.path.exists('/Users/esthersun/Desktop/SER/SER_MSP'):
        data_root = '/Users/esthersun/Desktop/SER/SER_MSP'
        print(f"Data root: {data_root}")
    else:
        print("Error: Data root not found at /Users/esthersun/Desktop/SER/SER_MSP")
        return
    
    # JSON files to fix
    json_files = [
        'msp_train_10class.json',
        'msp_valid_10class.json',
        'msp_test_10class.json'
    ]
    
    print("\nWill fix the following files:")
    for json_file in json_files:
        full_path = os.path.join(data_root, json_file)
        print(f"  - {full_path}")
    
    # Ask for confirmation
    response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Operation cancelled.")
        return
    
    # Fix each file
    for json_file in json_files:
        full_path = os.path.join(data_root, json_file)
        fix_json_paths(full_path, backup=True)
    
    print("\n=== Complete! ===")
    print("Original files have been backed up with .backup_timestamp extension")
    print("You can now run sanity_check.py again to verify everything works")

if __name__ == "__main__":
    main()