#!/usr/bin/env python3
"""
Quick script to verify transformer_explainer.py has been updated
"""

import os
import sys

# Path to your repo
REPO_PATH = r"C:\Users\Divyesh Jayswal\Desktop\BPM_Research_app_frontend"
FILE_PATH = os.path.join(REPO_PATH, "explainability", "transformers", "transformer_explainer.py")

def check_file_updated():
    print("="*70)
    print("CHECKING IF FILE WAS UPDATED")
    print("="*70)
    
    if not os.path.exists(FILE_PATH):
        print(f"❌ FILE NOT FOUND: {FILE_PATH}")
        return False
    
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for debug markers
    checks = {
        "[DEBUG explain_samples]": "explain_samples debug output",
        "[LIME] Plotting": "LIME plotting debug output",
        "[DEBUG] Extracted": "Diverse sample extraction debug",
        "original_idx": "Original index parameter",
        "plots_saved = 0": "Plot counter"
    }
    
    print(f"\nFile: {FILE_PATH}")
    print(f"Size: {len(content):,} bytes\n")
    
    all_good = True
    for marker, description in checks.items():
        if marker in content:
            print(f"✅ FOUND: {description}")
        else:
            print(f"❌ MISSING: {description}")
            all_good = False
    
    print("\n" + "="*70)
    if all_good:
        print("✅ FILE IS UPDATED - You have the latest version")
        print("   If still seeing 1 plot, check console output")
    else:
        print("❌ FILE NOT UPDATED - Please replace the file:")
        print("   1. Delete transformer_explainer.py")
        print("   2. Copy transformer_explainer_FIXED.py")
        print("   3. Rename to transformer_explainer.py")
        print("   4. Run this script again")
    print("="*70)
    
    return all_good

if __name__ == "__main__":
    check_file_updated()