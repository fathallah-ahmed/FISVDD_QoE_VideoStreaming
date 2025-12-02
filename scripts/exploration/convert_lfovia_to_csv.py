# convert_lfovia_to_csv.py
"""
Convert LFOVIA_QoE .mat files to CSV format for FISVDD training.
"""
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
FEATURES = ["NIQE", "PSNR", "SSIM", "STRRED", "TSL", "Nrebuffers"]
TARGET = "score_continuous"
ALL_COLS = FEATURES + [TARGET]

def load_and_flatten(mat_path):
    """Load a .mat file and flatten selected features into a DataFrame."""
    try:
        data = loadmat(mat_path)
        filename = os.path.basename(mat_path)
        
        # Extract arrays
        arrays = {}
        min_len = float('inf')
        
        for col in ALL_COLS:
            if col not in data:
                print(f"⚠️  Missing column {col} in {filename}")
                return None
            
            arr = data[col]
            # Handle (1, N) shape
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr.flatten()
            
            arrays[col] = arr
            min_len = min(min_len, len(arr))
        
        # Truncate to minimum length (some features might be longer)
        # Usually features are aligned, but scores might be different
        df_data = {}
        for col in ALL_COLS:
            df_data[col] = arrays[col][:min_len]
            
        df = pd.DataFrame(df_data)
        df['file'] = filename
        df['content'] = filename.replace('.mat', '') # Use filename as content ID
        
        return df
        
    except Exception as e:
        print(f"❌ Error processing {mat_path}: {e}")
        return None

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "resources", "LFOVIA_QoE")
    
    print(f"Scanning {dataset_dir}...")
    mat_files = sorted(Path(dataset_dir).glob("*.mat"))
    
    if not mat_files:
        print("❌ No .mat files found!")
        return

    all_dfs = []
    for f in mat_files:
        df = load_and_flatten(str(f))
        if df is not None:
            all_dfs.append(df)
            
    if not all_dfs:
        print("❌ No data loaded.")
        return
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded {len(full_df)} total samples from {len(all_dfs)} files.")
    
    # Split by video file (to avoid data leakage)
    unique_files = full_df['file'].unique()
    train_files, test_files = train_test_split(unique_files, test_size=0.2, random_state=42)
    
    train_df = full_df[full_df['file'].isin(train_files)]
    test_df = full_df[full_df['file'].isin(test_files)]
    
    # Save CSVs
    train_path = os.path.join(dataset_dir, "LFOVIA_QoE_train.csv")
    test_path = os.path.join(dataset_dir, "LFOVIA_QoE_test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✅ Created Train set: {train_path} ({len(train_df)} rows)")
    print(f"✅ Created Test set:  {test_path} ({len(test_df)} rows)")
    print("\nReady for training!")

if __name__ == "__main__":
    main()
