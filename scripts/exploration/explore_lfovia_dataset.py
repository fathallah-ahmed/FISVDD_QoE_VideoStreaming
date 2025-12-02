# explore_lfovia_dataset.py
"""
Explore the LFOVIA_QoE dataset (.mat files) and prepare it for training.

This script:
1. Loads all .mat files from the LFOVIA_QoE directory
2. Examines the structure and available features
3. Helps you understand what data is available
4. Optionally converts to CSV format for training

Usage:
    python explore_lfovia_dataset.py
"""
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path


def explore_mat_file(filepath: str):
    """
    Explore a single .mat file and print its structure.
    
    Args:
        filepath: Path to the .mat file
    """
    print(f"\n{'='*60}")
    print(f"File: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    try:
        data = loadmat(filepath)
        
        # Filter out metadata keys
        data_keys = [k for k in data.keys() if not k.startswith('__')]
        
        print(f"\nAvailable keys: {data_keys}")
        
        for key in data_keys:
            value = data[key]
            print(f"\n  {key}:")
            print(f"    Type: {type(value)}")
            
            if isinstance(value, np.ndarray):
                print(f"    Shape: {value.shape}")
                print(f"    Dtype: {value.dtype}")
                
                # Show sample values for small arrays
                if value.size < 20:
                    print(f"    Values: {value}")
                else:
                    print(f"    Sample (first 5): {value.flat[:5]}")
                    
                # Show statistics for numeric arrays
                if np.issubdtype(value.dtype, np.number):
                    print(f"    Min: {np.min(value):.4f}")
                    print(f"    Max: {np.max(value):.4f}")
                    print(f"    Mean: {np.mean(value):.4f}")
                    print(f"    Std: {np.std(value):.4f}")
        
        return data
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def explore_all_files(dataset_dir: str, max_files: int = 3):
    """
    Explore multiple .mat files to understand the dataset structure.
    
    Args:
        dataset_dir: Directory containing .mat files
        max_files: Maximum number of files to explore in detail
    """
    mat_files = sorted(Path(dataset_dir).glob("*.mat"))
    
    print(f"\n{'#'*60}")
    print(f"# LFOVIA_QoE Dataset Exploration")
    print(f"{'#'*60}")
    print(f"\nFound {len(mat_files)} .mat files")
    
    if not mat_files:
        print("No .mat files found!")
        return
    
    # Explore first few files in detail
    print(f"\nExploring first {min(max_files, len(mat_files))} files in detail...")
    
    all_data = []
    for i, filepath in enumerate(mat_files[:max_files]):
        data = explore_mat_file(str(filepath))
        if data:
            all_data.append((os.path.basename(filepath), data))
    
    # Summary across all files
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {len(mat_files)}")
    
    if all_data:
        # Check if all files have the same structure
        first_keys = set([k for k in all_data[0][1].keys() if not k.startswith('__')])
        consistent = all(
            set([k for k in data.keys() if not k.startswith('__')]) == first_keys
            for _, data in all_data
        )
        
        if consistent:
            print("✅ All explored files have consistent structure")
            print(f"Common keys: {sorted(first_keys)}")
        else:
            print("⚠️  Files have different structures")
    
    return all_data


def convert_to_csv(dataset_dir: str, output_file: str = None):
    """
    Convert .mat files to a unified CSV format.
    
    NOTE: This is a template function. You'll need to customize it
    based on the actual structure of your .mat files.
    
    Args:
        dataset_dir: Directory containing .mat files
        output_file: Output CSV path (optional)
    """
    print(f"\n{'='*60}")
    print("Converting to CSV")
    print(f"{'='*60}")
    
    mat_files = sorted(Path(dataset_dir).glob("*.mat"))
    
    # TODO: Customize this based on your data structure
    # This is a placeholder that you'll need to modify
    
    rows = []
    for filepath in mat_files:
        try:
            data = loadmat(str(filepath))
            
            # EXAMPLE: Extract features from the .mat file
            # You'll need to replace this with actual field names
            # from your dataset
            
            # Example structure (MODIFY THIS):
            # row = {
            #     'video_id': os.path.basename(filepath).replace('.mat', ''),
            #     'feature1': data['field1'][0, 0],
            #     'feature2': data['field2'][0, 0],
            #     # ... add more features
            # }
            # rows.append(row)
            
            print(f"⚠️  Please customize the convert_to_csv function")
            print(f"   based on your .mat file structure!")
            break
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    if rows:
        df = pd.DataFrame(rows)
        
        if output_file is None:
            output_file = os.path.join(dataset_dir, "LFOVIA_QoE_data.csv")
        
        df.to_csv(output_file, index=False)
        print(f"\n✅ Saved to: {output_file}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        return df
    
    return None


def main():
    # Get dataset directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "resources", "LFOVIA_QoE")
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return
    
    # Explore the dataset
    explore_all_files(dataset_dir, max_files=3)
    
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("1. Review the output above to understand your data structure")
    print("2. Customize the convert_to_csv() function based on your data")
    print("3. Update configs/lfovia_qoe_config.py with the correct features")
    print("4. Run the conversion to create CSV files for training")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
