# quick_explore_lfovia.py
"""
Quick exploration of LFOVIA_QoE .mat files (minimal dependencies).
"""
import os
from scipy.io import loadmat
from pathlib import Path


def explore_mat_file(filepath):
    """Explore a single .mat file."""
    print(f"\n{'='*70}")
    print(f"File: {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    data = loadmat(filepath)
    
    # Filter out metadata keys
    data_keys = [k for k in data.keys() if not k.startswith('__')]
    
    print(f"\nAvailable keys: {data_keys}")
    
    for key in data_keys:
        value = data[key]
        print(f"\n  {key}:")
        print(f"    Type: {type(value).__name__}")
        
        if hasattr(value, 'shape'):
            print(f"    Shape: {value.shape}")
            print(f"    Dtype: {value.dtype}")
            
            # Show sample
            if value.size < 20:
                print(f"    Values: {value}")
            else:
                print(f"    Sample: {value.flat[:5]}")
    
    return data


# Main execution
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "resources", "LFOVIA_QoE")
output_file = os.path.join(base_dir, "lfovia_structure.txt")

with open(output_file, 'w') as f:
    mat_files = sorted(Path(dataset_dir).glob("*.mat"))
    
    f.write(f"{'#'*70}\n")
    f.write(f"# LFOVIA_QoE Dataset Exploration\n")
    f.write(f"{'#'*70}\n")
    f.write(f"\nFound {len(mat_files)} .mat files\n")

    # Explore first 3 files
    for filepath in mat_files[:3]:
        f.write(f"\n{'='*70}\n")
        f.write(f"File: {os.path.basename(filepath)}\n")
        f.write(f"{'='*70}\n")
        
        try:
            data = loadmat(str(filepath))
            
            # Filter out metadata keys
            data_keys = [k for k in data.keys() if not k.startswith('__')]
            
            f.write(f"\nAvailable keys: {data_keys}\n")
            
            for key in data_keys:
                value = data[key]
                f.write(f"\n  {key}:\n")
                f.write(f"    Type: {type(value).__name__}\n")
                
                if hasattr(value, 'shape'):
                    f.write(f"    Shape: {value.shape}\n")
                    f.write(f"    Dtype: {value.dtype}\n")
                    
                    # Show sample
                    if value.size < 20:
                        f.write(f"    Values: {value}\n")
                    else:
                        f.write(f"    Sample: {value.flat[:5]}\n")
        except Exception as e:
            f.write(f"Error reading file: {e}\n")

    f.write(f"\n{'='*70}\n")
    f.write("NEXT STEPS:\n")
    f.write("1. Review the data structure above\n")
    f.write("2. Update configs/lfovia_qoe_config.py with feature names\n")
    f.write("3. Create a conversion script to generate CSV files\n")
    f.write(f"{'='*70}\n")

print(f"Exploration saved to: {output_file}")
