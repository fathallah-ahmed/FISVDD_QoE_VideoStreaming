# update_model_incremental.py
"""
Update an existing FISVDD model with new data incrementally.

Usage:
    python update_model_incremental.py --dataset LIVE_NFLX_II --new-data new_samples.csv
    python update_model_incremental.py --dataset LFOVIA_QoE --new-data new_samples.csv --batch-size 50
"""
import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from fisvdd import fisvdd
from common_features import transform_df_generic
from configs import get_config


def update_model(dataset_name: str, new_data_path: str, batch_size: int = None):
    """
    Update existing FISVDD model with new data.
    
    Args:
        dataset_name: Name of the dataset
        new_data_path: Path to CSV file with new data
        batch_size: Batch size for processing (None = process all at once)
    """
    # Get dataset configuration
    config = get_config(dataset_name)
    
    # Use config default if not specified
    if batch_size is None:
        batch_size = config.INCREMENTAL_BATCH_SIZE
    
    print(f"\n{'='*60}")
    print(f"Updating FISVDD Model: {config.DATASET_NAME}")
    print(f"{'='*60}\n")
    
    # Load existing model
    artifact_path = config.get_artifact_path(config.get_model_artifact_name())
    print(f"[1/5] Loading existing model from: {artifact_path}")
    
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"Model artifact not found: {artifact_path}\n"
            f"Please train the model first using train_fisvdd.py"
        )
    
    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    sigma = artifact["sigma"]
    
    # Backward compatibility: older models may not have num_processed
    if not hasattr(model, 'num_processed'):
        model.num_processed = len(model.sv)  # Estimate based on SVs
        print(f"      Note: Model trained with older version, estimated processed samples")
    
    print(f"      Current SVs: {len(model.sv)}")
    print(f"      Total processed: {model.num_processed}")
    
    # Load new data
    print(f"\n[2/5] Loading new data from: {new_data_path}")
    
    if not os.path.exists(new_data_path):
        raise FileNotFoundError(f"New data file not found: {new_data_path}")
    
    df = pd.read_csv(new_data_path)
    print(f"      Loaded {len(df)} new samples")
    
    # Preprocess and extract features
    print(f"[3/5] Preprocessing new data...")
    dfX = transform_df_generic(
        df,
        clip_features=config.CLIP_FEATURES,
        log_features=config.LOG_TRANSFORM_FEATURES,
        invert_features=config.INVERT_FEATURES
    )
    X_raw = dfX[config.FEATURE_COLUMNS].astype(float).values
    X_new = scaler.transform(X_raw)  # Use existing scaler
    print(f"      Features: {config.FEATURE_COLUMNS}")
    print(f"      Shape: {X_new.shape}")
    
    # Update model incrementally
    print(f"\n[4/5] Updating model with new data...")
    
    if len(X_new) > batch_size:
        # Process in batches
        num_batches = int(np.ceil(len(X_new) / batch_size))
        print(f"      Processing {num_batches} batches of size {batch_size}")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_new))
            batch = X_new[start_idx:end_idx]
            
            print(f"      [Batch {i+1}/{num_batches}] Processing {len(batch)} samples...")
            stats = model.update_incremental(batch, verbose=False)
            print(f"         → SVs: {stats['final_sv_count']} "
                  f"(+{stats['sv_added']} added, -{stats['sv_removed']} removed)")
    else:
        # Process all at once
        print(f"      Processing all {len(X_new)} samples...")
        stats = model.update_incremental(X_new, verbose=True)
        print(f"      Final SVs: {stats['final_sv_count']}")
    
    # Recompute threshold on combined data
    print(f"\n[5/5] Recomputing threshold...")
    
    # Function to score batch
    def score_batch(m, Z):
        s = np.empty(len(Z))
        for i, z in enumerate(Z):
            s[i], _ = m.score_fcn(z.reshape(1, -1))
        return s
    
    # Score new data to update threshold
    new_scores = score_batch(model, X_new)
    # Combine with original threshold concept (could also rescore all training data)
    threshold = float(np.quantile(new_scores, config.THRESHOLD_QUANTILE))
    print(f"      New τ = {threshold:.6f}")
    
    # Save updated model
    artifact["model"] = model
    artifact["threshold"] = threshold
    artifact["last_update"] = {
        "num_new_samples": len(X_new),
        "new_data_source": new_data_path
    }
    
    joblib.dump(artifact, artifact_path)
    
    print(f"\n{'='*60}")
    print(f"✅ Model update complete!")
    print(f"{'='*60}")
    print(f"Model saved to: {artifact_path}")
    print(f"New samples added: {len(X_new)}")
    print(f"Total samples processed: {model.num_processed}")
    print(f"Support Vectors: {len(model.sv)}")
    print(f"Updated Threshold (τ): {threshold:.6f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Update existing FISVDD model with new data"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["LIVE_NFLX_II", "LFOVIA_QoE"],
        help="Dataset name"
    )
    parser.add_argument(
        "--new-data",
        type=str,
        required=True,
        help="Path to CSV file with new data"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for processing (default: use config value)"
    )
    
    args = parser.parse_args()
    update_model(args.dataset, args.new_data, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
