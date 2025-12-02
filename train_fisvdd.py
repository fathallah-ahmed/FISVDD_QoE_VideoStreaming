# train_fisvdd.py
"""
Train FISVDD model on a specified dataset.

Usage:
    python train_fisvdd.py --dataset LIVE_NFLX_II
    python train_fisvdd.py --dataset LFOVIA_QoE
"""
import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances

from fisvdd import fisvdd
from common_features import transform_df_generic
from configs import get_config


def train_model(dataset_name: str, batch_size: int = None, use_batches: bool = False):
    """
    Train FISVDD model for the specified dataset.
    
    Args:
        dataset_name: Name of the dataset to train on
        batch_size: Batch size for incremental learning (None = use config default)
        use_batches: If True, use batch-based incremental learning
    """
    # Get dataset configuration
    config = get_config(dataset_name)
    
    # Use config default if not specified
    if batch_size is None:
        batch_size = config.INITIAL_BATCH_SIZE
    
    print(f"\n{'='*60}")
    print(f"Training FISVDD on {config.DATASET_NAME}")
    print(f"Description: {config.DATASET_DESCRIPTION}")
    print(f"Mode: {'Batch Incremental Learning' if use_batches else 'Standard Training'}")
    if use_batches:
        print(f"Batch Size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Load training data
    train_path = config.get_resource_path(config.TRAIN_FILE)
    print(f"[1/6] Loading training data from: {train_path}")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training file not found: {train_path}\n"
            f"Please ensure the data is in the correct location."
        )
    
    df = pd.read_csv(train_path)
    print(f"      Loaded {len(df)} samples")
    
    # Preprocess and extract features
    print(f"[2/6] Preprocessing data...")
    dfX = transform_df_generic(
        df,
        clip_features=config.CLIP_FEATURES,
        log_features=config.LOG_TRANSFORM_FEATURES,
        invert_features=config.INVERT_FEATURES
    )
    X_raw = dfX[config.FEATURE_COLUMNS].astype(float).values
    print(f"      Features: {config.FEATURE_COLUMNS}")
    print(f"      Shape: {X_raw.shape}")
    
    # Scale features
    print(f"[3/6] Scaling features...")
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)
    
    # Compute sigma
    print(f"[4/6] Computing sigma using {config.SIGMA_METHOD}...")
    if config.SIGMA_METHOD == "median_heuristic":
        # Median distance heuristic
        sub = X[np.random.choice(len(X), size=min(2000, len(X)), replace=False)]
        D = pairwise_distances(sub, sub, metric="euclidean")
        sigma = float(max(1e-6, np.median(D[D > 0]) / np.sqrt(2.0)))
    elif config.SIGMA_METHOD == "fixed":
        sigma = config.SIGMA_VALUE
    else:
        raise ValueError(f"Unknown sigma method: {config.SIGMA_METHOD}")
    
    print(f"      σ = {sigma:.6f}")
    
    # Train FISVDD
    print(f"[5/6] Training FISVDD model...")
    
    if use_batches:
        # Batch-based incremental learning
        num_batches = int(np.ceil(len(X) / batch_size))
        print(f"      Training with {num_batches} batches of size {batch_size}")
        
        # Initialize with first batch
        first_batch = X[:batch_size]
        print(f"      [Batch 1/{num_batches}] Initializing with {len(first_batch)} samples...")
        model = fisvdd(first_batch, sigma, initial_batch_only=True)
        model.find_sv(first_batch)
        print(f"         → SVs: {len(model.sv)}")
        
        # Process remaining batches
        for i in range(1, num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X))
            batch = X[start_idx:end_idx]
            
            print(f"      [Batch {i+1}/{num_batches}] Processing {len(batch)} samples...")
            stats = model.update_incremental(batch, verbose=False)
            print(f"         → SVs: {stats['final_sv_count']} "
                  f"(+{stats['sv_added']} added, -{stats['sv_removed']} removed)")
            
            # Checkpoint if enabled
            if config.ENABLE_BATCH_CHECKPOINTS and (i + 1) % config.CHECKPOINT_EVERY_N_BATCHES == 0:
                checkpoint_path = config.get_artifact_path(
                    f"{config.DATASET_NAME}_checkpoint_batch_{i+1}.joblib"
                )
                print(f"         → Saving checkpoint: {checkpoint_path}")
                joblib.dump(model.get_state(), checkpoint_path)
        
        print(f"      Total samples processed: {model.num_processed}")
    else:
        # Standard training (all data at once)
        model = fisvdd(X, sigma)
        model.find_sv()
    
    print(f"      Support vectors: {len(model.sv)}")
    
    # Compute threshold
    print(f"[6/6] Computing threshold at {config.THRESHOLD_QUANTILE*100}th percentile...")
    
    def score_batch(m, Z):
        s = np.empty(len(Z))
        for i, z in enumerate(Z):
            s[i], _ = m.score_fcn(z.reshape(1, -1))
        return s
    
    train_scores = score_batch(model, X)
    threshold = float(np.quantile(train_scores, config.THRESHOLD_QUANTILE))
    print(f"      τ = {threshold:.6f}")
    
    # Save artifacts
    artifact_path = config.get_artifact_path(config.get_model_artifact_name())
    
    # Ensure artifacts directory exists
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    
    joblib.dump({
        "dataset_name": config.DATASET_NAME,
        "features": config.FEATURE_COLUMNS,
        "scaler": scaler,
        "sigma": sigma,
        "threshold": threshold,
        "model": model,
        "config": {
            "clip_features": config.CLIP_FEATURES,
            "log_features": config.LOG_TRANSFORM_FEATURES,
            "threshold_quantile": config.THRESHOLD_QUANTILE,
        },
        "training_mode": "batch" if use_batches else "standard",
        "batch_size": batch_size if use_batches else None,
    }, artifact_path)
    
    print(f"\n{'='*60}")
    print(f"✅ Training complete!")
    print(f"{'='*60}")
    print(f"Model saved to: {artifact_path}")
    print(f"Samples: {len(X)}")
    print(f"Features: {len(config.FEATURE_COLUMNS)}")
    print(f"Support Vectors: {len(model.sv)}")
    print(f"Sigma (σ): {sigma:.6f}")
    print(f"Threshold (τ): {threshold:.6f}")
    if use_batches:
        print(f"Training Mode: Batch Incremental (batch_size={batch_size})")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train FISVDD model on a specified dataset (uses incremental batch learning by default)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["LIVE_NFLX_II", "LFOVIA_QoE"],
        help="Dataset to train on"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for incremental learning (default: use config value)"
    )
    parser.add_argument(
        "--standard-mode",
        action="store_true",
        help="Use standard training (all data at once) instead of incremental batch learning"
    )
    
    args = parser.parse_args()
    # Default to batch mode (incremental learning) unless --standard-mode is specified
    use_batches = not args.standard_mode
    train_model(args.dataset, batch_size=args.batch_size, use_batches=use_batches)


if __name__ == "__main__":
    main()
