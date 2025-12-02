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


def train_model(dataset_name: str):
    """
    Train FISVDD model for the specified dataset.
    
    Args:
        dataset_name: Name of the dataset to train on
    """
    # Get dataset configuration
    config = get_config(dataset_name)
    
    print(f"\n{'='*60}")
    print(f"Training FISVDD on {config.DATASET_NAME}")
    print(f"Description: {config.DATASET_DESCRIPTION}")
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
        }
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
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train FISVDD model on a specified dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["LIVE_NFLX_II", "LFOVIA_QoE"],
        help="Dataset to train on"
    )
    
    args = parser.parse_args()
    train_model(args.dataset)


if __name__ == "__main__":
    main()
