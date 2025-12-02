# test_fisvdd.py
"""
Evaluate FISVDD model on a specified dataset.

Usage:
    python test_fisvdd.py --dataset LIVE_NFLX_II
    python test_fisvdd.py --dataset LFOVIA_QoE
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

from common_features import transform_df_generic
from configs import get_config


def evaluate_model(dataset_name: str, save_results: bool = True):
    """
    Evaluate FISVDD model for the specified dataset.
    
    Args:
        dataset_name: Name of the dataset to evaluate
        save_results: Whether to save results to JSON file
    """
    # Get dataset configuration
    config = get_config(dataset_name)
    
    print(f"\n{'='*60}")
    print(f"Evaluating FISVDD on {config.DATASET_NAME}")
    print(f"{'='*60}\n")
    
    # Load model artifacts
    artifact_path = config.get_artifact_path(config.get_model_artifact_name())
    
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"Model artifact not found: {artifact_path}\n"
            f"Please train the model first: python train_fisvdd.py --dataset {dataset_name}"
        )
    
    print(f"[1/5] Loading model from: {artifact_path}")
    A = joblib.load(artifact_path)
    scaler = A["scaler"]
    model = A["model"]
    threshold = A["threshold"]
    features = A["features"]
    
    print(f"      Threshold: {threshold:.6f}")
    print(f"      Support Vectors: {len(model.sv)}")
    
    # Load test data
    test_path = config.get_resource_path(config.TEST_FILE)
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test file not found: {test_path}\n"
            f"Please ensure the test data is available."
        )
    
    print(f"\n[2/5] Loading test data from: {test_path}")
    df = pd.read_csv(test_path)
    print(f"      Total samples: {len(df)}")
    
    # Preprocess
    print(f"\n[3/5] Preprocessing data...")
    dfX = transform_df_generic(
        df,
        clip_features=config.CLIP_FEATURES,
        log_features=config.LOG_TRANSFORM_FEATURES,
        invert_features=config.INVERT_FEATURES
    )
    
    # Create test split (hold-out last 3 contents if available)
    if "content" in dfX.columns:
        contents = dfX["content"].dropna().unique()
        test_contents = set(contents[-3:])
        test = dfX[dfX["content"].isin(test_contents)].copy()
        print(f"      Test contents: {len(test_contents)}")
        print(f"      Test samples: {len(test)}")
    else:
        # Use last 20% as test
        split_idx = int(len(dfX) * 0.8)
        test = dfX.iloc[split_idx:].copy()
        print(f"      Test samples (last 20%): {len(test)}")
    
    # Extract features and labels
    X = scaler.transform(test[features].astype(float).values)
    
    if config.TARGET_COLUMN and config.TARGET_COLUMN in test.columns:
        y = test[config.TARGET_COLUMN].astype(float).values
        has_labels = True
    else:
        y = None
        has_labels = False
        print(f"      ⚠️  No target column found, skipping metric computation")
    
    # Score samples
    print(f"\n[4/5] Computing anomaly scores...")
    
    def score_batch(m, Z):
        s = np.empty(len(Z))
        for i, z in enumerate(Z):
            s[i], _ = m.score_fcn(z.reshape(1, -1))
        return s
    
    scores = score_batch(model, X)
    predictions = (scores > threshold).astype(int)
    flag_rate = predictions.mean() * 100
    
    print(f"      Scored {len(scores)} samples")
    print(f"      Flag rate: {flag_rate:.2f}%")
    
    # Compute metrics if labels available
    print(f"\n[5/5] Computing metrics...")
    
    results = {
        "dataset": dataset_name,
        "test_samples": len(test),
        "threshold": float(threshold),
        "flag_rate": float(flag_rate),
        "num_sv": len(model.sv),
    }
    
    if has_labels:
        # Use dataset-specific anomaly definition
        y_true = np.array([config.is_anomaly(val) for val in y])
        
        # Check if we have both classes
        if len(np.unique(y_true)) < 2:
            print(f"⚠️  Warning: Test set contains only one class (all {'anomalies' if y_true[0] else 'normal'}).")
            print(f"    AUC and other metrics might be undefined.")
        
        try:
            auc = roc_auc_score(y_true, scores)
        except ValueError:
            auc = float('nan')
            
        ap = average_precision_score(y_true, scores)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, predictions, average='binary', zero_division=0
        )
        
        results.update({
            "auc": float(auc),
            "ap": float(ap),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        })
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"AUC:        {auc:.4f}")
        print(f"AP:         {ap:.4f}")
        print(f"F1:         {f1:.4f}")
        print(f"Precision:  {precision:.4f}")
        print(f"Recall:     {recall:.4f}")
        print(f"Flag Rate:  {flag_rate:.2f}%")
        print(f"Threshold:  {threshold:.6f}")
        print(f"{'='*60}\n")
    else:
        print(f"\n⚠️  No labels available - only showing flag rate")
        print(f"Flag Rate: {flag_rate:.2f}%")
        print(f"Threshold: {threshold:.6f}\n")
    
    # Save results
    if save_results:
        results_dir = os.path.dirname(config.get_results_path("metrics.json"))
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = config.get_results_path("metrics.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to: {results_path}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FISVDD model on a specified dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="LIVE_NFLX_II",
        choices=["LIVE_NFLX_II", "LFOVIA_QoE"],
        help="Dataset to evaluate on (default: LIVE_NFLX_II)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    
    args = parser.parse_args()
    evaluate_model(args.dataset, save_results=not args.no_save)


if __name__ == "__main__":
    main()
