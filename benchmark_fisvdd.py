# benchmark_fisvdd.py
"""
Comprehensive benchmark of FISVDD on a specified dataset with K-fold validation.

Usage:
    python benchmark_fisvdd.py --dataset LIVE_NFLX_II
    python benchmark_fisvdd.py --dataset LFOVIA_QoE
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, roc_curve, precision_recall_curve
)
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from common_features import transform_df_generic
from fisvdd import fisvdd
from configs import get_config


def score_batch(model, X):
    """Score a batch of samples."""
    out = np.empty(len(X))
    for i, x in enumerate(X):
        s, _ = model.score_fcn(x.reshape(1, -1))
        out[i] = s
    return out


def benchmark_dataset(dataset_name: str):
    """
    Run comprehensive benchmark on specified dataset.
    
    Args:
        dataset_name: Name of the dataset to benchmark
    """
    # Get configuration
    config = get_config(dataset_name)
    
    print(f"\n{'='*70}")
    print(f"FISVDD Benchmark - {config.DATASET_NAME}")
    print(f"{'='*70}\n")
    
    # Load artifacts
    artifact_path = config.get_artifact_path(config.get_model_artifact_name())
    
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"Model artifact not found: {artifact_path}\n"
            f"Please train first: python train_fisvdd.py --dataset {dataset_name}"
        )
    
    print(f"Loading artifacts from: {artifact_path}")
    A = joblib.load(artifact_path)
    scaler = A["scaler"]
    sigma = A["sigma"]
    
    # Load data
    test_path = config.get_resource_path(config.TEST_FILE)
    print(f"Loading data from: {test_path}")
    df = pd.read_csv(test_path)
    
    # Preprocess
    print(f"Preprocessing {len(df)} samples...")
    dfX = transform_df_generic(
        df,
        clip_features=config.CLIP_FEATURES,
        log_features=config.LOG_TRANSFORM_FEATURES,
        invert_features=config.INVERT_FEATURES
    )
    
    X_all = scaler.transform(dfX[config.FEATURE_COLUMNS].astype(float).values)
    
    if config.TARGET_COLUMN and config.TARGET_COLUMN in dfX.columns:
        # Use dataset-specific anomaly definition
        y_raw = dfX[config.TARGET_COLUMN].values
        y_all = np.array([config.is_anomaly(val) for val in y_raw])
    else:
        print("⚠️  No target column - cannot compute metrics")
        return
    
    # Group by content if available
    if "content" in dfX.columns:
        groups = dfX["content"].astype(str).values
    else:
        # Create pseudo-groups
        groups = np.arange(len(dfX)) // 100  # groups of ~100
    
    # K-fold cross-validation
    K = 5 if len(np.unique(groups)) >= 5 else max(2, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=K)
    
    print(f"\nRunning {K}-fold cross-validation...")
    print(f"{'='*70}\n")
    
    fold_rows = []
    t0_all = time.time()
    last_fold_cache = None
    
    for k, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups=groups), 1):
        print(f"Fold {k}/{K}...", end=" ")
        
        # Train only on good windows
        good_mask = (y_all[train_idx] == 0)
        X_train = X_all[train_idx][good_mask]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]
        
        # Train FISVDD
        t0 = time.time()
        model = fisvdd(X_train, sigma)
        model.find_sv()
        t_train = time.time() - t0
        
        # Score test set
        t0 = time.time()
        scores = score_batch(model, X_test)
        t_score = time.time() - t0
        
        # Metrics
        auc = roc_auc_score(y_test, scores)
        ap = average_precision_score(y_test, scores)
        
        # Threshold from training data
        s_train = score_batch(model, X_train)
        tau = float(np.quantile(s_train, config.THRESHOLD_QUANTILE))
        
        # Predictions
        pred_bad = (scores > tau).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, pred_bad, average="binary", zero_division=0
        )
        flag_rate = pred_bad.mean()
        
        fold_rows.append({
            "fold": k,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "AUC": auc,
            "AP": ap,
            "precision": prec,
            "recall": rec,
            "F1": f1,
            "flag_rate": flag_rate,
            "tau": tau,
            "t_train_s": t_train,
            "t_score_s": t_score
        })
        
        print(f"AUC={auc:.3f}, F1={f1:.3f}")
        
        # Keep last fold for visualization
        last_fold_cache = {
            "X_train": X_train,
            "X_test": X_test,
            "y_test": y_test,
            "scores": scores,
            "tau": tau,
            "model": model,
            "test_idx": test_idx
        }
    
    # Results summary
    res = pd.DataFrame(fold_rows)
    
    # Save summary to CSV
    summary_path = config.get_results_path("benchmark_summary.csv")
    res.to_csv(summary_path, index=False)
    print(f"✅ Saved benchmark summary to: {summary_path}")
    
    print(f"\n{'='*70}")
    print("WINDOW-LEVEL K-FOLD RESULTS")
    print(f"{'='*70}\n")
    print(res[["fold", "n_train", "n_test", "AUC", "AP", "F1", "flag_rate"]].to_string(index=False))
    
    print(f"\n{'='*70}")
    print("MEAN ± STD")
    print(f"{'='*70}")
    for m in ["AUC", "AP", "F1", "precision", "recall", "flag_rate"]:
        print(f"{m:12s} {res[m].mean():.4f} ± {res[m].std():.4f}")
    
    print(f"\n{'='*70}")
    print("TIMING")
    print(f"{'='*70}")
    print(f"Train (avg):  {res['t_train_s'].mean():.4f}s")
    print(f"Score (avg):  {res['t_score_s'].mean():.4f}s")
    print(f"Total:        {time.time()-t0_all:.2f}s")
    
    # Create visualizations
    results_dir = os.path.dirname(config.get_results_path("plots"))
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print(f"{'='*70}\n")
    
    lf = last_fold_cache
    
    # 1) ROC curve
    fpr, tpr, _ = roc_curve(lf["y_test"], lf["scores"])
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {res["AUC"].mean():.3f}')
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {config.DATASET_NAME}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = config.get_results_path("roc_curve.png")
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"✅ Saved: {roc_path}")
    
    # 2) Precision-Recall curve
    prec, rec, _ = precision_recall_curve(lf["y_test"], lf["scores"])
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, linewidth=2, label=f'AP = {res["AP"].mean():.3f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {config.DATASET_NAME}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = config.get_results_path("pr_curve.png")
    plt.savefig(pr_path, dpi=200)
    plt.close()
    print(f"✅ Saved: {pr_path}")
    
    # 3) PCA visualization of support vectors
    pca = PCA(n_components=2, random_state=0)
    Xtr_2d = pca.fit_transform(lf["X_train"])
    SV_2d = pca.transform(lf["model"].sv)
    
    plt.figure(figsize=(7, 6))
    plt.scatter(Xtr_2d[:, 0], Xtr_2d[:, 1], s=10, alpha=0.5, label="Normal samples")
    plt.scatter(SV_2d[:, 0], SV_2d[:, 1], marker="*", s=100, 
                color="red", label=f"Support Vectors ({len(SV_2d)})")
    plt.legend()
    plt.title(f"PCA: Support Vectors - {config.DATASET_NAME}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    pca_path = config.get_results_path("pca_support_vectors.png")
    plt.savefig(pca_path, dpi=200)
    plt.close()
    print(f"✅ Saved: {pca_path}")
    
    # 4) Per-content AUC (if content column exists)
    if "content" in dfX.columns:
        df_last = dfX.iloc[lf["test_idx"]].copy()
        df_last["score"] = lf["scores"]
        df_last["bad"] = lf["y_test"]
        
        content_aucs = []
        for c in sorted(df_last["content"].unique()):
            m = (df_last["content"] == c)
            if m.sum() >= 5:
                auc_c = roc_auc_score(df_last.loc[m, "bad"], df_last.loc[m, "score"])
                content_aucs.append((c, auc_c, int(m.sum())))
        
        if content_aucs:
            names = [c for c, _, _ in content_aucs]
            aucs = [a for _, a, _ in content_aucs]
            
            plt.figure(figsize=(10, 5))
            bars = plt.bar(range(len(names)), aucs, color='steelblue')
            plt.xticks(range(len(names)), names, rotation=60, ha="right")
            plt.ylim(0.0, 1.0)
            plt.ylabel("AUC")
            plt.title(f"Per-Content AUC - {config.DATASET_NAME}")
            plt.axhline(y=res["AUC"].mean(), color='red', linestyle='--', 
                       label=f'Mean = {res["AUC"].mean():.3f}')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            content_path = config.get_results_path("per_content_auc.png")
            plt.savefig(content_path, dpi=200)
            plt.close()
            print(f"✅ Saved: {content_path}")
    
    print(f"\n{'='*70}")
    print("✅ BENCHMARK COMPLETE")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FISVDD on a specified dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="LIVE_NFLX_II",
        choices=["LIVE_NFLX_II", "LIVE_NFLX", "LFOVIA_QoE"],
        help="Dataset to benchmark (default: LIVE_NFLX_II)"
    )
    
    args = parser.parse_args()
    benchmark_dataset(args.dataset)


if __name__ == "__main__":
    main()
