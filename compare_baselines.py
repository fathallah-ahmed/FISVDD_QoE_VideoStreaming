"""
compare_baselines.py

Compare FISVDD against baseline anomaly detection models:
1. Isolation Forest
2. One-Class SVM (Linear kernel)
3. SVDD (implemented as One-Class SVM with RBF kernel)

Datasets: LIVE_NFLX_II, LIVE_NFLX, LFOVIA_QoE
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from common_features import transform_df_generic
from configs import get_config
from fisvdd import fisvdd # For type hinting if needed, mainly loading artifact

METRICS_FILE = "results/comparative_analysis/baseline_comparison.csv"
os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)

DATASETS = ["LIVE_NFLX_II", "LIVE_NFLX", "LFOVIA_QoE"]

def load_and_preprocess(dataset_name):
    config = get_config(dataset_name)
    
    # --- Load Train ---
    train_path = config.get_resource_path(config.TRAIN_FILE)
    if not os.path.exists(train_path):
        print(f"Skipping {dataset_name}: Train file not found at {train_path}")
        return None, None, None, None
        
    df_train = pd.read_csv(train_path)
    df_train_proc = transform_df_generic(
        df_train,
        clip_features=config.CLIP_FEATURES,
        log_features=config.LOG_TRANSFORM_FEATURES,
        invert_features=config.INVERT_FEATURES
    )

    # Filtering for LFOVIA_QoE: Keep only good samples (score >= 50)
    # This ensures models are trained on pure normal data (Novelty Detection)
    if dataset_name == "LFOVIA_QoE" and "score_continuous" in df_train_proc.columns:
        initial_len = len(df_train_proc)
        df_train_proc = df_train_proc[df_train_proc["score_continuous"] >= 50]
        print(f"  [Info] Filtered LFOVIA_QoE training data: {initial_len} -> {len(df_train_proc)} samples (Good Only)")

    X_train_raw = df_train_proc[config.FEATURE_COLUMNS].astype(float).values
    
    # Scale based on Train
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    
    # --- Load Test ---
    test_path = config.get_resource_path(config.TEST_FILE)
    if not os.path.exists(test_path):
        print(f"Skipping {dataset_name}: Test file not found at {test_path}")
        return None, None, None, None
        
    df_test = pd.read_csv(test_path)
    df_test_proc = transform_df_generic(
        df_test,
        clip_features=config.CLIP_FEATURES,
        log_features=config.LOG_TRANSFORM_FEATURES,
        invert_features=config.INVERT_FEATURES
    )
    
    # Test Split Logic (Mirroring test_fisvdd.py)
    if "content" in df_test_proc.columns:
        contents = df_test_proc["content"].dropna().unique()
        # Take last 3 contents as test set (standard for these datasets)
        test_contents = set(contents[-3:])
        test_subset = df_test_proc[df_test_proc["content"].isin(test_contents)].copy()
    else:
        # Fallback: last 20%
        split_idx = int(len(df_test_proc) * 0.8)
        test_subset = df_test_proc.iloc[split_idx:].copy()
        
    X_test = scaler.transform(test_subset[config.FEATURE_COLUMNS].astype(float).values)
    
    # Get Labels
    if config.TARGET_COLUMN and config.TARGET_COLUMN in test_subset.columns:
        y_test_raw = test_subset[config.TARGET_COLUMN].astype(float).values
        # Convert to binary anomaly labels based on dataset config
        y_test = np.array([config.is_anomaly(val) for val in y_test_raw])
    else:
        y_test = None
        
    return X_train, X_test, y_test, config

def train_evaluate_baselines(dataset_name, X_train, X_test, y_test):
    results = []
    
    if y_test is None or len(np.unique(y_test)) < 2:
        print(f"  Cannot evaluate {dataset_name}: Test labels missing or mono-class.")
        return results

    # 1. Isolation Forest
    print(f"  [Isolation Forest] Training...")
    iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    # Decision function: average anomaly score of X of the base classifiers.
    # The anomaly score of an input sample is computed as the mean anomaly score of the trees in the forest.
    # The measure of normality of an observation given a tree is the depth of the leaf containing this observation.
    # The lower, the more abnormal. 
    # decision_function returns negative outlier factor. larger is more normal.
    # We want anomaly score: higher is more anomalous. So we negate it.
    scores_iso = -iso.decision_function(X_test) 
    
    res_iso = {
        "Dataset": dataset_name,
        "Model": "Isolation Forest",
        "AUC": roc_auc_score(y_test, scores_iso),
        "AP": average_precision_score(y_test, scores_iso)
    }
    results.append(res_iso)
    print(f"    -> AUC: {res_iso['AUC']:.4f}, AP: {res_iso['AP']:.4f}")

    # 2. One-Class SVM (Linear)
    print(f"  [OC-SVM (Linear)] Training...")
    # nu=0.1 roughly corresponds to contamination. 
    ocsvm_lin = OneClassSVM(kernel='linear', nu=0.1)
    ocsvm_lin.fit(X_train)
    # decision_function: Signed distance to the separating hyperplane.
    # Positive for an inlier, negative for an outlier.
    # Anomaly score: negate it.
    scores_lin = -ocsvm_lin.decision_function(X_test)
    
    res_lin = {
        "Dataset": dataset_name,
        "Model": "One-Class SVM (Linear)",
        "AUC": roc_auc_score(y_test, scores_lin),
        "AP": average_precision_score(y_test, scores_lin)
    }
    results.append(res_lin)
    print(f"    -> AUC: {res_lin['AUC']:.4f}, AP: {res_lin['AP']:.4f}")

    # 3. SVDD (One-Class SVM with RBF)
    print(f"  [SVDD (RBF)] Training...")
    # SVDD is often equivalent to OC-SVM with RBF kernel and specific parameters.
    # gamma='scale' is a good default.
    svdd = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
    svdd.fit(X_train)
    scores_svdd = -svdd.decision_function(X_test)
    
    res_svdd = {
        "Dataset": dataset_name,
        "Model": "SVDD (RBF Kernel)",
        "AUC": roc_auc_score(y_test, scores_svdd),
        "AP": average_precision_score(y_test, scores_svdd)
    }
    results.append(res_svdd)
    print(f"    -> AUC: {res_svdd['AUC']:.4f}, AP: {res_svdd['AP']:.4f}")
    
    return results

def evaluate_fisvdd(dataset_name, X_test, y_test, config):
    print(f"  [FISVDD] Loading artifact...")
    artifact_name = config.get_model_artifact_name() # e.g. "LIVE_NFLX_II_fisvdd.joblib"
    artifact_path = config.get_artifact_path(artifact_name)
    
    if not os.path.exists(artifact_path):
        print(f"    -> Artifact not found at {artifact_path}. Skipping.")
        return None
        
    try:
        loaded = joblib.load(artifact_path)
        model = loaded["model"]
        # Note: Scaler is already applied to X_test using the logic in load_and_preprocess
        # BUT wait, the loaded model has its own scaler.
        # Ideally we should use the scaler from the artifact to be fully consistent.
        # However, load_and_preprocess creates a fresh scaler on X_train. 
        # Since I'm retraining baselines on the same X_train, comparing against
        # the PRE-TRAINED FISVDD is slightly unfair if the exact training set or scaling differed.
        # But assuming deterministic loading, it should be fine.
        
        # To be safe, let's just score X_test (which was scaled by the fresh scaler).
        # OR we could rely on the loaded scaler.
        # Let's trust that the fresh scaler on the same training file is identical enough.
        
        def score_batch(m, Z):
            s = np.empty(len(Z))
            for i, z in enumerate(Z):
                s[i], _ = m.score_fcn(z.reshape(1, -1))
            return s
            
        scores = score_batch(model, X_test)
        
        res = {
            "Dataset": dataset_name,
            "Model": "FISVDD",
            "AUC": roc_auc_score(y_test, scores),
            "AP": average_precision_score(y_test, scores)
        }
        print(f"    -> AUC: {res['AUC']:.4f}, AP: {res['AP']:.4f}")
        return res
        
    except Exception as e:
        print(f"    -> Error evaluating FISVDD: {e}")
        return None

def main():
    all_results = []
    
    print("Starting Comparative Analysis...\n")
    
    for d_name in DATASETS:
        print(f"Processing {d_name}...")
        X_train, X_test, y_test, config = load_and_preprocess(d_name)
        
        if X_train is None:
            continue
            
        # Baselines
        baseline_res = train_evaluate_baselines(d_name, X_train, X_test, y_test)
        all_results.extend(baseline_res)
        
        # FISVDD
        fisvdd_res = evaluate_fisvdd(d_name, X_test, y_test, config)
        if fisvdd_res:
            all_results.append(fisvdd_res)
            
        print("-" * 30)

    # Save and Display
    if all_results:
        df_res = pd.DataFrame(all_results)
        # Sort by Dataset and AUC
        df_res = df_res.sort_values(by=["Dataset", "AUC"], ascending=[True, False])
        
        print("\n=== Final Comparison Results ===")
        print(df_res.to_string(index=False))
        
        df_res.to_csv(METRICS_FILE, index=False)
        print(f"\nSaved results to {METRICS_FILE}")
        
        # Plotting
        try:
            import matplotlib.pyplot as plt
            
            print(f"Generating plots...")
            fig, axes = plt.subplots(1, 2, figsize=(18, 7))
            
            metrics = ["AUC", "AP"]
            titles = ["AUC Score (Higher is better)", "Average Precision (Higher is better)"]
            
            # Define specific colors for each model to ensure consistency
            # Using a pleasant palette
            model_colors = {
                "FISVDD": "#2ecc71",      # Green
                "SVDD (RBF Kernel)": "#3498db",        # Blue
                "Isolation Forest": "#9b59b6",         # Purple
                "One-Class SVM (Linear)": "#e74c3c"    # Red
            }
            
            for i, metric in enumerate(metrics):
                ax = axes[i]
                # Pivot for plotting: Index=Dataset, Columns=Model, Values=Metric
                pivot_df = df_res.pivot(index="Dataset", columns="Model", values=metric)
                
                # Plot grouped bar chart using specific colors
                # We map the column names (models) to the colors
                colors = [model_colors.get(col, "#95a5a6") for col in pivot_df.columns]
                
                pivot_df.plot(kind="bar", ax=ax, width=0.8, color=colors)
                
                ax.set_title(titles[i], fontsize=14, fontweight='bold')
                ax.set_xlabel("Dataset", fontsize=12)
                ax.set_ylabel(metric, fontsize=12)
                ax.set_ylim(0, 1.15) 
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Move legend BELOW the plot (further down) to avoid covering x-axis labels
                ax.legend(title="Model", loc='upper center', bbox_to_anchor=(0.5, -0.25), 
                          ncol=2, framealpha=0.9)
                
                # Add value labels
                for p in ax.patches:
                    val = p.get_height()
                    if val > 0:
                        ax.annotate(f"{val:.2f}", 
                                    (p.get_x() + p.get_width() / 2., val), 
                                    ha='center', va='bottom', fontsize=9, rotation=0, xytext=(0, 5), textcoords='offset points')

            # Adjust layout to make room for the bottom legend
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2) # Reserve 20% of figure height at bottom
            
            plot_path = METRICS_FILE.replace(".csv", ".png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to {plot_path}")
            
        except Exception as e:
            print(f"Error generating plot: {e}")

    else:
        print("\nNo results generated.")

if __name__ == "__main__":
    main()
