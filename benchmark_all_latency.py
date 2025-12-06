import time
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from common_features import transform_df_generic
from configs import get_config
from fisvdd import fisvdd

# Configuration
DATASETS = ["LIVE_NFLX_II", "LIVE_NFLX", "LFOVIA_QoE"]
N_SAMPLES = 1000 # Number of samples to test inference on

def load_data_and_train_baselines(dataset_name):
    print(f"Loading data for {dataset_name}...")
    config = get_config(dataset_name)
    train_path = config.get_resource_path(config.TRAIN_FILE)
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return None, None, None, None, None

    df_train = pd.read_csv(train_path)
    df_train_proc = transform_df_generic(
        df_train,
        clip_features=config.CLIP_FEATURES,
        log_features=config.LOG_TRANSFORM_FEATURES,
        invert_features=config.INVERT_FEATURES
    )
    
    # Filter LFOVIA to be consistent with comparison (Good Only training)
    if dataset_name == "LFOVIA_QoE" and "score_continuous" in df_train_proc.columns:
        df_train_proc = df_train_proc[df_train_proc["score_continuous"] >= 50]
        
    X_train_raw = df_train_proc[config.FEATURE_COLUMNS].astype(float).values
    
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    
    # Train Baselines
    print(f"[{dataset_name}] Training Isolation Forest...")
    iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1).fit(X_train)
    
    print(f"[{dataset_name}] Training OC-SVM (Linear)...")
    ocsvm = OneClassSVM(kernel='linear', nu=0.1).fit(X_train)
    
    print(f"[{dataset_name}] Training SVDD (RBF)...")
    svdd = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale').fit(X_train)
    
    return X_train, iso, ocsvm, svdd, config

def load_fisvdd(config):
    # ... (same as before)
    print("Loading FISVDD artifact...")
    artifact_name = config.get_model_artifact_name()
    artifact_path = config.get_artifact_path(artifact_name)
    
    if os.path.exists(artifact_path):
        loaded = joblib.load(artifact_path)
        return loaded["model"]
    else:
        print(f"FISVDD artifact not found at {artifact_path}")
        return None

def benchmark_model(dataset, name, model, X_sample, predict_fn_name="decision_function"):
    # ... (same logic, just includes dataset name)
    # print(f"Benchmarking {name}...") 
    latencies = []
    
    # Warmup
    for _ in range(10):
        if predict_fn_name == "score_fcn":
            model.score_fcn(X_sample[0].reshape(1, -1))
        else:
            getattr(model, predict_fn_name)(X_sample[0].reshape(1, -1))
            
    # timed run
    t0_total = time.perf_counter()
    for i in range(len(X_sample)):
        sample = X_sample[i].reshape(1, -1)
        t_start = time.perf_counter()
        
        if predict_fn_name == "score_fcn": # FISVDD
            model.score_fcn(sample)
        else: # Sklearn
            getattr(model, predict_fn_name)(sample)
            
        latencies.append((time.perf_counter() - t_start) * 1000) # ms
        
    total_time = time.perf_counter() - t0_total
    
    latencies = np.array(latencies)
    mean_lat = latencies.mean()
    p99_lat = np.percentile(latencies, 99)
    throughput = len(X_sample) / total_time
    
    return {
        "Dataset": dataset,
        "Model": name,
        "Mean Latency (ms)": mean_lat,
        "P99 Latency (ms)": p99_lat,
        "Throughput (samples/s)": throughput
    }

def main():
    all_results = []
    
    for d_name in DATASETS:
        print(f"\nProcessing {d_name}...")
        X_train, iso, ocsvm, svdd_rbf, config = load_data_and_train_baselines(d_name)
        
        if X_train is None:
            continue

        fisvdd_model = load_fisvdd(config)
        
        # Sample data
        indices = np.random.choice(len(X_train), size=min(N_SAMPLES, len(X_train)), replace=False)
        X_sample = X_train[indices]
        
        # 1. Isolation Forest
        all_results.append(benchmark_model(d_name, "Isolation Forest", iso, X_sample, "decision_function"))
        
        # 2. OC-SVM (Linear)
        all_results.append(benchmark_model(d_name, "One-Class SVM (Linear)", ocsvm, X_sample, "decision_function"))
        
        # 3. SVDD (RBF)
        all_results.append(benchmark_model(d_name, "SVDD (RBF)", svdd_rbf, X_sample, "decision_function"))
        
        # 4. FISVDD
        if fisvdd_model:
            all_results.append(benchmark_model(d_name, "FISVDD", fisvdd_model, X_sample, "score_fcn"))
        
    # Summary
    print("\n" + "="*60)
    print("LATENCY BENCHMARK RESULTS (ALL DATASETS)")
    print("="*60)
    df_res = pd.DataFrame(all_results)
    df_res = df_res.sort_values(by=["Dataset", "Mean Latency (ms)"])
    print(df_res.to_string(index=False))
    
    # Save
    results_path = "results/comparative_analysis/latency_benchmark.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    df_res.to_csv(results_path, index=False)
    print(f"\nSaved results to {results_path}")

    # Plot (Grouped Bar Chart by Dataset)
    try:
        import matplotlib.pyplot as plt
        
        # Manual grouping
        plt.figure(figsize=(14, 8))
        
        # Pivot: Index=Dataset, Columns=Model, Values=Mean Latency
        pivot_df = df_res.pivot(index="Dataset", columns="Model", values="Mean Latency (ms)")
        
        model_colors = {
            "FISVDD": "#2ecc71",
            "SVDD (RBF)": "#3498db",
            "Isolation Forest": "#9b59b6",
            "One-Class SVM (Linear)": "#e74c3c"
        }
        # Get colors for columns in correct order
        colors = [model_colors.get(col, "gray") for col in pivot_df.columns]
        
        ax = pivot_df.plot(kind="bar",  width=0.8, color=colors, figsize=(12, 6))
        
        plt.title("Inference Latency Comparison across Datasets (Lower is Better)", fontsize=16)
        plt.xlabel("Dataset", fontsize=14)
        plt.ylabel("Mean Latency (ms) - Log Scale", fontsize=14)
        plt.yscale('log') # Log scale because IsoForest is much slower
        plt.grid(axis='y', linestyle='--', alpha=0.5, which='both')
        
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add labels
        for p in ax.patches:
            val = p.get_height()
            if val > 0:
                ax.annotate(f"{val:.3f}", 
                            (p.get_x() + p.get_width() / 2., val), 
                            ha='center', va='bottom', fontsize=8, rotation=90, xytext=(0, 5), textcoords='offset points')
                            
        plt.tight_layout()
        plt.savefig(results_path.replace(".csv", ".png"), dpi=300, bbox_inches='tight')
        print("Saved plot.")
        
    except Exception as e:
        print(f"Plot error: {e}")

if __name__ == "__main__":
    main()
