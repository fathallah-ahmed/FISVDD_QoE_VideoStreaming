# compare_datasets.py
"""
Compare FISVDD performance across multiple datasets.

Usage:
    python compare_datasets.py
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


def load_results(results_dir: str, dataset_name: str):
    """
    Load evaluation results for a specific dataset.
    
    Args:
        results_dir: Base results directory
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary of results or None if not found
    """
    results_path = os.path.join(results_dir, dataset_name, "metrics.json")
    
    if not os.path.exists(results_path):
        return None
    
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results for {dataset_name}: {e}")
        return None


def create_comparison_table(all_results: dict):
    """
    Create a comparison table of metrics across datasets.
    
    Args:
        all_results: Dictionary mapping dataset names to their results
        
    Returns:
        pandas DataFrame with comparison
    """
    rows = []
    
    for dataset_name, results in all_results.items():
        if results is None:
            continue
            
        row = {
            'Dataset': dataset_name,
            'AUC': results.get('auc', 'N/A'),
            'AP': results.get('ap', 'N/A'),
            'F1': results.get('f1', 'N/A'),
            'Precision': results.get('precision', 'N/A'),
            'Recall': results.get('recall', 'N/A'),
            'Flag Rate (%)': results.get('flag_rate', 'N/A'),
            'Threshold': results.get('threshold', 'N/A'),
            'Support Vectors': results.get('num_sv', 'N/A'),
            'Train Samples': results.get('train_samples', 'N/A'),
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_comparison(all_results: dict, output_dir: str):
    """
    Create comparison plots across datasets.
    
    Args:
        all_results: Dictionary mapping dataset names to their results
        output_dir: Directory to save plots
    """
    datasets = list(all_results.keys())
    
    # Extract metrics
    metrics = {
        'AUC': [],
        'AP': [],
        'F1': [],
        'Precision': [],
        'Recall': []
    }
    
    for dataset in datasets:
        results = all_results[dataset]
        if results:
            metrics['AUC'].append(results.get('auc', 0))
            metrics['AP'].append(results.get('ap', 0))
            metrics['F1'].append(results.get('f1', 0))
            metrics['Precision'].append(results.get('precision', 0))
            metrics['Recall'].append(results.get('recall', 0))
        else:
            for key in metrics:
                metrics[key].append(0)
    
    # Create bar plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('FISVDD Performance Comparison Across Datasets', fontsize=16, fontweight='bold')
    
    metric_names = list(metrics.keys())
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        bars = ax.bar(datasets, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(datasets)])
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        # Rotate x-axis labels if needed
        if len(datasets) > 2:
            ax.set_xticklabels(datasets, rotation=45, ha='right')
    
    # Hide the last subplot if we have odd number of metrics
    if len(metrics) < 6:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'dataset_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved to: {output_path}")
    
    plt.close()


def save_comparison_report(comparison_df: pd.DataFrame, output_dir: str):
    """
    Save comparison report as CSV and text.
    
    Args:
        comparison_df: DataFrame with comparison results
        output_dir: Directory to save report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'comparison_results.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"✅ Comparison CSV saved to: {csv_path}")
    
    # Save as formatted text report
    txt_path = os.path.join(output_dir, 'comparison_report.txt')
    with open(txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FISVDD MULTI-DATASET COMPARISON REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        # Add summary statistics
        f.write("="*80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        numeric_cols = ['AUC', 'AP', 'F1', 'Precision', 'Recall']
        for col in numeric_cols:
            if col in comparison_df.columns:
                try:
                    values = pd.to_numeric(comparison_df[col], errors='coerce')
                    f.write(f"{col}:\n")
                    f.write(f"  Mean: {values.mean():.4f}\n")
                    f.write(f"  Std:  {values.std():.4f}\n")
                    f.write(f"  Min:  {values.min():.4f}\n")
                    f.write(f"  Max:  {values.max():.4f}\n\n")
                except:
                    pass
    
    print(f"✅ Comparison report saved to: {txt_path}")


def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    output_dir = os.path.join(results_dir, "comparative_analysis")
    
    # Datasets to compare
    datasets = ["LIVE_NFLX_II", "LFOVIA_QoE"]
    
    print("\n" + "="*60)
    print("FISVDD Multi-Dataset Comparison")
    print("="*60 + "\n")
    
    # Load results for all datasets
    all_results = {}
    for dataset in datasets:
        print(f"Loading results for {dataset}...")
        results = load_results(results_dir, dataset)
        if results:
            all_results[dataset] = results
            print(f"  ✅ Loaded")
        else:
            print(f"  ⚠️  No results found (run evaluation first)")
            all_results[dataset] = None
    
    if not any(all_results.values()):
        print("\n❌ No results found for any dataset!")
        print("Please run evaluation on at least one dataset first.")
        return
    
    # Create comparison table
    print("\n" + "="*60)
    print("Comparison Table")
    print("="*60 + "\n")
    
    comparison_df = create_comparison_table(all_results)
    print(comparison_df.to_string(index=False))
    
    # Save comparison report
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60 + "\n")
    
    save_comparison_report(comparison_df, output_dir)
    
    # Create plots
    valid_results = {k: v for k, v in all_results.items() if v is not None}
    if len(valid_results) > 0:
        plot_comparison(valid_results, output_dir)
    
    print("\n" + "="*60)
    print("✅ Comparison complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
