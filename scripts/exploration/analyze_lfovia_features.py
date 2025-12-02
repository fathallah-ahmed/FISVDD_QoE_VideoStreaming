# analyze_lfovia_features.py
"""
Analyze feature correlations for LFOVIA dataset to understand model performance.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def analyze():
    # Load data
    train_path = "resources/LFOVIA_QoE/LFOVIA_QoE_train.csv"
    test_path = "resources/LFOVIA_QoE/LFOVIA_QoE_test.csv"
    
    if not os.path.exists(train_path):
        print("❌ Data not found. Run convert_lfovia_to_csv.py first.")
        return

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df = pd.concat([df_train, df_test])
    
    # Features to analyze
    features = ["NIQE", "PSNR", "SSIM", "STRRED", "TSL", "Nrebuffers"]
    target = "score_continuous"
    
    # 1. Correlation Matrix
    cols = features + [target]
    corr = df[cols].corr()
    
    # 2. Check distributions for Normal vs Anomaly (using current threshold < 50)
    df['is_anomaly'] = df[target] < 50
    
    # Save results to text file
    os.makedirs("results/LFOVIA_QoE", exist_ok=True)
    with open("results/LFOVIA_QoE/analysis_report.txt", "w") as f:
        f.write(f"Loaded {len(df)} samples.\n\n")
        
        f.write("Correlation with Target (score_continuous):\n")
        f.write(corr[target].sort_values(ascending=False).to_string())
        f.write("\n\n")
        
        f.write("Feature Means by Class (Normal vs Anomaly):\n")
        f.write(df.groupby('is_anomaly')[features].mean().to_string())
        f.write("\n")
    
    print("✅ Saved analysis report to results/LFOVIA_QoE/analysis_report.txt")
    
    # 3. Save correlation plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Feature Correlations - LFOVIA QoE")
    plt.tight_layout()
    plt.savefig("results/LFOVIA_QoE/feature_correlations.png")
    print("✅ Saved correlation plot to results/LFOVIA_QoE/feature_correlations.png")

if __name__ == "__main__":
    analyze()
