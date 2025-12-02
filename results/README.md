# Results Directory

This directory contains evaluation results, benchmarks, and visualizations for each dataset.

## 📁 Structure

```
results/
├── LIVE_NFLX_II/           # Results for LIVE-Netflix-II dataset
├── LFOVIA_QoE/             # Results for LFOVIA QoE dataset
└── comparative_analysis/    # Cross-dataset comparisons
```

## 📊 Generated Files

### Per-Dataset Results

Each dataset directory (`LIVE_NFLX_II/`, `LFOVIA_QoE/`) contains:

#### Metrics
- **`metrics.json`** - Performance metrics (AUC, AP, F1, precision, recall, flag rate)
- **`benchmark_summary.csv`** - K-fold cross-validation results with statistics
- **`analysis_report.txt`** - Feature correlation analysis (LFOVIA only)

#### Visualizations
- **`roc_curve.png`** - ROC curve showing true positive vs false positive rate
- **`pr_curve.png`** - Precision-Recall curve
- **`pca_support_vectors.png`** - 2D PCA projection of support vectors
- **`per_content_auc.png`** - AUC scores by video content
- **`feature_correlations.png`** - Heatmap of feature correlations (LFOVIA only)

### Comparative Analysis

The `comparative_analysis/` directory contains:
- Cross-dataset performance comparisons
- Feature importance across datasets
- Statistical analysis reports

## 🔄 Regenerating Results

Results are automatically generated when you run:

```bash
# Test a dataset (generates metrics.json)
python test_fisvdd.py --dataset DATASET_NAME

# Benchmark with K-fold validation (generates all visualizations)
python benchmark_fisvdd.py --dataset DATASET_NAME

# Compare datasets (generates comparative analysis)
python compare_datasets.py
```

## 📈 Understanding the Metrics

### metrics.json Structure

```json
{
  "dataset": "LFOVIA_QoE",
  "test_samples": 360,
  "threshold": -0.0019,
  "flag_rate": 24.72,
  "num_sv": 30,
  "auc": 0.80,
  "ap": 0.42,
  "f1": 0.49,
  "precision": 0.48,
  "recall": 0.49
}
```

**Key Metrics:**
- **AUC** (Area Under ROC Curve): Overall discrimination ability (0.5 = random, 1.0 = perfect)
- **AP** (Average Precision): Quality of ranked predictions
- **F1**: Harmonic mean of precision and recall
- **Precision**: Of flagged anomalies, how many are actual anomalies
- **Recall**: Of actual anomalies, how many were detected
- **Flag Rate**: Percentage of windows flagged as anomalies
- **num_sv**: Number of support vectors in the model

### What's Good Performance?

| Metric | Poor | Fair | Good | Excellent |
|--------|------|------|------|-----------|
| AUC | < 0.6 | 0.6-0.7 | 0.7-0.85 | > 0.85 |
| AP | < 0.4 | 0.4-0.6 | 0.6-0.8 | > 0.8 |
| F1 | < 0.3 | 0.3-0.5 | 0.5-0.7 | > 0.7 |

## 🎨 Visualization Guide

### ROC Curve
- **X-axis**: False Positive Rate (false alarms)
- **Y-axis**: True Positive Rate (correct detections)
- **Ideal**: Curve hugs top-left corner (high TPR, low FPR)

### PR Curve  
- **X-axis**: Recall (anomaly coverage)
- **Y-axis**: Precision (accuracy of flags)
- **Ideal**: Curve stays close to top-right

### PCA Projection
- Shows how well support vectors represent the normal data manifold
- Tighter clustering = better model representation

### Per-Content AUC
- Shows model performance on individual videos
- Helps identify which content types are harder to model

## 🧹 Cleaning Results

To remove all generated results:

```bash
# Windows
Remove-Item -Recurse results\*\*.png, results\*\*.json, results\*\*.csv, results\*\*.txt

# Linux/Mac
find results/ -type f \( -name "*.png" -o -name "*.json" -o -name "*.csv" -o -name "*.txt" \) -delete
```

**Note**: Directory structure is preserved by `.gitkeep` files.

## 📝 Notes

- Results are **not version controlled** (excluded by `.gitignore`)
- Results are **dataset-specific** and stored separately
- Regenerating results overwrites existing files
- Results are generated automatically - no manual intervention needed

---

For more information, see the main [README.md](../README.md) or [MULTI_DATASET_GUIDE.md](../MULTI_DATASET_GUIDE.md).
