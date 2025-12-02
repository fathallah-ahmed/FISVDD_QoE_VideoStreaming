# Multi-Dataset Organization Guide

This document explains the new multi-dataset structure for the FISVDD QoE project.

## 📁 Directory Structure

```
FISVDD_QoE_VideoStreaming/
│
├── resources/                          # Dataset files
│   ├── LIVE_NFLX_II/                  # LIVE-Netflix-II dataset
│   │   ├── LIVE_NFLX_II_FISVDD_train.csv
│   │   └── LIVE_NFLX_II_windows_minimal.csv
│   │
│   └── LFOVIA_QoE/                    # LFOVIA QoE dataset
│       ├── video_1.mat
│       ├── video_2.mat
│       └── ... (more .mat files)
│
├── artifacts/                          # Trained models
│   ├── LIVE_NFLX_II_fisvdd.joblib    # Model for LIVE-Netflix-II
│   └── LFOVIA_QoE_fisvdd.joblib      # Model for LFOVIA (to be created)
│
├── results/                            # Evaluation results
│   ├── LIVE_NFLX_II/                  # Results for LIVE-Netflix-II
│   │   ├── metrics.json
│   │   ├── roc_curve.png
│   │   └── pr_curve.png
│   │
│   ├── LFOVIA_QoE/                    # Results for LFOVIA
│   │   └── (to be generated)
│   │
│   └── comparative_analysis/          # Cross-dataset comparisons
│       ├── comparison_results.csv
│       ├── comparison_report.txt
│       └── dataset_comparison.png
│
├── configs/                            # Dataset configurations
│   ├── __init__.py                    # Config registry
│   ├── base_config.py                 # Base configuration class
│   ├── live_nflx_ii_config.py        # LIVE-Netflix-II config
│   └── lfovia_qoe_config.py          # LFOVIA QoE config
│
├── fisvdd.py                          # Core FISVDD algorithm
├── common_features.py                 # Preprocessing utilities
├── train_fisvdd.py                    # Training script (multi-dataset)
├── test_fisvdd.py                     # Evaluation script
├── explore_lfovia_dataset.py         # LFOVIA data exploration
└── compare_datasets.py                # Cross-dataset comparison
```

## 🚀 Workflow

### 1. Working with LIVE-Netflix-II (Existing Dataset)

```bash
# Train model
python train_fisvdd.py --dataset LIVE_NFLX_II

# Evaluate model
python test_fisvdd.py --dataset LIVE_NFLX_II

# Run benchmarks
python benchmark_fisvdd.py --dataset LIVE_NFLX_II
```

### 2. Working with LFOVIA_QoE (New Dataset)

#### Step 1: Explore the Dataset
```bash
python explore_lfovia_dataset.py
```

This will:
- Load and examine the .mat files
- Show you the data structure
- Help you understand available features

#### Step 2: Configure the Dataset

Edit `configs/lfovia_qoe_config.py` and update:
- `FEATURE_COLUMNS`: List of feature names from your .mat files
- `TARGET_COLUMN`: Name of the QoE/label column (if available)
- `CLIP_FEATURES`: Features that need clipping
- `LOG_TRANSFORM_FEATURES`: Features that need log transformation

#### Step 3: Prepare Training Data

Create a script to convert .mat files to CSV format, or modify `explore_lfovia_dataset.py` to do the conversion.

The CSV should have columns matching your `FEATURE_COLUMNS` configuration.

#### Step 4: Train the Model
```bash
python train_fisvdd.py --dataset LFOVIA_QoE
```

#### Step 5: Evaluate the Model
```bash
python test_fisvdd.py --dataset LFOVIA_QoE
```

### 3. Compare Across Datasets

After training and evaluating both datasets:

```bash
python compare_datasets.py
```

This generates:
- Comparison table (CSV and text)
- Visualization plots
- Summary statistics

## 🔧 Configuration System

Each dataset has its own configuration file that inherits from `BaseDatasetConfig`.

### Key Configuration Parameters

- **Dataset Identification**
  - `DATASET_NAME`: Unique identifier
  - `DATASET_DESCRIPTION`: Human-readable description

- **Features**
  - `FEATURE_COLUMNS`: List of feature names
  - `TARGET_COLUMN`: Target/label column name

- **Preprocessing**
  - `CLIP_FEATURES`: Features to clip to a range
  - `LOG_TRANSFORM_FEATURES`: Features to apply log1p transform

- **Training**
  - `TRAIN_FILE`: Training data filename
  - `TEST_FILE`: Test data filename
  - `THRESHOLD_QUANTILE`: Quantile for anomaly threshold

- **FISVDD Parameters**
  - `SIGMA_METHOD`: "median_heuristic" or "fixed"
  - `NORMAL_BUFFER_MAX`: Max normal samples to keep
  - `REFIT_EVERY`: How often to refit the model

## 📊 Results Tracking

Each dataset evaluation saves results to `results/{DATASET_NAME}/metrics.json`:

```json
{
  "dataset": "LIVE_NFLX_II",
  "auc": 0.832,
  "ap": 0.786,
  "f1": 0.654,
  "precision": 0.721,
  "recall": 0.598,
  "threshold": -0.0021,
  "flag_rate": 6.5,
  "num_sv": 234,
  "train_samples": 2847
}
```

## 🔄 Adding More Datasets

To add a new dataset:

1. **Create dataset directory**: `resources/{DATASET_NAME}/`
2. **Create config file**: `configs/{dataset_name}_config.py`
3. **Register in config**: Add to `DATASET_CONFIGS` in `configs/__init__.py`
4. **Prepare data**: Convert to CSV format with required features
5. **Train and evaluate**: Use the standard workflow

## 💡 Tips

- **Feature Engineering**: Each dataset may need different preprocessing
- **Hyperparameter Tuning**: Adjust `THRESHOLD_QUANTILE`, `SIGMA_METHOD` per dataset
- **Cross-Dataset Learning**: Compare which features generalize best
- **Transfer Learning**: Try using a model trained on one dataset to score another

## 🐛 Troubleshooting

**Issue**: "Training file not found"
- **Solution**: Ensure your CSV file is in `resources/{DATASET_NAME}/` and matches the filename in config

**Issue**: "Unknown dataset"
- **Solution**: Check that the dataset is registered in `configs/__init__.py`

**Issue**: "Feature column not found"
- **Solution**: Verify that your CSV has all columns listed in `FEATURE_COLUMNS`

## 📚 Next Steps

1. Explore LFOVIA_QoE dataset structure
2. Update LFOVIA configuration with correct features
3. Convert .mat files to CSV format
4. Train FISVDD on LFOVIA_QoE
5. Compare results across both datasets
6. Analyze which features work best across datasets
