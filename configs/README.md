# Dataset Configurations

This directory contains dataset-specific configurations for the FISVDD QoE project.

## 📁 Structure

```
configs/
├── __init__.py                  # Config registry and loader
├── base_config.py              # BaseDatasetConfig class
├── live_nflx_ii_config.py     # LIVE-Netflix-II configuration
└── lfovia_qoe_config.py       # LFOVIA QoE configuration
```

## 🔧 How It Works

Each dataset has its own configuration class that inherits from `BaseDatasetConfig`. The config defines:

1. **Dataset Identity**: Name, description
2. **Features**: Which columns to use for training
3. **Preprocessing**: How to transform features (clipping, log, inversion)
4. **File Paths**: Where to find train/test data
5. **FISVDD Parameters**: Threshold quantile, sigma method, etc.
6. **Anomaly Definition**: What constitutes "bad" QoE

## 📝 Example Configuration

```python
from .base_config import BaseDatasetConfig

class MyDatasetConfig(BaseDatasetConfig):
    # Identity
    DATASET_NAME = "MY_DATASET"
    DATASET_DESCRIPTION = "My Video QoE Dataset"
    
    # Features
    FEATURE_COLUMNS = ["feature1", "feature2", "feature3"]
    TARGET_COLUMN = "qoe_score"
    
    # Preprocessing
    CLIP_FEATURES = ["feature1"]
    LOG_TRANSFORM_FEATURES = ["feature1", "feature2"]
    INVERT_FEATURES = []  # Empty if all are "lower is worse"
    
    # Data files
    TRAIN_FILE = "MY_DATASET_train.csv"
    TEST_FILE = "MY_DATASET_test.csv"
    
    # FISVDD parameters
    THRESHOLD_QUANTILE = 0.95
    SIGMA_METHOD = "median_heuristic"
    
    @staticmethod
    def is_anomaly(y_value: float) -> int:
        \"\"\"Define anomaly threshold for this dataset.\"\"\"
        return 1 if y_value < 50 else 0
```

## 🎯 Key Configuration Parameters

### Feature Selection
- **FEATURE_COLUMNS**: List of features to use for training
- **TARGET_COLUMN**: QoE score column (for validation)

### Preprocessing
- **CLIP_FEATURES**: Features with outliers that need clipping
- **LOG_TRANSFORM_FEATURES**: Features with heavy tails (use log1p)
- **INVERT_FEATURES**: Features where "higher = better" (need inversion)

### FISVDD Tuning
- **THRESHOLD_QUANTILE**: Percentile for anomaly threshold (0.95 = 95th percentile)
  - Higher = fewer false positives, more false negatives
  - Lower = more sensitive, catches more anomalies
- **SIGMA_METHOD**: How to compute kernel width
  - `"median_heuristic"`: Automatic based on data (recommended)
  - `"fixed"`: Use predetermined value

### Incremental Learning
- **NORMAL_BUFFER_MAX**: Max normal samples to keep (500 typical)
- **REFIT_EVERY**: Refit model after N updates (100 typical)

## 🔍 Using Configurations

### In Python Scripts

```python
from configs import get_config

# Get configuration for a dataset
config = get_config("LFOVIA_QoE")

# Access parameters
print(config.FEATURE_COLUMNS)
print(config.THRESHOLD_QUANTILE)

# Get file paths
train_path = config.get_resource_path(config.TRAIN_FILE)
artifact_path = config.get_artifact_path(config.get_model_artifact_name())
```

### In Command Line Scripts

```bash
python train_fisvdd.py --dataset LFOVIA_QoE
python test_fisvdd.py --dataset LIVE_NFLX_II
```

The scripts automatically load the correct configuration.

## ➕ Adding a New Dataset

1. **Create config file**: `configs/my_dataset_config.py`
2. **Define config class**: Inherit from `BaseDatasetConfig`
3. **Register it**: Add to `DATASET_CONFIGS` in `__init__.py`
4. **Update scripts**: Add dataset choice to argument parsers

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed instructions.

## 💡 Best Practices

### Feature Selection
- Use features with `|correlation| > 0.15` with target QoE
- Combine streaming metrics with visual quality metrics
- Avoid highly correlated features (correlation > 0.9)

### Preprocessing
- **Always clip** streaming metrics (rebuffering, bitrate) to avoid outliers
- **Use log transform** for heavy-tailed distributions
- **Invert features** where needed for semantic alignment

### Threshold Tuning
- Start with `THRESHOLD_QUANTILE = 0.95`
- Increase if too many false positives
- Decrease if missing too many anomalies
- Monitor flag rate (10-30% is typical)

## 📊 Current Dataset Summary

| Dataset | Features | AUC | Samples | Notes |
|---------|----------|-----|---------|-------|
| LIVE_NFLX_II | 6 | 0.74 | 462 test | VMAF-based metrics |
| LFOVIA_QoE | 4 | 0.80 | 360 test | Streaming + visual quality |

---

For more information, see:
- [BaseDatasetConfig](base_config.py) - Base class documentation
- [MULTI_DATASET_GUIDE.md](../MULTI_DATASET_GUIDE.md) - Multi-dataset workflow
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Adding new datasets
