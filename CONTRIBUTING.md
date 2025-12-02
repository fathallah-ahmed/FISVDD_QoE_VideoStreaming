# Contributing to FISVDD-QoE

Thank you for your interest in contributing! This document provides guidelines for adding new datasets, improving the codebase, and submitting contributions.

## 🎯 Ways to Contribute

- **Add new video QoE datasets**
- **Improve feature engineering**
- **Enhance documentation**
- **Report bugs or issues**
- **Optimize performance**
- **Add new evaluation metrics**

---

## 📊 Adding a New Dataset

### Step 1: Prepare Dataset Files

1. Create directory: `resources/YOUR_DATASET_NAME/`
2. Place raw data files (`.mat`, `.csv`, or other formats)

### Step 2: Explore the Data

Create an exploration script in `scripts/exploration/`:

```python
# scripts/exploration/explore_your_dataset.py
from scipy.io import loadmat
import pandas as pd

# Load and examine your data
# Identify available features
# Compute correlations with QoE scores
# Determine train/test split strategy
```

### Step 3: Convert to CSV Format

Your CSV files must include:
- **Feature columns**: Numerical features for model training
- **Target column**: QoE scores or quality ratings (continuous values)
- **content column** (recommended): Video/content ID for group k-fold validation
- **file column** (optional): Source filename for tracking

Example CSV structure:
```csv
feature1,feature2,feature3,qoe_score,content,file
0.52,15.3,0.95,78.2,video_01,session_001.mat
```

### Step 4: Create Dataset Configuration

Create `configs/your_dataset_config.py`:

```python
from .base_config import BaseDatasetConfig

class YourDatasetConfig(BaseDatasetConfig):
    """Configuration for Your Dataset."""
    
    # Dataset identification
    DATASET_NAME = "YOUR_DATASET"
    DATASET_DESCRIPTION = "Description of your dataset"
    
    # Features  
    FEATURE_COLUMNS = [
        "feature1",  # Description
        "feature2",  # Description
        "feature3"   # Description
    ]
    
    TARGET_COLUMN = "qoe_score"
    
    # Preprocessing
    CLIP_FEATURES = ["feature1"]  # Features to clip outliers
    LOG_TRANSFORM_FEATURES = ["feature2"]  # Features needing log transform
    INVERT_FEATURES = ["feature3"]  # Features where higher = worse
    
    # Data files
    TRAIN_FILE = "YOUR_DATASET_train.csv"
    TEST_FILE = "YOUR_DATASET_test.csv"
    
    # FISVDD parameters (tune these)
    THRESHOLD_QUANTILE = 0.95
    SIGMA_METHOD = "median_heuristic"
    
    @staticmethod
    def is_anomaly(y_value: float) -> int:
        """Define what constitutes an anomaly for your dataset."""
        # Example: QoE score < 50 on a 0-100 scale
        return 1 if y_value < 50 else 0
```

### Step 5: Register the Dataset

Add to `configs/__init__.py`:

```python
from .your_dataset_config import YourDatasetConfig

DATASET_CONFIGS = {
    "LIVE_NFLX_II": LiveNflxIIConfig,
    "LFOVIA_QoE": LfoviaQoeConfig,
    "YOUR_DATASET": YourDatasetConfig,  # Add this line
}
```

### Step 6: Update Scripts

Add your dataset to the choices in:
- `train_fisvdd.py`
- `test_fisvdd.py`
- `benchmark_fisvdd.py`

```python
parser.add_argument(
    "--dataset",
    choices=["LIVE_NFLX_II", "LFOVIA_QoE", "YOUR_DATASET"],
    ...
)
```

### Step 7: Train and Evaluate

```bash
python train_fisvdd.py --dataset YOUR_DATASET
python test_fisvdd.py --dataset YOUR_DATASET
python benchmark_fisvdd.py --dataset YOUR_DATASET
```

---

## 🔬 Feature Engineering Guidelines

### Feature Selection Criteria

1. **Correlation Analysis**: Include features with `|correlation| > 0.2` with QoE target
2. **Domain Relevance**: Prioritize streaming-specific metrics (rebuffering, stalls, bitrate)
3. **Statistical Significance**: Verify features have different distributions for normal vs anomaly classes
4. **Independence**: Avoid highly correlated features (correlation > 0.9)

### Preprocessing Best Practices

**Clipping**: Use for features with outliers
```python
CLIP_FEATURES = ["bitrate", "stall_count"]
```

**Log Transform**: Use for heavy-tailed or exponential distributions
```python
LOG_TRANSFORM_FEATURES = ["rebuffer_duration", "stall_ratio"]
```

**Inversion**: Use when "higher values = worse quality"
```python
INVERT_FEATURES = ["time_since_last_stall"]  # Higher TSL = better quality
```

### Testing Feature Combinations

Use the analysis script to test different feature sets:

```bash
python scripts/exploration/analyze_your_features.py
```

Compare AUC scores across feature combinations and choose the best performing set.

---

## 🧪 Testing Requirements

Before submitting changes:

1. **Run existing tests**:
   ```bash
   pytest tests/ -v
   ```

2. **Test both datasets**:
   ```bash
   python test_fisvdd.py --dataset LIVE_NFLX_II
   python test_fisvdd.py --dataset LFOVIA_QoE
   ```

3. **Verify no regression**: Ensure existing dataset performance doesn't decrease

---

## 📝 Code Style

- **Docstrings**: Use Google-style docstrings for all functions/classes
- **Type Hints**: Add type hints for function parameters and returns
- **Comments**: Explain *why*, not *what* (code should be self-explanatory)
- **Naming**: Use descriptive variable names (e.g., `threshold_quantile` not `tq`)

Example:
```python
def compute_threshold(scores: np.ndarray, quantile: float = 0.95) -> float:
    """
    Compute anomaly threshold from training scores.
    
    Args:
        scores: Array of anomaly scores from training data
        quantile: Percentile for threshold (0-1)
        
    Returns:
        Threshold value for anomaly detection
    """
    return float(np.quantile(scores, quantile))
```

---

## 📚 Documentation Standards

### README Updates

When adding a dataset, update the main `README.md`:
- Add dataset badge
- Include dataset description
- Show performance metrics
- Add citation information

### Configuration Documentation

Document all config parameters with inline comments:

```python
THRESHOLD_QUANTILE = 0.95  # Higher = fewer false positives, more false negatives
SIGMA_METHOD = "median_heuristic"  # Automatic kernel width selection
```

---

## 🐛 Reporting Issues

When reporting bugs, include:

1. **Environment**: OS, Python version, package versions
2. **Dataset**: Which dataset you're using
3. **Command**: Exact command that produces the error
4. **Error Message**: Full stack trace
5. **Expected Behavior**: What you expected to happen

---

## 🚀 Submitting Changes

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/your-dataset`
3. **Make changes** following the guidelines above
4. **Test thoroughly** with both datasets
5. **Commit** with descriptive messages:
   ```
   Add DATASET_NAME support with feature engineering
   
   - Created config for DATASET_NAME
   - Added exploration scripts
   - Updated documentation
   - Achieved AUC of X.XX on benchmark
   ```
6. **Push** and create a Pull Request

---

## 💡 Tips for Success

- **Start Small**: Test with a subset of data first
- **Iterate**: Try different feature combinations and compare results
- **Document**: Keep notes on what works and what doesn't
- **Ask Questions**: Open an issue if you need help

---

## 📧 Contact

For questions or discussions:
- Open an issue on GitHub
- Or contact: [Your contact method]

Thank you for contributing to FISVDD-QoE! 🎉
