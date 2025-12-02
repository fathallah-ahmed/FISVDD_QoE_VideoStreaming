# Exploration Scripts

This directory contains utility scripts for exploring and preparing new datasets for the FISVDD QoE project.

## Scripts

### 📊 `analyze_lfovia_features.py`
**Purpose**: Analyze feature correlations and importance for the LFOVIA_QoE dataset.

**What it does**:
- Loads LFOVIA training data
- Computes correlations with target QoE scores
- Compares feature distributions between normal and anomaly classes
- Saves analysis report to `results/LFOVIA_QoE/analysis_report.txt`

**When to use**: When deciding which features to include in your model configuration.

```bash
python scripts/exploration/analyze_lfovia_features.py
```

---

### 🔍 `quick_explore_lfovia.py`
**Purpose**: Quick inspection of LFOVIA .mat file structure.

**What it does**:
- Loads sample .mat files from `resources/LFOVIA_QoE/`
- Prints data structure (keys, shapes, dtypes)
- Saves output to `lfovia_structure.txt`

**When to use**: When first adding the dataset to understand its structure.

```bash
python scripts/exploration/quick_explore_lfovia.py
```

---

### 🔄 `convert_lfovia_to_csv.py`
**Purpose**: Convert LFOVIA .mat files to CSV format for training.

**What it does**:
- Reads all .mat files from `resources/LFOVIA_QoE/`
- Extracts features: NIQE, PSNR, SSIM, STRRED, TSL, Nrebuffers, score_continuous
- Splits into train (80%) and test (20%) by video content
- Saves to `LFOVIA_QoE_train.csv` and `LFOVIA_QoE_test.csv`

**When to use**: 
- Initial dataset preparation
- After changing feature extraction logic
- When updating train/test splits

```bash
python scripts/exploration/convert_lfovia_to_csv.py
```

**Output**:
- `resources/LFOVIA_QoE/LFOVIA_QoE_train.csv`
- `resources/LFOVIA_QoE/LFOVIA_QoE_test.csv`

---

### 📈 `explore_lfovia_dataset.py`
**Purpose**: Comprehensive exploration and visualization of LFOVIA dataset.

**What it does**:
- Loads and summarizes all LFOVIA data
- Computes descriptive statistics
- Creates visualizations of feature distributions
- Analyzes QoE score distributions
- Identifies anomaly patterns

**When to use**: For in-depth dataset analysis before model training.

```bash  
python scripts/exploration/explore_lfovia_dataset.py
```

---

## Adding a New Dataset

To add support for a new video QoE dataset:

1. **Place raw data**: Create `resources/YOUR_DATASET/` directory
2. **Explore structure**: Use `quick_explore_*.py` as template
3. **Convert to CSV**: Create conversion script similar to `convert_lfovia_to_csv.py`
4. **Analyze features**: Create analysis script to find best features
5. **Create config**: Add `configs/your_dataset_config.py`
6. **Register dataset**: Update `configs/__init__.py`

### Required CSV Format

Your final CSV files must have:
- **Feature columns**: Numerical features for training  
- **Target column**: QoE scores or quality ratings
- **content column** (optional): Video ID for group k-fold validation
- **file column** (optional): Source filename for tracking

Example:
```csv
feature1,feature2,feature3,score,content,file
0.5,10.2,0.98,75.3,video_1,session_001.mat
```

---

## Tips

- **Feature Selection**: Start with features that have |correlation| > 0.2 with target
- **Preprocessing**: Check feature distributions - use log transform for heavy-tailed data  
- **Train/Test Split**: Use content-based splits to avoid data leakage
- **Anomaly Definition**: Define what constitutes bad QoE for your dataset in the config

---

## Questions?

See the main [MULTI_DATASET_GUIDE.md](../../MULTI_DATASET_GUIDE.md) for the complete multi-dataset workflow.
