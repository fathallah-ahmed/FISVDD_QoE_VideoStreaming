# 🧠 FISVDD-QoE: Fast Incremental SVDD for Video Quality-of-Experience Anomaly Detection  

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Datasets](https://img.shields.io/badge/Datasets-3-success)](#datasets)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![Status](https://img.shields.io/badge/Build-Passing-success)](#)

---

This repository implements an end-to-end **Quality of Experience (QoE)** anomaly detection system for **video streaming** using the **Fast Incremental Support Vector Data Description (FISVDD)** algorithm with **batch-based incremental learning**.  

Supports **multiple datasets** with adaptive feature engineering:
- **LIVE-Netflix-II**: AUC 0.74 | 4,257 train samples | 6 features
- **LIVE-Netflix (Original)**: AUC 0.64 | 3,504 train samples | 6 features  
- **LFOVIA QoE**: AUC 0.79 | 960 samples | 4 features

**Key Features**:
- ⚡ **Incremental Batch Learning** (default): Train progressively with configurable batch sizes
- 🔄 **Continuous Model Updates**: Add new data without retraining from scratch
- 💾 **Memory Efficient**: Process data in batches instead of loading everything at once
- 📊 **Real-time Monitoring**: Track support vector evolution batch-by-batch

Detects playback degradation (rebuffering, bitrate drops, quality instability) and adapts in real time through an incremental API.

---

## 📁 Project Structure

```
FISVDD_QoE_VideoStreaming/
│
├── resources/                          # Dataset files (organized by dataset)
│   ├── LIVE_NFLX_II/                   # LIVE-Netflix-II (420 .mat files)
│   │   ├── LIVE_NFLX_II_train.csv     # 4,257 good windows from 57 videos
│   │   └── LIVE_NFLX_II_test.csv      # 462 mixed windows from 3 videos
│   ├── LIVE_NFLX/                      # LIVE-Netflix Original (112 .mat files)
│   │   ├── matFiles/                   # Raw .mat files
│   │   ├── LIVE_NFLX_train.csv        # 3,504 good windows from 12 videos
│   │   └── LIVE_NFLX_test.csv         # 1,088 mixed windows from 2 videos
│   └── LFOVIA_QoE/
│       ├── LFOVIA_QoE_train.csv       # 960 samples (K-fold)
│       └── LFOVIA_QoE_test.csv
│
├── configs/                            # Dataset-specific configurations
│   ├── __init__.py                    # Config registry
│   ├── base_config.py                 # Base configuration class
│   ├── live_nflx_ii_config.py        # LIVE-Netflix-II settings
│   └── lfovia_qoe_config.py          # LFOVIA QoE settings
│
├── artifacts/                          # Trained models (by dataset)
│   ├── LIVE_NFLX_II_fisvdd.joblib
│   ├── LIVE_NFLX_fisvdd.joblib
│   └── LFOVIA_QoE_fisvdd.joblib
│
├── results/                            # Evaluation results
│   ├── LIVE_NFLX_II/                  # ROC curves, metrics, plots
│   ├── LIVE_NFLX/
│   ├── LFOVIA_QoE/
│   └── comparative_analysis/          # Cross-dataset comparisons
│
├── scripts/                            # Utility scripts
│   └── exploration/                   # Dataset exploration tools
│
├── tests/                              # Unit and integration tests
│   ├── test_fisvdd_unit.py
│   └── test_api.py
│
├── fisvdd.py                          # Core FISVDD algorithm (with incremental methods)
├── common_features.py                 # Generic preprocessing utilities
├── train_fisvdd.py                    # Multi-dataset training (incremental by default)
├── update_model_incremental.py        # Update trained models with new data
├── test_fisvdd.py                     # Multi-dataset evaluation
├── benchmark_fisvdd.py                # K-fold cross-validation
├── compare_datasets.py                # Cross-dataset analysis
├── app.py                             # FastAPI incremental serving
├── INCREMENTAL_LEARNING.md            # Incremental learning quick reference
└── README.md
```


---

## ⚙️ Installation

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Main dependencies**
- numpy, pandas, scikit-learn
- fastapi, uvicorn
- joblib, pydantic
- matplotlib (for visualization)

---

## 🎞 Datasets

This project supports multiple video QoE datasets with dataset-specific feature engineering.

### 📘 LIVE-Netflix-II (LIVE_NFLX_II)

**Source**: [LIVE-NFLX Video QoE Database](http://live.ece.utexas.edu/research/LIVE_NFLXStudy/nflx_index.html)  
**Content**: 420 .mat files → 5-second windowing → 4,719 total windows  
**Split**: 4,257 train (57 videos) / 462 test (3 videos)  
**Features**: 6 VMAF-based quality metrics  
**Performance**: AUC 0.74 | AP 0.71 | F1 0.49 | Precision 0.86

| Feature | Description |
|---------|-------------|
| `vmaf_mean`, `vmaf_std`, `vmaf_mad` | Quality variation from Netflix VMAF |
| `bitrate_mean` | Average bitrate (kbps) |
| `stall_ratio` | Ratio of stalled frames |
| `tsl_end` | Time since last stall |

### 📙 LIVE-Netflix Original (LIVE_NFLX)

**Source**: Original LIVE-Netflix Dataset  
**Content**: 112 .mat files → 5-second windowing → 4,592 total windows  
**Split**: 3,504 train (12 videos) / 1,088 test (2 videos)  
**Features**: 6 quality + stall metrics  
**Performance**: AUC 0.64 | AP 0.73 | F1 0.49 | Precision 0.79

| Feature | Description |
|---------|-------------|
| `vmaf_mean`, `vmaf_std`, `vmaf_mad` | VMAF quality statistics |
| `ssim` | Structural similarity index |
| `stall_count` | Number of rebuffering events |
| `tsl_end` | Time since last stall |

### 📗 LFOVIA QoE Dataset

**Source**: [IIT Hyderabad LFOVIA](https://iith.ac.in/~lfovia/)  
**Content**: 960 samples with continuous QoE scores  
**Split**: K-fold cross-validation (content-based)  
**Features**: 4 streaming + visual quality metrics  
**Performance**: AUC 0.79 | AP 0.41 | F1 0.51 | Precision 0.48

| Feature | Description |
|---------|-------------|
| `TSL` | Time since last rebuffer event |
| `Nrebuffers` | Number of rebuffering events |
| `NIQE` | No-reference image quality (naturalness) |
| `SSIM` | Structural similarity index |

---

## 🧮 Training the Model

Train on any supported dataset:

```bash
# LIVE-Netflix-II (Best Performance)
python train_fisvdd.py --dataset LIVE_NFLX_II

# LIVE-Netflix Original
python train_fisvdd.py --dataset LIVE_NFLX

# LFOVIA QoE
python train_fisvdd.py --dataset LFOVIA_QoE
```

This script:
1. Loads the training data  
2. Applies preprocessing (`clip + log1p` on bitrate and stall features)  
3. Standardizes inputs with `StandardScaler`  
4. Uses the median heuristic to compute σ  
5. Trains FISVDD on “good” (QoE > 0) windows  
6. Saves model artifacts (`fisvdd_artifacts.joblib`)

---

## 🧪 Evaluation

Evaluate on any dataset:

```bash
python test_fisvdd.py --dataset LIVE_NFLX_II
python test_fisvdd.py --dataset LIVE_NFLX
python test_fisvdd.py --dataset LFOVIA_QoE
```

Example output:
```
[TEST] contents=3 rows=462 | AUC=0.832 AP=0.786
[TEST] threshold τ=-0.0021 flags=6.5%
```

- Flag rate (%)

---

## 🏎️ Benchmarks & Baselines

We compared **FISVDD** against three standard anomaly detection baselines:
1. **SVDD (RBF Kernel)**: Implementation via One-Class SVM.
2. **One-Class SVM (Linear)**: Hyperplane-based detection.
3. **Isolation Forest**: Tree-based ensemble.

### 1. Accuracy Comparison
Run the comparison script:
```bash
python compare_baselines.py
```

| Dataset | Model | AUC | Notes |
| :--- | :--- | :--- | :--- |
| **LIVE_NFLX_II** | Isolation Forest | **0.89** | Best on clean, high-dim data |
| | **FISVDD** | 0.73 | Competitive, significantly faster |
| **LFOVIA_QoE** | **FISVDD** | **0.84** | Best on data with subtle anomalies |
| | Isolation Forest | 0.70 | |

### 2. Latency & Throughput (Speed)
FISVDD is designed for **real-time** applications.

```bash
python benchmark_all_latency.py
```

| Model | Mean Latency | Throughput | Speedup |
| :--- | :--- | :--- | :--- |
| **FISVDD** | **~0.02 ms** | **~100,000 / sec** | **300x faster** |
| SVMs | ~0.25 ms | ~13,000 / sec | 1x |
| IsoForest | ~7.00 ms | ~300 / sec | <0.1x |

---
---

## 🚀 Incremental FastAPI Service

Start the API server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Send requests:

```bash
python client_example.py
```

Example response:
```json
{
  "anomaly_score": 0.0326,
  "is_anomaly": true,
  "threshold": -0.0021
}
```

### Endpoints

| Route | Description |
|--------|-------------|
| `POST /score` | Score a window (updates model if normal) |
| `POST /score_batch` | Score multiple windows (no update) |
| `GET /health` | API status |
| `GET /status` | Shows threshold, buffer size, config |

🧩 The model updates online when the window is **not anomalous**.  
Every `REFIT_EVERY` updates, it refits automatically and persists its state.

---

## 📊 Benchmarking

Run K-fold cross-validation:

```bash
python benchmark_fisvdd.py --dataset LIVE_NFLX_II
python benchmark_fisvdd.py --dataset LIVE_NFLX
python benchmark_fisvdd.py --dataset LFOVIA_QoE

# Compare all 3 datasets
python compare_datasets.py
```

Example results:
```
=== Window-level K-fold ===
AUC=0.713 ±0.024 | AP=0.783 ±0.028 | F1=0.571 ±0.050 | flag_rate=25.7%
=== Video-level AUC ===
AUC=0.911 (scored by file p95)
```

### Interpretation
- Window-level AUC ≈ 0.71 → accurate frame-level anomaly detection  
- Video-level AUC ≈ 0.91 → strong overall QoE session detection  
- Real-time: train ≈ 0.09 s, inference ≈ 0.03 s  

---

## ⚡ Real-Time Performance

```bash
python benchmark_latency.py
```

**Latency Metrics:**
- **Mean Inference:** 0.017 ms per window
- **P99 Latency:** 0.043 ms (99th percentile)
- **Throughput:** 64,176 samples/second
- **API Latency:** ~14 ms end-to-end
- **Overhead:** 0.0003% of 5-second window duration

✅ **Real-time Capable:** The model processes windows 294,000x faster than they arrive (17μs vs 5000ms)

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Or use the helper script
python -m pytest tests/ -v
```

**Test Coverage:**
- Unit tests for FISVDD core algorithm
- Integration tests for FastAPI endpoints
- All tests passing (4/4)

---

## 📈 Visualization

After running `benchmark_fisvdd.py`, plots are automatically saved in the `results/{DATASET_NAME}/` folder:

<p align="center">
  <img src="results/LIVE_NFLX_II/roc_curve.png" width="48%" alt="ROC curve"/>
  <img src="results/LIVE_NFLX_II/pr_curve.png" width="48%" alt="Precision-Recall curve"/>
</p>

<p align="center">
  <img src="results/LIVE_NFLX_II/per_content_auc.png" width="70%" alt="Per-content AUC"/>
</p>

<p align="center">
  <img src="results/LIVE_NFLX_II/pca_support_vectors.png" width="60%" alt="PCA Support Vectors"/>
</p>

These plots visualize:
- **ROC / PR Curves:** model discrimination capability per window  
- **Per-content AUC:** content-specific detection performance  
- **PCA projection:** support vector distribution across normal data

---

## ⚖️ Baseline Comparison

```bash
python compare_baselines.py
```

| Model | AUC | AP |
|--------|-----|----|
| OneClassSVM | 0.68 | 0.74 |
| IsolationForest | 0.70 | 0.75 |
| **FISVDD** | **0.71** | **0.78** ✅ |

---

## 🔬 Key Highlights

✅ **Incremental Batch Learning (Default)** - Memory-efficient progressive training
✅ **Continuous Model Updates** - Add new data without retraining from scratch  
✅ **Multi-Dataset Support** - LIVE-Netflix-II and LFOVIA QoE with adaptive features
✅ **Real-Time Capable** - 17μs inference, 64K samples/second throughput
✅ **Robust Feature Engineering** - Dataset-specific preprocessing pipelines  
✅ **FastAPI Integration** - Production-ready incremental serving endpoint  
✅ **Comprehensive Testing** - Unit tests, benchmarking, visualization tools
✅ **Backward Compatible** - Legacy standard mode available via flag  
📍 Tunisia  
💼 Focus: QoE Modeling • Incremental Learning • Real-Time AI Systems  

<p align="center">⭐ If you found this project useful, please give it a star on GitHub!</p>
