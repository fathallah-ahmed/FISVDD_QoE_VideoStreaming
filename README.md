# 🧠 FISVDD-QoE: Fast Incremental SVDD for Video Quality-of-Experience Anomaly Detection  

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Dataset](https://img.shields.io/badge/Dataset-LIVE--Netflix%20II-FF6C37?logo=netflix&logoColor=white)](http://live.ece.utexas.edu/research/LIVE_NFLXStudy/nflx_index.html)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](#citation)
[![Status](https://img.shields.io/badge/Build-Passing-success)](#)

---

This repository implements an end-to-end **Quality of Experience (QoE)** anomaly detection system for **video streaming**, trained on the **LIVE-Netflix-II dataset** using the **Fast Incremental Support Vector Data Description (FISVDD)** algorithm.  
It detects playback degradation (rebuffering, bitrate drops, quality instability) and adapts in real time through an incremental API.

---

## 📁 Project Structure

```
FISVDD_QoE_VideoStreaming/
│
├── resources/
│   ├── LIVE_NFLX_II_windows_minimal.csv      # Extracted 5-second windows
│   ├── LIVE_NFLX_II_FISVDD_train.csv         # Training subset (QoE > 0)
│
├── tests/                                    # Unit and integration tests
│   ├── test_fisvdd_unit.py                   # Core algorithm tests
│   ├── test_api.py                           # API endpoint tests
│
├── fisvdd.py                                 # Core FISVDD implementation
├── common_features.py                        # Preprocessing (log + clip transform)
├── config.py                                 # Centralized configuration
├── train_fisvdd.py                           # Offline model training
├── test_fisvdd.py                            # Evaluation and threshold tuning
├── app.py                                    # FastAPI incremental serving
├── client_example.py                         # Example client for the API
├── benchmark_fisvdd.py                       # K-fold evaluation benchmark + visualization
├── benchmark_latency.py                      # Real-time latency benchmarking
├── fisvdd_artifacts.joblib                   # Saved model and parameters
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

## 🎞 Dataset

📘 **LIVE-Netflix-II (2018)**  
http://live.ece.utexas.edu/research/LIVE_NFLXStudy/nflx_index.html  

The dataset contains **420 distorted video sequences** with subjective QoE scores.  
We extract **5-second windows** with the following features:

| Feature | Description |
|----------|-------------|
| `vmaf_mean`, `vmaf_std`, `vmaf_mad` | Quality variation from Netflix VMAF |
| `bitrate_mean` | Average bitrate (kbps) |
| `stall_ratio` | Ratio of stalled frames |
| `tsl_end` | Time since last stall |
| `QoE_win` | Z-scored subjective QoE (for validation only) |

---

## 🧮 Training the Model

```bash
python train_fisvdd.py
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

```bash
python test_fisvdd.py
```

Example output:
```
[TEST] contents=3 rows=462 | AUC=0.832 AP=0.786
[TEST] threshold τ=-0.0021 flags=6.5%
```

✅ Metrics:
- AUC (Area Under ROC)
- Average Precision (AP)
- Precision / Recall / F1
- Flag rate (%)

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

```bash
python benchmark_fisvdd.py
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

After running `benchmark_fisvdd.py`, four plots will be automatically saved in the `docs/` folder:

<p align="center">
  <img src="docs/roc_curve.png" width="48%" alt="ROC curve"/>
  <img src="docs/pr_curve.png" width="48%" alt="Precision-Recall curve"/>
</p>

<p align="center">
  <img src="docs/per_content_auc.png" width="70%" alt="Per-content AUC"/>
</p>

<p align="center">
  <img src="docs/pca_support_vectors.png" width="60%" alt="PCA Support Vectors"/>
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

✅ Fully incremental online QoE anomaly detection  
✅ Real-time adaptation without retraining  
✅ Robust feature preprocessing for streaming metrics  
✅ High video-level accuracy on LIVE-Netflix-II  
✅ FastAPI endpoint for integration with dashboards or monitoring  
✅ Automated performance visualizations  

---

## 🏫 Citation

If you use this framework or dataset:

> C. G. Bampis, Z. Li, I. Katsavounidis, T.-Y. Huang, C. Ekanadham, and A. C. Bovik,  
> *“Towards Perceptually Optimized End-to-End Adaptive Video Streaming,”*  
> IEEE Transactions on Image Processing, 2018.  
> [LIVE-Netflix Video QoE Database](http://live.ece.utexas.edu/research/LIVE_NFLXStudy)

---

## 👨‍💻 Author

**Ahmed Fathallah**  
Software & Machine Learning Engineer  
📍 Tunisia  
💼 Focus: QoE Modeling • Incremental Learning • Real-Time AI Systems  

<p align="center">⭐ If you found this project useful, please give it a star on GitHub!</p>
