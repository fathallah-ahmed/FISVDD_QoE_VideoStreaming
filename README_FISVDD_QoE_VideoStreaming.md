# FISVDD for QoE Anomaly Detection — Notebook Documentation

**Student Assignment** · *Fast Incremental Support Vector Data Description (FISVDD) applied to video streaming Quality of Experience (QoE).*  
Notebook: `FISVDD_QoE_VideoStreaming.ipynb` · Code: `fisvdd.py` · Sample data: `Dataset_VideoStreaming_Example.csv`

---

## 1) Objective

This notebook demonstrates how to **train and use an incremental one‑class model (FISVDD)** to detect **anomalous QoE sessions** in video streaming telemetry. The workflow covers: loading the demo dataset, training an unsupervised boundary of “normal” behaviour, visualizing the learned boundary in 2D, and exposing a tiny **FastAPI** service to score incoming feature vectors in real time.

**Why FISVDD?** Support Vector Data Description (SVDD) encloses normal data in a hypersphere (or kernelized contour). **FISVDD** extends this with fast **incremental updates**: as each new point arrives, the support‑vector set is **expanded** or **shrunk** in \(\mathcal{O}(k^2)\) time (with \(k\) current SVs) using rank‑one updates of the inverse kernel matrix.

---

## 2) Repository Layout (as used in this notebook)

```
.
├── FISVDD_QoE_VideoStreaming.ipynb   # The demo notebook (training + API)
├── fisvdd.py                         # FISVDD implementation (class `fisvdd`)
├── Dataset_VideoStreaming_Example.csv# Sample telemetry (5000 rows)
└── models/                           # Saved model state (created by the notebook)
```

> Ensure `fisvdd.py` is in the **same folder** as the notebook.

---

## 3) Dataset Summary

- **Rows:** 5000  
- **Columns:** 11 (9 numeric features + 1 target-like score `MOS` + 1 categorical `network_type`)

### 3.1 Columns & Types

| Column | Dtype | #Unique | Example |
|---|---:|---:|---|
| `throughput_kbps` | float64 | 4999 | 2191.903009656064 |
| `startup_delay_ms` | float64 | 4858 | 55.618601720247256 |
| `rebuffer_count` | int64 | 3 | 0 |
| `rebuffer_duration_s` | float64 | 572 | 0.0 |
| `bitrate_kbps` | float64 | 4999 | 1763.6165690192151 |
| `resolution_height` | int64 | 4 | 720 |
| `dropped_frames_pct` | float64 | 4532 | 0.3949596527487133 |
| `latency_ms` | float64 | 4860 | 72.9391214485284 |
| `buffer_health_s` | float64 | 4883 | 25.27638784917697 |
| `MOS` | float64 | 4977 | 4.5770075431797865 |
| `network_type` | object | 3 | 4g |

### 3.2 Feature semantics (short)

- **`throughput_kbps`** — Estimated available network throughput (kilobits per second).
- **`startup_delay_ms`** — Initial startup/join latency before playback begins (milliseconds).
- **`rebuffer_count`** — Count of buffering stalls during session.
- **`rebuffer_duration_s`** — Total time spent rebuffering (seconds).
- **`bitrate_kbps`** — Encoded / selected video bitrate (kilobits per second).
- **`resolution_height`** — Video vertical resolution (e.g., 720, 1080, 1440).
- **`dropped_frames_pct`** — Percentage of frames dropped by the renderer/decoder.
- **`latency_ms`** — Playback end‑to‑end latency (milliseconds).
- **`buffer_health_s`** — Seconds of media currently in the buffer.
- **`MOS`** — Mean Opinion Score (1–5): higher is better (not used directly for unsupervised training).
- **`network_type`** — Access type (e.g., `4g`, `5g`, `wifi`).

**Notes.**
- In unsupervised SVDD/FISVDD we typically **train on normal data only** (e.g., sessions with high `MOS`). In this demo, we keep all rows to simplify, but you can **filter by a MOS threshold** (e.g., `MOS ≥ 4.2`) to obtain a cleaner “normal” training set.
- Because Gaussian kernels use Euclidean distances, **feature scaling matters**. For a quick start, the demo works “as is”; for a more robust model, consider z‑scaling the numeric features before training and keep the same scaler for inference.

---

## 4) FISVDD Algorithm (intuitive)

Given support vectors (SVs) and their kernel matrix \(A\), the model maintains \(A^{-1}\), coefficients \(\alpha\), and a **score** (radius proxy). For each new point \(x\):

1. **Score:** compute kernel similarities \(v = K(x, SVs)\) (RBF with bandwidth `sigma`), then evaluate an acceptance score.
2. **Accept/Reject:** if the point lies **outside** the boundary (positive score), attempt to **expand** the SV set with \(x\) using a rank‑one **up‑date** of \(A^{-1}\).
3. **Repair:** if any \(\alpha\) becomes negative, iteratively **shrink** (remove) the most offending SVs and **down‑date** \(A^{-1}\) until all \(\alpha > 0\).
4. **Normalize:** update the global **score** and renormalize \(\alpha\).

> The notebook imports a minimal numpy/scipy implementation from `fisvdd.py` (class `fisvdd` with methods: `find_sv`, `expand`, `shrink`, `score_fcn`, `up_inv`, `down_inv`, `model_update`).

---

## 5) How to Run the Notebook

### 5.1 Environment
The first cell installs the only runtime deps used in‑notebook (NumPy, SciPy, pandas, matplotlib, FastAPI, Uvicorn, etc.). Run the cells in order.

### 5.2 Paths
The notebook creates a `models/` folder next to itself for saved artifacts.

### 5.3 Training (Section “Train FISVDD”)
1. **Select features:** use the 9 numeric telemetry fields (exclude `MOS` and `network_type`).
2. **Choose `sigma`:** start with a **median heuristic** over pairwise distances of a small sample (e.g., 1024 points). Example:
   ```python
   import numpy as np
   X = train_numeric.values
   idx = np.random.default_rng(0).choice(len(X), size=min(1024, len(X)), replace=False)
   D = np.linalg.norm(X[idx, None, :] - X[None, idx, :], axis=2)
   sigma = np.median(D[D>0])
   ```
3. **Initialize & fit:**
   ```python
   from fisvdd import fisvdd

   model = fisvdd(data=X, sigma=sigma, eps_cp=1e-8, eps_ol=1e-8)
   model.find_sv()  # single‑pass incremental training
   ```
4. **Decision rule:** the minimal demo uses the model’s internal score test (positive “outside” ⇒ anomaly). For production, hold out a validation set and **tune a threshold** on the acceptance score to meet a target false‑positive rate.

### 5.4 Visualization (PCA 2D)
The notebook projects features to 2D (PCA) to visually inspect which points get accepted/rejected by the boundary. This is qualitative only.

### 5.5 Save/Load
The notebook pickles the learned state into `models/` and reloads it for the API.

---

## 6) Minimal API (FastAPI + Uvicorn)

The notebook spins up an in‑process API with two routes:

- `GET /status` → small health/metadata payload (e.g., number of SVs).  
- `POST /predict` → body:
  ```json
  {
    "features": {
      "throughput_kbps": 5200,
      "startup_delay_ms": 300,
      "rebuffer_count": 0,
      "rebuffer_duration_s": 0.0,
      "bitrate_kbps": 3800,
      "resolution_height": 1080,
      "dropped_frames_pct": 0.3,
      "latency_ms": 90,
      "buffer_health_s": 22.0
    },
    "learn": false
  }
  ```
  **Response** (example):
  ```json
  {
    "accepted": true,          // inside the boundary (normal) if True
    "score": -0.013,           // ≤ 0 means inside; > 0 means outlier (demo rule)
    "support_vectors": 187     // current SV count
  }
  ```

**Important:** If you send `"learn": true` for a clearly bad sample, the model will **adapt in the wrong direction** (it will try to enclose that outlier). Keep `"learn": false` in scoring, and only enable learning for verified‑good points.

---

## 7) Evaluation Ideas (optional for the assignment)

- **Train/Test split:** Train on high‑quality sessions (e.g., `MOS ≥ 4.2`) and evaluate on both high‑ and low‑MOS sets; report Precision/Recall vs threshold.  
- **Ablation:** Compare with a non‑incremental one‑class baseline (e.g., scikit‑learn `OneClassSVM`) using the same features & scaling.  
- **Stability:** Plot SV count vs number of processed points; discuss sensitivity to `sigma` and scaling.  
- **Latency:** On your hardware, report per‑sample scoring latency (mean/95th percentile).

---

## 8) Reproducibility & Suggested Defaults

- **Randomness:** fix `np.random.seed(0)` before sampling for the median heuristic.  
- **Scaling:** standardize numeric features with `mean=0, std=1` learned on the training set; persist the scaler.  
- **Sigma:** median pairwise distance on a 512–2048 sample is a robust starting point.  
- **Stopping:** a single pass (`find_sv`) is often enough on stationary normal‑only data; use a sliding window if drift is expected.

---

## 9) Known Limitations

- **Scale‑sensitivity:** RBF distances magnify unscaled features. Always align units or z‑scale.  
- **Concept drift:** If real‑world traffic shifts, the boundary may become stale; use periodic re‑training or calibrated online learning.  
- **Label‑free:** Without labels, threshold selection for “anomaly” vs “normal” requires a small, trusted validation set or domain rules.

---

## 10) Appendix

### A. Feature order (as used by the API)

```python
FEATURES = [
  "throughput_kbps", "startup_delay_ms", "rebuffer_count",
  "rebuffer_duration_s", "bitrate_kbps", "resolution_height",
  "dropped_frames_pct", "latency_ms", "buffer_health_s"
]
```

### B. Example Python client

```python
import requests

x = {
  "throughput_kbps": 5200,
  "startup_delay_ms": 300,
  "rebuffer_count": 0,
  "rebuffer_duration_s": 0.0,
  "bitrate_kbps": 3800,
  "resolution_height": 1080,
  "dropped_frames_pct": 0.3,
  "latency_ms": 90,
  "buffer_health_s": 22.0
}

r = requests.post("http://127.0.0.1:8000/predict", json={"features": x, "learn": False})
print(r.json())
```

### C. Mermaid (flow overview)

```mermaid
flowchart LR
  A[Raw QoE CSV] --> B[Select numeric features]
  B --> C[Scale (optional)]
  C --> D[Choose sigma (median heuristic)]
  D --> E[Train FISVDD (incremental)]
  E --> F[Save model]
  F --> G[FastAPI /predict]
  G --> H[Client scoring]
```

---

*Prepared by: Student — FISVDD QoE Mini‑Project.*