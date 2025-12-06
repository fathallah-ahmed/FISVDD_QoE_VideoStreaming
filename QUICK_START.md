# Quick Start Guide

Get started with FISVDD QoE anomaly detection in minutes!

---

## Prerequisites

Make sure you have Python 3.8+ installed and the virtual environment activated.

---

## 🚀 One-Time Setup

**1. Activate Virtual Environment**

Every time you open a new terminal, run:

```powershell
.venv\Scripts\activate
```

✅ You'll see `(.venv)` appear at the start of your command line.

**2. Install Dependencies** (first time only)

```bash
pip install -r requirements.txt
```

---

## 📊 Working with Datasets

This project supports **3 datasets**. Choose yours and follow the workflow:

### Option 1: LIVE-Netflix-II Dataset

```bash
# Step 1: Train the model (incremental batch learning by default)
python train_fisvdd.py --dataset LIVE_NFLX_II

# Optional: Customize batch size
python train_fisvdd.py --dataset LIVE_NFLX_II --batch-size 200

# Step 2: Test and evaluate
python test_fisvdd.py --dataset LIVE_NFLX_II

# Step 3: Run benchmark (K-Fold validation)
python benchmark_fisvdd.py --dataset LIVE_NFLX_II
```

**Expected Results:**
- AUC: ~0.74
- AP: ~0.71
- F1: ~0.42

💡 **New:** Training now uses incremental batch learning by default! See [INCREMENTAL_LEARNING.md](INCREMENTAL_LEARNING.md)

---

### Option 2: LIVE-Netflix Original Dataset

```bash
# Step 1: Train the model
python train_fisvdd.py --dataset LIVE_NFLX

# Step 2: Test and evaluate  
python test_fisvdd.py --dataset LIVE_NFLX

# Step 3: Run benchmark
python benchmark_fisvdd.py --dataset LIVE_NFLX
```

**Expected Results:**
- AUC: ~0.64
- AP: ~0.73
- F1: ~0.49

---

### Option 3: LFOVIA QoE Dataset

```bash
# Step 1: Train the model (incremental batch learning by default)
python train_fisvdd.py --dataset LFOVIA_QoE

# Step 2: Test and evaluate
python test_fisvdd.py --dataset LFOVIA_QoE

# Step 3: Run benchmark (K-Fold validation)
python benchmark_fisvdd.py --dataset LFOVIA_QoE
```

**Expected Results:**
- AUC: ~0.79
- AP: ~0.41
- F1: ~0.51

**💡 Tip:** Update models with new data using `update_model_incremental.py`

---

## 🏎️ Run Benchmarks (New)

Compare accuracy and speed against baselines:

```bash
# Compare Accuracy (AUC/AP)
python compare_baselines.py

# Benchmark Speed (Latency)
python benchmark_all_latency.py
```

---

## 🔄 Compare All 3 Datasets

After training all three models:

```bash
python compare_datasets.py
```

This generates comparative analysis in `results/comparative_analysis/`.

---

## 🌐 Run API Server (Optional)

Start the FastAPI server for real-time anomaly detection:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Test it with the client:

```bash
python client_example.py
```

---

## ❌ Common Issues

### "ModuleNotFoundError: No module named 'X'"
**Solution:** You forgot to activate the virtual environment!
```powershell
.venv\Scripts\activate
```

### "FileNotFoundError: Training file not found"
**Solution:** Make sure CSV files are in `resources/DATASET_NAME/` directory.

### "AUC: nan" or very low AUC
**Solution:** 
- Check that your test set has both normal and anomaly samples
- Verify the anomaly threshold in `configs/your_dataset_config.py`
- For LFOVIA: anomaly is score < 50

### "Threshold τ=nan"
**Solution:** Training data might be corrupted or missing. Re-run data preparation scripts.

---

## 📁 Where to Find Results

All results are saved automatically:

- **Metrics**: `results/DATASET_NAME/metrics.json`
- **Plots**: `results/DATASET_NAME/*.png`
- **Models**: `artifacts/DATASET_NAME_fisvdd.joblib`
- **Comparisons**: `results/comparative_analysis/`

---

## 💡 Quick Tips

1. **Always activate the virtual environment** before running commands
2. **Training is fast** (~5-10 seconds per dataset)
3. **Benchmarking takes longer** (~30-60 seconds due to K-fold validation)
4. **Results are automatically saved** - no need to redirect output
5. **Check metrics.json** for precise numerical results

---

## 📚 Need More Help?

- **Full Documentation**: See [README.md](README.md)
- **Multi-Dataset Guide**: See [MULTI_DATASET_GUIDE.md](MULTI_DATASET_GUIDE.md)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Exploration Scripts**: See [scripts/exploration/README.md](scripts/exploration/README.md)

---

**Ready to go!** Start with training on your chosen dataset. 🚀
