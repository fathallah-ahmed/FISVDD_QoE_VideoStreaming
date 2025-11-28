# benchmark_latency.py
"""
Dedicated latency benchmarking for real-time performance evaluation.
Measures inference latency, throughput, and API response times.
"""
import os, time, numpy as np, pandas as pd, joblib
import requests
from common_features import transform_df, FEATS
from fisvdd import fisvdd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR  = os.path.join(BASE_DIR, "resources")
WIN_CSV  = os.path.join(RES_DIR, "LIVE_NFLX_II_windows_minimal.csv")
ART      = os.path.join(BASE_DIR, "fisvdd_artifacts.joblib")

print("=" * 60)
print("FISVDD Real-Time Latency Benchmark")
print("=" * 60)

# Load artifacts and data
A = joblib.load(ART)
scaler, sigma, model = A["scaler"], A["sigma"], A["model"]

df = pd.read_csv(WIN_CSV)
dfX = transform_df(df)
X_all = scaler.transform(dfX[FEATS].astype(float).values)

# Take a sample for testing
n_samples = min(1000, len(X_all))
X_sample = X_all[:n_samples]

print(f"\n📊 Testing with {n_samples} samples")
print(f"Model has {len(model.sv)} support vectors\n")

# ============================================
# 1. INFERENCE LATENCY (Model Only)
# ============================================
print("1️⃣  INFERENCE LATENCY (Model scoring only)")
print("-" * 60)

latencies = []
for i in range(n_samples):
    t0 = time.perf_counter()
    score, _ = model.score_fcn(X_sample[i].reshape(1, -1))
    latency = (time.perf_counter() - t0) * 1000  # Convert to ms
    latencies.append(latency)

latencies = np.array(latencies)

print(f"  Mean:       {latencies.mean():.3f} ms")
print(f"  Median:     {np.median(latencies):.3f} ms")
print(f"  Std Dev:    {latencies.std():.3f} ms")
print(f"  Min:        {latencies.min():.3f} ms")
print(f"  Max:        {latencies.max():.3f} ms")
print(f"  P95:        {np.percentile(latencies, 95):.3f} ms")
print(f"  P99:        {np.percentile(latencies, 99):.3f} ms")

# ============================================
# 2. THROUGHPUT
# ============================================
print("\n2️⃣  THROUGHPUT (Samples per second)")
print("-" * 60)

t0 = time.perf_counter()
for i in range(n_samples):
    model.score_fcn(X_sample[i].reshape(1, -1))
elapsed = time.perf_counter() - t0

throughput = n_samples / elapsed
print(f"  Throughput: {throughput:.1f} samples/sec")
print(f"  Total time: {elapsed:.3f} seconds for {n_samples} samples")

# ============================================
# 3. API LATENCY (End-to-End)
# ============================================
print("\n3️⃣  API LATENCY (End-to-end HTTP request)")
print("-" * 60)

API_URL = "http://localhost:8000/score"
api_latencies = []
api_success = 0

# Test with 100 API calls
n_api_tests = min(100, n_samples)

for i in range(n_api_tests):
    # Create a sample request
    row = dfX.iloc[i]
    payload = {
        "vmaf_mean": float(row["vmaf_mean"]),
        "vmaf_std": float(row["vmaf_std"]),
        "vmaf_mad": float(row["vmaf_mad"]),
        "bitrate_mean": float(row["bitrate_mean"]),
        "stall_ratio": float(row["stall_ratio"]),
        "tsl_end": float(row["tsl_end"])
    }
    
    try:
        t0 = time.perf_counter()
        response = requests.post(API_URL, json=payload, timeout=5)
        latency = (time.perf_counter() - t0) * 1000
        
        if response.status_code == 200:
            api_latencies.append(latency)
            api_success += 1
    except requests.exceptions.RequestException as e:
        print(f"  ⚠️  API not running. Start with: uvicorn app:app --reload")
        break

if len(api_latencies) > 0:
    api_latencies = np.array(api_latencies)
    print(f"  Successful requests: {api_success}/{n_api_tests}")
    print(f"  Mean:       {api_latencies.mean():.3f} ms")
    print(f"  Median:     {np.median(api_latencies):.3f} ms")
    print(f"  Std Dev:    {api_latencies.std():.3f} ms")
    print(f"  Min:        {api_latencies.min():.3f} ms")
    print(f"  Max:        {api_latencies.max():.3f} ms")
    print(f"  P95:        {np.percentile(api_latencies, 95):.3f} ms")
    print(f"  P99:        {np.percentile(api_latencies, 99):.3f} ms")

# ============================================
# 4. REAL-TIME FEASIBILITY
# ============================================
print("\n4️⃣  REAL-TIME FEASIBILITY ANALYSIS")
print("-" * 60)

window_duration_ms = 5000  # 5-second windows
mean_latency = latencies.mean()
p99_latency = np.percentile(latencies, 99)

print(f"  Window duration:     {window_duration_ms} ms (5 seconds)")
print(f"  Mean inference time: {mean_latency:.3f} ms")
print(f"  P99 inference time:  {p99_latency:.3f} ms")
print(f"  Latency overhead:    {(mean_latency/window_duration_ms)*100:.3f}%")
print(f"  Max throughput:      {throughput:.1f} windows/sec")

if mean_latency < 100:
    print(f"\n  ✅ EXCELLENT: Mean latency < 100ms (real-time capable)")
elif mean_latency < 500:
    print(f"\n  ✅ GOOD: Mean latency < 500ms (suitable for near real-time)")
else:
    print(f"\n  ⚠️  WARNING: Mean latency > 500ms (may not be real-time)")

if p99_latency < 200:
    print(f"  ✅ EXCELLENT: P99 latency < 200ms (consistent performance)")
elif p99_latency < 1000:
    print(f"  ✅ GOOD: P99 latency < 1s (acceptable tail latency)")
else:
    print(f"  ⚠️  WARNING: P99 latency > 1s (high tail latency)")

# ============================================
# 5. SUMMARY
# ============================================
print("\n" + "=" * 60)
print("📋 SUMMARY")
print("=" * 60)
print(f"  Model Type:          FISVDD (Incremental)")
print(f"  Support Vectors:     {len(model.sv)}")
print(f"  Mean Inference:      {mean_latency:.3f} ms")
print(f"  P99 Inference:       {p99_latency:.3f} ms")
print(f"  Throughput:          {throughput:.1f} samples/sec")
if len(api_latencies) > 0:
    print(f"  API Mean Latency:    {api_latencies.mean():.3f} ms")
print(f"  Real-time Capable:   {'✅ YES' if mean_latency < 100 else '⚠️  MARGINAL' if mean_latency < 500 else '❌ NO'}")
print("=" * 60)
