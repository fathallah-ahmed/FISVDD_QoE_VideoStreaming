# test_fisvdd.py
import os, numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score, average_precision_score
from common_features import transform_df, FEATS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR  = os.path.join(BASE_DIR, "resources")
ARTIFACTS = os.path.join(BASE_DIR, "fisvdd_artifacts.joblib")

WIN_CSV = os.path.join(RES_DIR, "LIVE_NFLX_II_windows_minimal.csv")

A = joblib.load(ARTIFACTS)
FEATS_art, scaler, model, tau = A["features"], A["scaler"], A["model"], A["threshold"]

# 1) load + PREPROCESS (clip + log1p), then select FEATS
df = pd.read_csv(WIN_CSV)
dfX = transform_df(df)
# hold-out contents (last 3 by name; change as needed)
contents = dfX["content"].dropna().unique()
test_contents = set(contents[-3:])
test = dfX[dfX["content"].isin(test_contents)].copy()

X = scaler.transform(test[FEATS].astype(float).values)
y = test["QoE_win"].astype(float).values

def score_batch(m, Z):
    s = np.empty(len(Z))
    for i, z in enumerate(Z):
        s[i], _ = m.score_fcn(z.reshape(1, -1))
    return s

scores = score_batch(model, X)                 # higher = more abnormal
auc = roc_auc_score((y <= 0).astype(int), scores)
ap  = average_precision_score((y <= 0).astype(int), scores)

print(f"[TEST] contents={len(test_contents)} rows={len(test)} | AUC={auc:.3f} AP={ap:.3f}")
print(f"[TEST] threshold τ={tau:.4f} flags={(scores > tau).mean()*100:.1f}%")

# --- Optional: tune τ to target flag rate (e.g., 5%) and save back to artifacts ---
target_rate = 0.10
tau_tuned = float(np.quantile(scores, 1 - target_rate))
flag_rate = (scores > tau_tuned).mean()
print(f"[TUNE] τ→{tau_tuned:.4f} for target {target_rate*100:.1f}% flags; actual={flag_rate*100:.1f}%")

A["threshold"] = tau_tuned
joblib.dump(A, ARTIFACTS)
print("[TUNE] Saved tuned τ into fisvdd_artifacts.joblib")
