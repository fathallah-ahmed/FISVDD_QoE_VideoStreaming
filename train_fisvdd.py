# train_fisvdd.py
import os, numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from fisvdd import fisvdd                      # uses your class  :contentReference[oaicite:3]{index=3}
from common_features import transform_df, FEATS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR  = os.path.join(BASE_DIR, "resources")
ARTIFACTS = os.path.join(BASE_DIR, "fisvdd_artifacts.joblib")

TRAIN_CSV = os.path.join(RES_DIR, "LIVE_NFLX_II_FISVDD_train.csv")

# 1) load
df = pd.read_csv(TRAIN_CSV)

# 2) PREPROCESS (clip + log1p) then select features
dfX = transform_df(df)
X_raw = dfX[FEATS].astype(float).values

# 3) scale
scaler = StandardScaler().fit(X_raw)
X = scaler.transform(X_raw)

# 4) sigma (median distance heuristic)
sub = X[np.random.choice(len(X), size=min(2000, len(X)), replace=False)]
from sklearn.metrics import pairwise_distances
D = pairwise_distances(sub, sub, metric="euclidean")
sigma = float(max(1e-6, np.median(D[D > 0]) / np.sqrt(2.0)))

# 5) fit FISVDD (batch build)  :contentReference[oaicite:4]{index=4}
model = fisvdd(X, sigma)
model.find_sv()

# 6) training scores → threshold at 95th percentile by default
def score_batch(m, Z):
    s = np.empty(len(Z))
    for i, z in enumerate(Z):
        s[i], _ = m.score_fcn(z.reshape(1, -1))
    return s

train_scores = score_batch(model, X)
tau = float(np.quantile(train_scores, 0.95))

joblib.dump({
    "features": FEATS,
    "scaler": scaler,
    "sigma": sigma,
    "threshold": tau,
    "model": model
}, ARTIFACTS)

print(f"[OK] saved → {ARTIFACTS} | N={len(X)} sigma={sigma:.4f} τ={tau:.4f}")
