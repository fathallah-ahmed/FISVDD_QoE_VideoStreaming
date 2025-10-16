# benchmark_fisvdd.py
import os, time, numpy as np, pandas as pd, joblib
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, roc_curve, precision_recall_curve
)
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from common_features import transform_df, FEATS
from fisvdd import fisvdd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR  = os.path.join(BASE_DIR, "resources")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
WIN_CSV  = os.path.join(RES_DIR, "LIVE_NFLX_II_windows_minimal.csv")
ART      = os.path.join(BASE_DIR, "fisvdd_artifacts.joblib")

os.makedirs(DOCS_DIR, exist_ok=True)

# --- helpers ---
def score_batch(model, X):
    out = np.empty(len(X))
    for i, x in enumerate(X):
        s, _ = model.score_fcn(x.reshape(1, -1))
        out[i] = s
    return out

def video_level_labels(df):
    g = df.groupby("file")["QoE_win"].mean()
    return (g <= 0.0).astype(int)  # 1 = bad

# --- load data & artifacts ---
df = pd.read_csv(WIN_CSV)
A  = joblib.load(ART)
scaler, sigma = A["scaler"], A["sigma"]

# preprocess (clip+log1p) and pick features
dfX   = transform_df(df)
X_all = scaler.transform(dfX[FEATS].astype(float).values)
y_all = (dfX["QoE_win"].values <= 0.0).astype(int)  # 1=bad
groups = dfX["content"].astype(str).values          # group by content

# --- K-fold by content ---
K = 5 if len(np.unique(groups)) >= 5 else max(2, len(np.unique(groups)))
gkf = GroupKFold(n_splits=K)

fold_rows = []
t0_all = time.time()

last_fold_cache = None  # we’ll keep last fold data for plots

for k, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups=groups), 1):
    # train only on GOOD windows from the training groups
    good_mask = (y_all[train_idx] == 0)
    X_train = X_all[train_idx][good_mask]
    X_test  = X_all[test_idx]
    y_test  = y_all[test_idx]

    # fit fisvdd on training-good
    t0 = time.time()
    model = fisvdd(X_train, sigma)
    model.find_sv()
    t_train = time.time() - t0

    # score test
    t0 = time.time()
    scores = score_batch(model, X_test)   # higher = more abnormal
    t_score = time.time() - t0

    # metrics independent of threshold
    auc = roc_auc_score(y_test, scores)
    ap  = average_precision_score(y_test, scores)

    # operating threshold from train-good (95th pct)
    s_train = score_batch(model, X_train)
    tau = float(np.quantile(s_train, 0.95))

    # predictions at operating point
    pred_bad = (scores > tau).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred_bad, average="binary", zero_division=0)
    flag_rate = pred_bad.mean()

    fold_rows.append({
        "fold": k, "n_train": len(X_train), "n_test": len(X_test),
        "AUC": auc, "AP": ap, "precision": prec, "recall": rec, "F1": f1,
        "flag_rate": flag_rate, "tau": tau, "t_train_s": t_train, "t_score_s": t_score
    })

    # keep the last fold’s state for plots
    last_fold_cache = {
        "X_train": X_train, "X_test": X_test, "y_test": y_test,
        "scores": scores, "tau": tau, "model": model,
        "test_idx": test_idx
    }

res = pd.DataFrame(fold_rows)
print("\n=== Window-level K-fold ===")
print(res[["fold","n_train","n_test","AUC","AP","precision","recall","F1","flag_rate","tau"]].to_string(index=False))

print("\nMEAN ± STD (AUC, AP, F1, flag_rate):")
for m in ["AUC","AP","F1","flag_rate"]:
    print(f"{m:10s} {res[m].mean():.3f} ± {res[m].std():.3f}")

print(f"\nTiming: train {res['t_train_s'].mean():.4f}s avg | score {res['t_score_s'].mean():.4f}s avg | folds={K} | total {time.time()-t0_all:.2f}s")

# --- Per-content breakdown on the last fold ---
lf = last_fold_cache
model = lf["model"]
scores_last = lf["scores"]
df_last = dfX.iloc[lf["test_idx"]].copy()
df_last["score"] = scores_last
df_last["bad"]   = lf["y_test"]

print("\n=== Per-content AUC (last fold) ===")
content_aucs = []
for c in sorted(df_last["content"].unique()):
    m = (df_last["content"] == c)
    if m.sum() >= 5:
        auc_c = roc_auc_score(df_last.loc[m, "bad"], df_last.loc[m, "score"])
        content_aucs.append((c, auc_c, int(m.sum())))
        print(f"{c:35s} AUC={auc_c:.3f}  n={m.sum()}")

# --- Video-level metric (aggregate windows per file) ---
agg = df_last.groupby("file").agg(
    file_bad=("bad","mean"),
    score_p95=("score", lambda s: np.quantile(s, 0.95)),
    score_mean=("score","mean")
).reset_index()
vid_truth = video_level_labels(dfX.iloc[lf["test_idx"]][["file","QoE_win"]])
agg = agg.merge(vid_truth.rename("video_bad"), left_on="file", right_index=True)
auc_vid = roc_auc_score(agg["video_bad"].values, agg["score_p95"].values)
print(f"\n=== Video-level AUC (last fold) ===\nAUC={auc_vid:.3f}  (scored by file p95)")

# ==========================================================
# --------------------  PLOTS  -----------------------------
# ==========================================================

# 1) ROC curve (last fold)
fpr, tpr, _ = roc_curve(lf["y_test"], lf["scores"])
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (last fold)")
plt.tight_layout()
plt.savefig(os.path.join(DOCS_DIR, "roc_curve.png"), dpi=200)
plt.close()

# 2) Precision–Recall curve (last fold)
prec, rec, _ = precision_recall_curve(lf["y_test"], lf["scores"])
plt.figure(figsize=(6,5))
plt.plot(rec, prec, linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (last fold)")
plt.tight_layout()
plt.savefig(os.path.join(DOCS_DIR, "pr_curve.png"), dpi=200)
plt.close()

# 3) Per-content AUC bar chart
if content_aucs:
    names = [c for c,_,_ in content_aucs]
    aucs  = [a for _,a,_ in content_aucs]
    plt.figure(figsize=(10,5))
    plt.bar(range(len(names)), aucs)
    plt.xticks(range(len(names)), names, rotation=60, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("AUC (last fold)")
    plt.title("Per-content AUC")
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, "per_content_auc.png"), dpi=200)
    plt.close()

# 4) PCA 2-D view of train data with Support Vectors (last fold)
#    (analogue of the author's support-vector visualization)
pca = PCA(n_components=2, random_state=0)
Xtr_2d = pca.fit_transform(lf["X_train"])
# project SVs (model.sv are in scaled feature space already)
SV_2d  = pca.transform(model.sv)

plt.figure(figsize=(6.5,5.5))
plt.scatter(Xtr_2d[:,0], Xtr_2d[:,1], s=10, alpha=0.6, label="Train (good)")
plt.scatter(SV_2d[:,0],  SV_2d[:,1],  marker="*", s=60, label="Support Vectors")
plt.legend()
plt.title("PCA of Train Data with Support Vectors (last fold)")
plt.tight_layout()
plt.savefig(os.path.join(DOCS_DIR, "pca_support_vectors.png"), dpi=200)
plt.close()

print(f"\nSaved plots to: {DOCS_DIR}")
print(" - roc_curve.png")
print(" - pr_curve.png")
print(" - per_content_auc.png")
print(" - pca_support_vectors.png")
