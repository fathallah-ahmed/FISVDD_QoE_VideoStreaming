# app.py
import os, numpy as np, joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fisvdd import fisvdd
from common_features import transform_row

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(BASE_DIR, "fisvdd_artifacts.joblib")

A = joblib.load(ARTIFACTS)
FEATS     = A["features"]
scaler    = A["scaler"]
sigma     = A["sigma"]
threshold = A["threshold"]
model     = A["model"]

# incremental state (rolling normal buffer and periodic refit)
NORMAL_BUFFER_MAX = 4000
REFIT_EVERY       = 500
EWMA_ALPHA        = 0.10

ACCUM_NORMALS = []
seen_since_refit = 0

app = FastAPI(title="FISVDD QoE Online")

class Window(BaseModel):
    vmaf_mean: float
    vmaf_std: float
    vmaf_mad: float
    bitrate_mean: float
    stall_ratio: float
    tsl_end: float

def score_point(m, xz):
    s, _ = m.score_fcn(xz.reshape(1, -1))
    return float(s)

@app.post("/score")
def score(w: Window):
    global model, threshold, ACCUM_NORMALS, seen_since_refit

    raw = [w.vmaf_mean, w.vmaf_std, w.vmaf_mad, w.bitrate_mean, w.stall_ratio, w.tsl_end]
    x = transform_row(raw).reshape(1, -1)  # clip + log1p, same as train/test
    xz = scaler.transform(x)
    s = score_point(model, xz[0])
    MARGIN = 0.02
    is_anom = bool(s > threshold+MARGIN)

    # update only on normal
    if not is_anom:
        # expand with similarity vector from score_fcn
        s_now, sim_vec = model.score_fcn(xz.reshape(1, -1))
        model.expand(xz.reshape(1, -1), sim_vec)
        if model.alpha.min() < 0:
            backup = model.shrink()
            for b in backup:
                sb, sim_b = model.score_fcn(b.reshape(1, -1))
                if sb > 0:
                    model.expand(b.reshape(1, -1), sim_b)
        model.model_update()

        # buffer + light threshold adaptation
        ACCUM_NORMALS.append(xz[0])
        if len(ACCUM_NORMALS) > NORMAL_BUFFER_MAX:
            ACCUM_NORMALS = ACCUM_NORMALS[-NORMAL_BUFFER_MAX:]
        threshold = float((1 - EWMA_ALPHA) * threshold + EWMA_ALPHA * s)

        # periodic refit from buffer (keeps model compact)
        seen_since_refit += 1
        if seen_since_refit >= REFIT_EVERY and len(ACCUM_NORMALS) > 50:
            Xref = np.vstack(ACCUM_NORMALS)
            model = fisvdd(Xref, sigma)
            model.find_sv()
            # refresh threshold from recent normals (95th percentile)
            sc = [score_point(model, xi) for xi in Xref[:min(2000, len(Xref))]]
            threshold = float(np.quantile(sc, 0.95))
            seen_since_refit = 0

    return {"anomaly_score": s, "is_anomaly": is_anom, "threshold": threshold}
