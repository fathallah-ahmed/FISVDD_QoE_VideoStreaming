# app.py
import os
import numpy as np
import joblib
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fisvdd import fisvdd
from common_features import transform_row
import config

# --- Pydantic Models ---
class Window(BaseModel):
    vmaf_mean: float
    vmaf_std: float
    vmaf_mad: float
    bitrate_mean: float
    stall_ratio: float
    tsl_end: float

class ScoreResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    threshold: float

# --- Model Manager ---
class ModelManager:
    def __init__(self, artifacts_path: str):
        self.artifacts_path = artifacts_path
        self.model: Optional[fisvdd] = None
        self.scaler: Any = None
        self.threshold: float = 0.0
        self.sigma: float = 0.0
        self.accum_normals: List[np.ndarray] = []
        self.seen_since_refit: int = 0
        
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.artifacts_path):
            raise FileNotFoundError(f"Artifacts not found at {self.artifacts_path}")
            
        A = joblib.load(self.artifacts_path)
        self.scaler = A["scaler"]
        self.sigma = A["sigma"]
        self.threshold = A["threshold"]
        self.model = A["model"]
        
        # Reset incremental buffers on load/reload
        self.accum_normals = []
        self.seen_since_refit = 0
        print(f"Model loaded from {self.artifacts_path}. Threshold={self.threshold:.4f}")

    def score_point(self, xz: np.ndarray) -> float:
        s, _ = self.model.score_fcn(xz.reshape(1, -1))
        return float(s)

    def update(self, xz: np.ndarray, score: float):
        """
        Update model with a normal point (incremental learning).
        """
        # Expand
        s_now, sim_vec = self.model.score_fcn(xz.reshape(1, -1))
        self.model.expand(xz.reshape(1, -1), sim_vec)
        
        # Shrink if needed
        if self.model.alpha.min() < 0:
            backup = self.model.shrink()
            for b in backup:
                sb, sim_b = self.model.score_fcn(b.reshape(1, -1))
                if sb > 0:
                    self.model.expand(b.reshape(1, -1), sim_b)
        
        self.model.model_update()

        # Buffer & Threshold Adaptation
        self.accum_normals.append(xz)
        if len(self.accum_normals) > config.NORMAL_BUFFER_MAX:
            self.accum_normals = self.accum_normals[-config.NORMAL_BUFFER_MAX:]
            
        self.threshold = float((1 - config.EWMA_ALPHA) * self.threshold + config.EWMA_ALPHA * score)

        # Periodic Refit
        self.seen_since_refit += 1
        if self.seen_since_refit >= config.REFIT_EVERY and len(self.accum_normals) > config.MIN_REFIT_SIZE:
            self._refit()

    def _refit(self):
        """
        Refit the model using the accumulated normal buffer.
        """
        Xref = np.vstack(self.accum_normals)
        # Create fresh model
        self.model = fisvdd(Xref, self.sigma)
        self.model.find_sv()
        
        # Refresh threshold
        # Score a subset of the buffer to estimate quantile
        subset_size = min(config.REFIT_SAMPLE_SIZE, len(Xref))
        sc = [self.score_point(xi) for xi in Xref[:subset_size]]
        self.threshold = float(np.quantile(sc, config.THRESHOLD_QUANTILE))
        
        self.seen_since_refit = 0
        print(f"Model refitted. New threshold={self.threshold:.4f}")

# --- Dependency Injection ---
model_manager: Optional[ModelManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model_manager
    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_path = os.path.join(base_dir, config.ARTIFACTS_FILENAME)
    model_manager = ModelManager(artifacts_path)
    yield
    # Clean up if needed

app = FastAPI(title="FISVDD QoE Online", lifespan=lifespan)

def get_model_manager() -> ModelManager:
    if model_manager is None:
        raise RuntimeError("ModelManager is not initialized")
    return model_manager

# --- Endpoints ---

@app.post("/score", response_model=ScoreResponse)
def score(w: Window, mgr: ModelManager = Depends(get_model_manager)):
    raw = [w.vmaf_mean, w.vmaf_std, w.vmaf_mad, w.bitrate_mean, w.stall_ratio, w.tsl_end]
    x = transform_row(raw).reshape(1, -1)
    xz = mgr.scaler.transform(x)[0]
    
    s = mgr.score_point(xz)
    is_anom = bool(s > mgr.threshold + config.MARGIN)

    if not is_anom:
        mgr.update(xz, s)

    return {
        "anomaly_score": s, 
        "is_anomaly": is_anom, 
        "threshold": mgr.threshold
    }

@app.get("/status")
def status(mgr: ModelManager = Depends(get_model_manager)):
    return {
        "threshold": mgr.threshold,
        "buffer_size": len(mgr.accum_normals),
        "support_vectors": len(mgr.model.sv),
        "seen_since_refit": mgr.seen_since_refit
    }

@app.get("/health")
def health():
    return {"status": "ok"}

