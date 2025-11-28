import pytest
from fastapi.testclient import TestClient
from app import app, get_model_manager, ModelManager
import numpy as np
import joblib
import os

# Mock ModelManager to avoid loading real artifacts
class MockModelManager:
    def __init__(self):
        self.threshold = 0.5
        self.scaler = MockScaler()
        self.model = MockFISVDD()
        self.accum_normals = []
        self.seen_since_refit = 0

    def score_point(self, xz):
        return 0.6 # Always return a score > threshold for anomaly test

    def update(self, xz, score):
        pass

class MockScaler:
    def transform(self, x):
        return x # Identity

class MockFISVDD:
    def __init__(self):
        self.sv = np.array([[0,0]])
    def score_fcn(self, x):
        return 0.6, np.array([0.1])

@pytest.fixture
def client():
    # Override dependency
    app.dependency_overrides[get_model_manager] = lambda: MockModelManager()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_status(client):
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "threshold" in data
    assert "buffer_size" in data

def test_score_anomaly(client):
    # Mock manager returns 0.6, threshold 0.5 -> Anomaly
    payload = {
        "vmaf_mean": 90.0,
        "vmaf_std": 5.0,
        "vmaf_mad": 4.0,
        "bitrate_mean": 5000.0,
        "stall_ratio": 0.0,
        "tsl_end": 10.0
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_anomaly"] is True
    assert data["anomaly_score"] == 0.6
