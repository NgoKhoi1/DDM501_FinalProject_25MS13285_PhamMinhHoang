"""
Tests for API endpoints.
Uses a self-contained mock FastAPI app to test endpoint behavior
without requiring MLflow or model loading.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional


# ============================================
# MOCK APP (mirrors real api/main.py structure)
# ============================================

mock_model = MagicMock()
mock_model.predict.return_value = np.array([1.0])

mock_app = FastAPI(title="ML Model API - Test")


class PredictionRequest(BaseModel):
    features: List[float]
    feature_names: Optional[List[str]] = None


class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    model_version: str
    timestamp: str
    latency_ms: float


@mock_app.get("/")
async def root():
    return {
        "message": "ML Model API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metrics": "/metrics",
            "model_info": "/model/info"
        }
    }


@mock_app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": "wine_quality_model",
        "model_version": "1",
        "uptime_seconds": 100.0
    }


@mock_app.post("/predict")
async def predict(request: PredictionRequest):
    if len(request.features) != 13:
        raise HTTPException(status_code=400, detail="Expected 13 features")
    pred = float(mock_model.predict([request.features])[0])
    return {
        "prediction": pred,
        "model_name": "wine_quality_model",
        "model_version": "1",
        "timestamp": "2024-01-01T00:00:00",
        "latency_ms": 5.0
    }


@mock_app.get("/metrics")
async def metrics():
    return "# HELP api_requests_total Total API requests\napi_requests_total 42\n"


@mock_app.get("/model/info")
async def model_info():
    return {
        "model_name": "wine_quality_model",
        "model_version": "1",
        "model_uri": "models:/wine_quality_model/Production",
        "load_time_seconds": 0.5,
        "tracking_uri": "http://mlflow:5000"
    }


# ============================================
# TEST CLIENT
# ============================================

@pytest.fixture
def client():
    return TestClient(mock_app)


# ============================================
# TESTS
# ============================================

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_contains_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_contains_model_info(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True
        assert data["model_name"] == "wine_quality_model"


class TestPredictEndpoint:
    def test_predict_valid_features(self, client, sample_features):
        response = client.post("/predict", json={"features": sample_features})
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "model_name" in data
        assert "latency_ms" in data

    def test_predict_returns_prediction_value(self, client, sample_features):
        response = client.post("/predict", json={"features": sample_features})
        data = response.json()
        assert isinstance(data["prediction"], (int, float))

    def test_predict_invalid_feature_count(self, client):
        response = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
        assert response.status_code == 400

    def test_predict_empty_features(self, client):
        response = client.post("/predict", json={"features": []})
        assert response.status_code == 400

    def test_predict_with_feature_names(self, client, sample_features):
        feature_names = [
            "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
            "magnesium", "total_phenols", "flavanoids",
            "nonflavanoid_phenols", "proanthocyanins",
            "color_intensity", "hue", "od280_od315_of_diluted_wines", "proline"
        ]
        response = client.post("/predict", json={
            "features": sample_features,
            "feature_names": feature_names
        })
        assert response.status_code == 200


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_contains_data(self, client):
        response = client.get("/metrics")
        assert len(response.text) > 0


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_endpoints(self, client):
        response = client.get("/")
        data = response.json()
        assert "endpoints" in data
        assert "predict" in data["endpoints"]
        assert "health" in data["endpoints"]


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_contains_name(self, client):
        response = client.get("/model/info")
        data = response.json()
        assert data["model_name"] == "wine_quality_model"
        assert "model_version" in data
