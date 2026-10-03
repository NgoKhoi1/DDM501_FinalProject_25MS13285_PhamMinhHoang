"""
Integration tests for the serving API (api/main.py).
The real FastAPI app is exercised through TestClient; only the MLflow model
and the outgoing call to Evidently are replaced.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api import main as api_main


@pytest.fixture
def client():
    # No `with`: the startup event (which loads the model from MLflow) is not run
    return TestClient(api_main.app)


@pytest.fixture
def loaded_model(monkeypatch):
    """A model is loaded in the API: always predicts class 1."""
    model = MagicMock()
    model.predict.return_value = np.array([1])
    monkeypatch.setattr(api_main.model_manager, "model", model)
    monkeypatch.setattr(api_main.model_manager, "model_version", "3")
    return model


@pytest.fixture
def no_model(monkeypatch):
    monkeypatch.setattr(api_main.model_manager, "model", None)
    monkeypatch.setattr(api_main.model_manager, "model_version", None)


class TestHealthEndpoint:
    def test_healthy_when_model_loaded(self, client, loaded_model):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == "wine_quality_model"
        assert data["model_version"] == "3"

    def test_unhealthy_without_model(self, client, no_model):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "unhealthy"
        assert response.json()["model_loaded"] is False


class TestPredictEndpoint:
    def test_predict_valid_features(self, client, loaded_model, sample_features):
        response = client.post("/predict", json={"features": sample_features})
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 1.0
        assert data["model_version"] == "3"
        assert data["prediction_id"]
        assert data["latency_ms"] >= 0

    def test_predict_passes_named_features_to_model(self, client, loaded_model, sample_features):
        client.post("/predict", json={"features": sample_features})
        features_df = loaded_model.predict.call_args[0][0]
        assert list(features_df.columns) == api_main.FEATURE_NAMES
        assert features_df.iloc[0].tolist() == sample_features

    def test_prediction_ids_are_unique(self, client, loaded_model, sample_features):
        ids = {client.post("/predict", json={"features": sample_features}).json()["prediction_id"] for _ in range(3)}
        assert len(ids) == 3

    @pytest.mark.parametrize("features", [[], [1.0, 2.0, 3.0], [0.0] * 13])
    def test_predict_wrong_feature_count(self, client, loaded_model, features):
        response = client.post("/predict", json={"features": features})
        assert response.status_code == 400
        assert "Expected 11 features" in response.json()["detail"]
        loaded_model.predict.assert_not_called()

    def test_predict_invalid_body(self, client, loaded_model):
        assert client.post("/predict", json={"features": "not a list"}).status_code == 422
        assert client.post("/predict", json={}).status_code == 422

    def test_predict_without_model(self, client, no_model, sample_features):
        response = client.post("/predict", json={"features": sample_features})
        assert response.status_code == 503

    def test_predict_model_failure(self, client, loaded_model, sample_features):
        loaded_model.predict.side_effect = RuntimeError("boom")
        response = client.post("/predict", json={"features": sample_features})
        assert response.status_code == 500
        assert response.json()["detail"] == "Prediction failed"


class TestEvidentlyCapture:
    def test_inference_is_forwarded_to_evidently(self, client, loaded_model, sample_features, monkeypatch):
        monkeypatch.setattr(api_main, "EVIDENTLY_URL", "http://evidently:8001")
        with patch.object(api_main.requests, "post") as mock_post:
            response = client.post("/predict", json={"features": sample_features})

        assert response.status_code == 200
        url = mock_post.call_args[0][0]
        payload = mock_post.call_args.kwargs["json"]
        assert url == "http://evidently:8001/capture"
        assert payload["features"] == dict(zip(api_main.FEATURE_NAMES, sample_features))
        assert payload["prediction"] == 1.0
        assert payload["model_version"] == "3"

    def test_capture_failure_does_not_break_prediction(self, client, loaded_model, sample_features, monkeypatch):
        monkeypatch.setattr(api_main, "EVIDENTLY_URL", "http://evidently:8001")
        with patch.object(api_main.requests, "post", side_effect=ConnectionError("evidently down")):
            response = client.post("/predict", json={"features": sample_features})
        assert response.status_code == 200

    def test_no_capture_when_evidently_not_configured(self, client, loaded_model, sample_features, monkeypatch):
        monkeypatch.setattr(api_main, "EVIDENTLY_URL", "")
        with patch.object(api_main.requests, "post") as mock_post:
            client.post("/predict", json={"features": sample_features})
        mock_post.assert_not_called()


class TestMetricsEndpoint:
    def test_metrics_exposes_request_and_prediction_counters(self, client, loaded_model, sample_features):
        client.post("/predict", json={"features": sample_features})
        response = client.get("/metrics")
        assert response.status_code == 200
        assert 'api_requests_total{endpoint="/predict",method="POST",status="200"}' in response.text
        assert 'model_predictions_total{model_name="wine_quality_model",model_version="3"}' in response.text
        assert "model_prediction_latency_seconds_bucket" in response.text

    def test_metrics_counts_prediction_errors(self, client, no_model, sample_features):
        client.post("/predict", json={"features": sample_features})
        assert 'error_type="model_not_loaded"' in client.get("/metrics").text


class TestModelEndpoints:
    def test_root_lists_endpoints(self, client):
        data = client.get("/").json()
        assert data["endpoints"]["predict"] == "/predict"
        assert data["endpoints"]["health"] == "/health"

    def test_model_info(self, client, loaded_model):
        data = client.get("/model/info").json()
        assert data["model_name"] == "wine_quality_model"
        assert data["model_version"] == "3"

    def test_model_info_without_model(self, client, no_model):
        assert client.get("/model/info").status_code == 503

    def test_reload_success(self, client, loaded_model):
        with patch.object(api_main.model_manager, "load_model", return_value=True):
            response = client.post("/model/reload")
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_reload_failure(self, client):
        with patch.object(api_main.model_manager, "load_model", return_value=False):
            assert client.post("/model/reload").status_code == 500


class TestModelManager:
    """load_model against a mocked MLflow registry."""

    def test_load_model_from_registry(self, monkeypatch):
        manager = api_main.ModelManager()
        fake_model = MagicMock()
        monkeypatch.setattr(api_main.mlflow.pyfunc, "load_model", lambda uri: fake_model)
        client = MagicMock()
        client.get_latest_versions.return_value = [MagicMock(version="7")]
        monkeypatch.setattr(api_main.mlflow.tracking, "MlflowClient", lambda: client)

        assert manager.load_model() is True
        assert manager.model is fake_model
        assert manager.model_version == "7"
        assert manager.model_uri == "models:/wine_quality_model/Production"

    def test_load_model_failure_returns_false(self, monkeypatch):
        manager = api_main.ModelManager()

        def fail(uri):
            raise RuntimeError("registry unreachable")

        monkeypatch.setattr(api_main.mlflow.pyfunc, "load_model", fail)
        assert manager.load_model() is False
        assert manager.model is None


class TestOpenApi:
    def test_openapi_has_working_example(self, client):
        schema = client.get("/openapi.json").json()
        example = schema["components"]["schemas"]["PredictionRequest"]["example"]
        assert len(example["features"]) == len(api_main.FEATURE_NAMES)
