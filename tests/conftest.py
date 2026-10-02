import pytest
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def wine_data():
    data = load_wine()
    return data.data, data.target, data.feature_names

@pytest.fixture
def sample_features():
    # 13 features
    return [13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050.0]

@pytest.fixture
def trained_model(wine_data):
    X, y, _ = wine_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model

@pytest.fixture
def mock_mlflow():
    with patch('mlflow.start_run') as mock_run:
        yield mock_run

@pytest.fixture
def api_client():
    try:
        from serving.app.main import app
    except ImportError:
        try:
            from api.main import app
        except ImportError:
            from fastapi import FastAPI
            app = FastAPI()
    
    from fastapi.testclient import TestClient
    return TestClient(app)
