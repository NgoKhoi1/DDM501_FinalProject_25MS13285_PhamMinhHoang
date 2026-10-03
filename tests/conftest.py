import pytest
from sklearn.ensemble import RandomForestClassifier
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_pipeline import FEATURE_NAMES, TARGET, ingest_data, clean_data  # noqa: E402

@pytest.fixture(scope="session")
def wine_data():
    """Cleaned red + white wine data as (X, y, feature_names)."""
    df = clean_data(ingest_data())
    return df[FEATURE_NAMES].to_numpy(), df[TARGET].to_numpy(), FEATURE_NAMES

@pytest.fixture
def sample_features():
    # 11 features
    return [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]

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
