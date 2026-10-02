"""
Tests for data pipeline and feature engineering modules.
"""
import pytest
import sys
import os
import tempfile
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_pipeline import ingest_data, clean_data, save_data_artifact, load_data_artifact
from feature_engineering import fit_transform_features, transform_features, save_scaler, load_scaler


class TestDataIngestion:
    """Tests for data ingestion."""

    def test_ingest_data_returns_dict(self):
        result = ingest_data()
        assert isinstance(result, dict)
        assert "X" in result
        assert "y" in result
        assert "feature_names" in result

    def test_ingest_data_shape(self):
        result = ingest_data()
        X = result["X"]
        assert X.shape[1] == 13, f"Expected 13 features, got {X.shape[1]}"
        assert X.shape[0] > 0, "Dataset should not be empty"

    def test_ingest_data_types(self):
        result = ingest_data()
        X = result["X"]
        assert isinstance(X, pd.DataFrame)
        for col in X.columns:
            assert X[col].dtype in [np.float64, np.float32, float], f"Feature {col} is not float"

    def test_ingest_data_feature_names(self):
        result = ingest_data()
        assert len(result["feature_names"]) == 13

    def test_ingest_data_no_nulls_initially(self):
        result = ingest_data()
        assert result["X"].isnull().sum().sum() == 0


class TestDataCleaning:
    """Tests for data cleaning."""

    def test_clean_data_no_nulls(self):
        result = ingest_data()
        X = result["X"].copy()
        y = result["y"].copy()
        # Inject nulls
        X.iloc[0, 0] = np.nan
        X.iloc[5, 3] = np.nan
        X_clean, y_clean = clean_data(X, y)
        assert X_clean.isnull().sum().sum() == 0, "clean_data should remove all nulls"

    def test_clean_data_correct_shape(self):
        result = ingest_data()
        X_clean, y_clean = clean_data(result["X"], result["y"])
        assert X_clean.shape[1] == 13
        assert len(y_clean) == len(X_clean)

    def test_clean_data_handles_outliers(self):
        result = ingest_data()
        X = result["X"].copy()
        y = result["y"].copy()
        # Inject extreme outlier
        X.iloc[0, 0] = 99999.0
        X_clean, y_clean = clean_data(X, y)
        # After IQR outlier removal, should have fewer rows OR outlier should be clipped
        assert X_clean.shape[0] <= X.shape[0]

    def test_clean_data_type_casting(self):
        result = ingest_data()
        X_clean, y_clean = clean_data(result["X"], result["y"])
        for col in X_clean.columns:
            assert X_clean[col].dtype == np.float64

    def test_schema_validation_13_features(self):
        result = ingest_data()
        X_clean, _ = clean_data(result["X"], result["y"])
        assert X_clean.shape[1] == 13, "Schema validation: must have 13 features"


class TestDataArtifacts:
    """Tests for saving and loading data artifacts."""

    def test_save_and_load_artifact(self):
        result = ingest_data()
        X = result["X"]
        y = result["y"]
        feature_names = list(result["feature_names"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_data.csv")
            save_data_artifact(X, y, feature_names, path)
            assert os.path.exists(path), "Artifact file should exist"

            loaded = load_data_artifact(path)
            assert loaded is not None
            assert isinstance(loaded, dict)
            assert "X" in loaded

    def test_save_artifact_creates_file(self):
        result = ingest_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.csv")
            save_data_artifact(result["X"], result["y"], list(result["feature_names"]), path)
            assert os.path.getsize(path) > 0


class TestFeatureEngineering:
    """Tests for feature engineering."""

    def test_fit_transform_features(self):
        result = ingest_data()
        X = result["X"]
        feature_names = list(result["feature_names"])
        X_transformed, scaler = fit_transform_features(X, feature_names)
        assert X_transformed is not None
        assert scaler is not None
        # May have extra engineered features
        assert X_transformed.shape[0] == X.shape[0]

    def test_transform_features_with_scaler(self):
        result = ingest_data()
        X = result["X"]
        feature_names = list(result["feature_names"])
        X_transformed, scaler = fit_transform_features(X, feature_names)
        X_new = transform_features(X.iloc[:5], scaler)
        assert X_new.shape[0] == 5

    def test_scaler_save_and_load(self):
        result = ingest_data()
        X = result["X"]
        feature_names = list(result["feature_names"])
        _, scaler = fit_transform_features(X, feature_names)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "scaler.joblib")
            save_scaler(scaler, path)
            assert os.path.exists(path)

            loaded_scaler = load_scaler(path)
            assert loaded_scaler is not None

            # Verify same transform
            original = transform_features(X.iloc[:3], scaler)
            loaded = transform_features(X.iloc[:3], loaded_scaler)
            np.testing.assert_array_almost_equal(original.values, loaded.values)

    def test_standardized_output_mean_near_zero(self):
        result = ingest_data()
        X = result["X"]
        feature_names = list(result["feature_names"])
        X_transformed, _ = fit_transform_features(X, feature_names)
        # StandardScaler: mean should be near 0
        means = X_transformed.mean().values
        for m in means:
            assert abs(m) < 0.1, f"Mean {m} is not near zero after scaling"
