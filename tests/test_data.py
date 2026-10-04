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

from data_pipeline import (
    FEATURE_NAMES, TARGET,
    ingest_data, clean_data, split_data, save_data_artifact, load_data_artifact
)
from feature_engineering import fit_scaler, transform_features, save_scaler, load_scaler


@pytest.fixture(scope="module")
def raw_data():
    return ingest_data()


@pytest.fixture(scope="module")
def clean(raw_data):
    return clean_data(raw_data)


class TestDataIngestion:
    """Tests for data ingestion."""

    def test_ingest_data_columns(self, raw_data):
        assert isinstance(raw_data, pd.DataFrame)
        assert list(raw_data.columns) == FEATURE_NAMES + ['quality', 'color']

    def test_ingest_data_all_colors(self, raw_data):
        assert raw_data['color'].value_counts().to_dict() == {'white': 4898, 'red': 1599}

    def test_ingest_data_single_color(self):
        red = ingest_data(['red'])
        assert len(red) == 1599
        assert set(red['color']) == {'red'}

    def test_ingest_data_no_nulls_initially(self, raw_data):
        assert raw_data.isnull().sum().sum() == 0

    def test_ingest_data_quality_range(self, raw_data):
        assert raw_data['quality'].between(0, 10).all()


class TestDataCleaning:
    """Tests for data cleaning."""

    def test_clean_data_fills_nulls(self, raw_data):
        dirty = raw_data.copy()
        dirty.loc[0, 'pH'] = np.nan
        dirty.loc[5, 'alcohol'] = np.nan
        assert clean_data(dirty)[FEATURE_NAMES].isnull().sum().sum() == 0

    def test_clean_data_removes_duplicates(self, raw_data, clean):
        assert len(clean) < len(raw_data)
        assert not clean.duplicated().any()

    def test_clean_data_type_casting(self, clean):
        for col in FEATURE_NAMES:
            assert clean[col].dtype == np.float64

    def test_clean_data_binary_target(self, raw_data, clean):
        assert 'quality' not in clean.columns
        assert set(clean[TARGET]) == {0, 1}
        # quality >= 6 is a good wine
        row = clean_data(raw_data.head(1).assign(quality=6))
        assert row[TARGET].iloc[0] == 1
        row = clean_data(raw_data.head(1).assign(quality=5))
        assert row[TARGET].iloc[0] == 0

    def test_schema_validation_missing_column(self, raw_data):
        with pytest.raises(ValueError, match="pH"):
            clean_data(raw_data.drop(columns=['pH']))


class TestSplitData:
    """Tests for the train/test split."""

    def test_split_sizes(self, clean):
        train, test = split_data(clean)
        assert len(train) + len(test) == len(clean)
        assert 0.19 < len(test) / len(clean) < 0.21

    def test_split_is_stratified(self, clean):
        train, test = split_data(clean)
        for color in ('red', 'white'):
            train_share = train[train.color == color][TARGET].mean()
            test_share = test[test.color == color][TARGET].mean()
            assert abs(train_share - test_share) < 0.02

    def test_red_test_set_independent_of_white(self, clean):
        """The red test rows must be the same with or without white wine, so models stay comparable."""
        _, test_all = split_data(clean)
        _, test_red = split_data(clean[clean.color == 'red'])
        pd.testing.assert_frame_equal(
            test_all[test_all.color == 'red'].reset_index(drop=True),
            test_red.reset_index(drop=True)
        )


class TestDataArtifacts:
    """Tests for saving and loading data artifacts."""

    def test_save_and_load_artifact(self, clean):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "test_data.csv")
            save_data_artifact(clean, path)
            assert os.path.exists(path), "Artifact file should exist"
            pd.testing.assert_frame_equal(load_data_artifact(path), clean)


class TestFeatureEngineering:
    """Tests for feature engineering."""

    def test_transform_keeps_shape_and_columns(self, clean):
        X = clean[FEATURE_NAMES]
        X_transformed = transform_features(X, fit_scaler(X))
        assert X_transformed.shape == X.shape
        assert list(X_transformed.columns) == FEATURE_NAMES

    def test_standardized_output(self, clean):
        X = clean[FEATURE_NAMES]
        X_transformed = transform_features(X, fit_scaler(X))
        # StandardScaler: mean near 0, std near 1
        np.testing.assert_allclose(X_transformed.mean().values, 0, atol=1e-6)
        np.testing.assert_allclose(X_transformed.std(ddof=0).values, 1, atol=1e-6)

    def test_scaler_save_and_load(self, clean):
        X = clean[FEATURE_NAMES]
        scaler = fit_scaler(X)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "scaler.joblib")
            save_scaler(scaler, path)
            assert os.path.exists(path)
            loaded_scaler = load_scaler(path)
            # Verify same transform
            original = transform_features(X.iloc[:3], scaler)
            loaded = transform_features(X.iloc[:3], loaded_scaler)
            np.testing.assert_array_almost_equal(original.values, loaded.values)
