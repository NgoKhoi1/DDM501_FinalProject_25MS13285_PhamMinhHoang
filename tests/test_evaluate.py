"""
Tests for evaluate module.
MLflow-dependent functions are mocked.
"""
import pytest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluate import evaluate_model, compare_with_production, promote_model


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_evaluate_returns_dict(self, trained_model, wine_data):
        X, y, _ = wine_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        result = evaluate_model(trained_model, X_test, y_test)
        assert isinstance(result, dict)

    def test_evaluate_contains_all_metrics(self, trained_model, wine_data):
        X, y, _ = wine_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        result = evaluate_model(trained_model, X_test, y_test)
        assert 'accuracy' in result
        assert 'f1_score' in result
        assert 'precision' in result
        assert 'recall' in result
        assert 'confusion_matrix' in result

    def test_evaluate_accuracy_range(self, trained_model, wine_data):
        X, y, _ = wine_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        result = evaluate_model(trained_model, X_test, y_test)
        assert 0.0 <= result['accuracy'] <= 1.0

    def test_evaluate_f1_range(self, trained_model, wine_data):
        X, y, _ = wine_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        result = evaluate_model(trained_model, X_test, y_test)
        assert 0.0 <= result['f1_score'] <= 1.0

    def test_evaluate_confusion_matrix_shape(self, trained_model, wine_data):
        X, y, _ = wine_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        result = evaluate_model(trained_model, X_test, y_test)
        cm = result['confusion_matrix']
        assert len(cm) == 3  # 3 classes
        assert len(cm[0]) == 3

    def test_evaluate_good_model(self, wine_data):
        X, y, _ = wine_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        result = evaluate_model(model, X_test, y_test)
        assert result['accuracy'] > 0.85


class TestCompareWithProduction:
    """Tests for compare_with_production (mocked MLflow)."""

    @patch('evaluate.get_production_model_metrics')
    def test_compare_better_model(self, mock_get_metrics):
        mock_get_metrics.return_value = {'accuracy': 0.85}
        candidate = {'accuracy': 0.95}
        assert compare_with_production(candidate, metric='accuracy', threshold=0.01) is True

    @patch('evaluate.get_production_model_metrics')
    def test_compare_worse_model(self, mock_get_metrics):
        mock_get_metrics.return_value = {'accuracy': 0.95}
        candidate = {'accuracy': 0.85}
        assert compare_with_production(candidate, metric='accuracy', threshold=0.01) is False

    @patch('evaluate.get_production_model_metrics')
    def test_compare_no_production_model(self, mock_get_metrics):
        mock_get_metrics.return_value = {}
        candidate = {'accuracy': 0.70}
        # No production model = candidate wins automatically
        assert compare_with_production(candidate) is True

    @patch('evaluate.get_production_model_metrics')
    def test_compare_equal_model_below_threshold(self, mock_get_metrics):
        mock_get_metrics.return_value = {'accuracy': 0.90}
        candidate = {'accuracy': 0.905}  # only 0.005 better, threshold is 0.01
        assert compare_with_production(candidate, metric='accuracy', threshold=0.01) is False


class TestPromoteModel:
    """Tests for promote_model (mocked MLflow)."""

    @patch('evaluate.MlflowClient')
    def test_promote_calls_transition(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        promote_model('wine_quality_model', version=1, stage='Production')
        mock_client.transition_model_version_stage.assert_called_once_with(
            name='wine_quality_model',
            version=1,
            stage='Production',
            archive_existing_versions=True
        )

    @patch('evaluate.MlflowClient')
    def test_promote_staging(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        promote_model('wine_quality_model', version=2, stage='Staging')
        mock_client.transition_model_version_stage.assert_called_once()

    @patch('evaluate.MlflowClient')
    def test_promote_handles_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.transition_model_version_stage.side_effect = Exception("Connection error")
        mock_client_cls.return_value = mock_client
        # Should not raise, just log error
        promote_model('wine_quality_model', version=1)


class TestGetProductionModelMetrics:
    """Tests for get_production_model_metrics."""

    @patch('evaluate.mlflow')
    @patch('evaluate.MlflowClient')
    def test_returns_metrics_when_production_exists(self, mock_client_cls, mock_mlflow):
        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.current_stage = 'Production'
        mock_version.run_id = 'test-run-id'
        mock_client.search_model_versions.return_value = [mock_version]
        mock_client_cls.return_value = mock_client

        mock_run = MagicMock()
        mock_run.data.metrics = {'accuracy': 0.95, 'f1_score': 0.93}
        mock_mlflow.get_run.return_value = mock_run

        from evaluate import get_production_model_metrics
        result = get_production_model_metrics('wine_quality_model')
        assert result == {'accuracy': 0.95, 'f1_score': 0.93}

    @patch('evaluate.MlflowClient')
    def test_returns_empty_when_no_production(self, mock_client_cls):
        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.current_stage = 'Staging'  # Not production
        mock_client.search_model_versions.return_value = [mock_version]
        mock_client_cls.return_value = mock_client

        from evaluate import get_production_model_metrics
        result = get_production_model_metrics('wine_quality_model')
        assert result == {}

    @patch('evaluate.MlflowClient')
    def test_returns_empty_on_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.search_model_versions.side_effect = Exception("Connection refused")
        mock_client_cls.return_value = mock_client

        from evaluate import get_production_model_metrics
        result = get_production_model_metrics('wine_quality_model')
        assert result == {}

