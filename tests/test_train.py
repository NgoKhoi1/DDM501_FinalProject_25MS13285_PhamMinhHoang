"""
Tests for train module.
MLflow calls are mocked to avoid requiring a running MLflow server.
"""
import pytest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import train_model


class TestTrainModel:
    """Tests for train_model function."""

    def test_train_returns_classifier(self, wine_data):
        X, y, _ = wine_data
        model = train_model(X, y)
        assert isinstance(model, RandomForestClassifier)

    def test_train_with_custom_params(self, wine_data):
        X, y, _ = wine_data
        params = {'n_estimators': 50, 'max_depth': 5, 'random_state': 42}
        model = train_model(X, y, params=params)
        assert model.n_estimators == 50
        assert model.max_depth == 5

    def test_train_with_default_params(self, wine_data):
        X, y, _ = wine_data
        model = train_model(X, y)
        assert model.n_estimators == 100
        assert model.max_depth == 10

    def test_trained_model_can_predict(self, wine_data):
        X, y, _ = wine_data
        model = train_model(X, y)
        predictions = model.predict(X[:5])
        assert len(predictions) == 5

    def test_trained_model_accuracy(self, wine_data):
        X, y, _ = wine_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        assert accuracy > 0.85, f"Accuracy {accuracy} is below threshold"

    def test_train_reproducible(self, wine_data):
        X, y, _ = wine_data
        model1 = train_model(X, y, params={'n_estimators': 10, 'random_state': 42})
        model2 = train_model(X, y, params={'n_estimators': 10, 'random_state': 42})
        pred1 = model1.predict(X[:10])
        pred2 = model2.predict(X[:10])
        np.testing.assert_array_equal(pred1, pred2)

    def test_train_has_feature_importances(self, wine_data):
        X, y, _ = wine_data
        model = train_model(X, y)
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == X.shape[1]
        assert np.isclose(model.feature_importances_.sum(), 1.0)

    def test_train_predicts_correct_classes(self, wine_data):
        X, y, _ = wine_data
        model = train_model(X, y)
        unique_preds = set(model.predict(X))
        assert unique_preds.issubset({0, 1, 2})


class TestLogToMlflow:
    """Tests for log_to_mlflow with mocked MLflow."""

    @patch('train.mlflow')
    @patch('train.plt')
    @patch('train.ConfusionMatrixDisplay')
    def test_log_to_mlflow_calls_mlflow(self, mock_cm_disp, mock_plt, mock_mlflow, wine_data):
        X, y, feature_names = wine_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}

        # Setup mock for plt.subplots to return (fig, ax) tuple
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Setup mock context manager
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from train import log_to_mlflow
        log_to_mlflow(model, X_train, y_train, X_test, y_test, params, list(feature_names))

        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once_with('wine_quality_experiment')
        mock_mlflow.log_params.assert_called_once_with(params)
        mock_mlflow.log_metrics.assert_called_once()
        mock_mlflow.sklearn.log_model.assert_called_once()

    @patch('train.mlflow')
    @patch('train.plt')
    @patch('train.ConfusionMatrixDisplay')
    def test_log_to_mlflow_logs_correct_metrics(self, mock_cm_disp, mock_plt, mock_mlflow, wine_data):
        X, y, feature_names = wine_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        params = {'n_estimators': 100, 'random_state': 42}

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from train import log_to_mlflow
        log_to_mlflow(model, X_train, y_train, X_test, y_test, params, list(feature_names))

        # Verify metrics were logged
        logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
        assert 'accuracy' in logged_metrics
        assert 'f1_score' in logged_metrics
        assert 'precision' in logged_metrics
        assert 'recall' in logged_metrics
        assert all(0.0 <= v <= 1.0 for v in logged_metrics.values())


class TestRunTrainingPipeline:
    """Tests for run_training_pipeline with mocked MLflow."""

    @patch('train.log_to_mlflow')
    def test_pipeline_runs_without_error(self, mock_log):
        from train import run_training_pipeline
        # Should not raise when MLflow logging is mocked
        run_training_pipeline()
        mock_log.assert_called_once()

    @patch('train.log_to_mlflow')
    def test_pipeline_calls_log_with_model(self, mock_log):
        from train import run_training_pipeline
        run_training_pipeline()
        args = mock_log.call_args
        # First arg should be a model
        model = args[0][0]
        assert hasattr(model, 'predict')

