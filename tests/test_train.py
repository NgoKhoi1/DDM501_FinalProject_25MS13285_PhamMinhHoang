"""
Tests for train module.
MLflow calls are mocked to avoid requiring a running MLflow server.
"""
import pytest
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import train_model, tune_model, build_serving_pipeline, log_to_mlflow


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
        assert accuracy > 0.7, f"Accuracy {accuracy} is below threshold"

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
        assert unique_preds.issubset({0, 1})


class TestBuildServingPipeline:
    """Tests for build_serving_pipeline."""

    def test_pipeline_takes_raw_features(self, wine_data):
        X, y, _ = wine_data
        scaler = StandardScaler().fit(X)
        model = train_model(scaler.transform(X), y)
        pipeline = build_serving_pipeline(scaler, model)
        # Raw features through the pipeline == scaled features through the bare model
        np.testing.assert_array_equal(pipeline.predict(X[:20]), model.predict(scaler.transform(X[:20])))


class TestLogToMlflow:
    """Tests for log_to_mlflow with mocked MLflow."""

    @pytest.fixture
    def fitted(self, wine_data):
        X, y, feature_names = wine_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler().fit(X_train)
        pipeline = build_serving_pipeline(scaler, train_model(scaler.transform(X_train), y_train))
        return pipeline, pd.DataFrame(X_test, columns=feature_names), y_test

    @patch('train.mlflow')
    def test_log_to_mlflow_calls_mlflow(self, mock_mlflow, fitted):
        pipeline, X_test, y_test = fitted
        params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
        mock_mlflow.start_run.return_value.__enter__.return_value.info.run_id = 'run-123'

        run_id = log_to_mlflow(pipeline, X_test, y_test, params, tags={'colors': 'red'})

        assert run_id == 'run-123'
        mock_mlflow.set_experiment.assert_called_once_with('wine_quality_experiment')
        mock_mlflow.log_params.assert_called_once_with(params)
        mock_mlflow.set_tags.assert_called_once_with({'colors': 'red'})
        mock_mlflow.log_figure.assert_called_once()
        assert mock_mlflow.sklearn.log_model.call_args.kwargs['sk_model'] is pipeline

    @patch('train.mlflow')
    def test_log_to_mlflow_logs_correct_metrics(self, mock_mlflow, fitted):
        pipeline, X_test, y_test = fitted
        log_to_mlflow(pipeline, X_test, y_test, {'n_estimators': 100})

        logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
        assert set(logged_metrics) == {'accuracy', 'f1_score', 'precision', 'recall'}
        assert all(0.0 <= v <= 1.0 for v in logged_metrics.values())

    @patch('train.mlflow')
    def test_log_to_mlflow_extra_metrics_and_artifacts(self, mock_mlflow, fitted, tmp_path):
        pipeline, X_test, y_test = fitted
        log_to_mlflow(pipeline, X_test, y_test, {}, extra_metrics={'accuracy_gap': 0.1}, artifacts_dir=str(tmp_path))

        assert mock_mlflow.log_metrics.call_args_list[0][0][0]['accuracy_gap'] == 0.1
        mock_mlflow.log_artifacts.assert_called_once_with(str(tmp_path))

    @patch('train.mlflow')
    def test_log_to_mlflow_logs_every_cv_candidate(self, mock_mlflow, fitted, wine_data):
        pipeline, X_test, y_test = fitted
        X, y, _ = wine_data
        search = tune_model(X[:300], y[:300], param_grid={'n_estimators': [5, 10]}, cv=3)

        log_to_mlflow(pipeline, X_test, y_test, search.best_params_, search=search)

        nested_runs = [c for c in mock_mlflow.start_run.call_args_list if c.kwargs.get('nested')]
        assert len(nested_runs) == 2
        logged = [c[0][0] for c in mock_mlflow.log_metrics.call_args_list]
        assert sum('cv_accuracy_mean' in metrics for metrics in logged) == 3  # best + 2 candidates


class TestTuneModel:
    """Tests for tune_model (grid search with cross-validation)."""

    def test_tune_evaluates_every_candidate(self, wine_data):
        X, y, _ = wine_data
        search = tune_model(X[:300], y[:300], param_grid={'n_estimators': [5, 10], 'max_depth': [2, 4]}, cv=3)
        assert len(search.cv_results_['params']) == 4
        assert search.best_params_ in search.cv_results_['params']
        assert 0.0 <= search.best_score_ <= 1.0

    def test_tune_returns_fitted_best_model(self, wine_data):
        X, y, _ = wine_data
        search = tune_model(X[:300], y[:300], param_grid={'n_estimators': [5]}, cv=3)
        assert isinstance(search.best_estimator_, RandomForestClassifier)
        assert len(search.best_estimator_.predict(X[:5])) == 5
