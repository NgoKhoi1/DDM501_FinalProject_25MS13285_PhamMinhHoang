"""
Tests for evaluate module.
MLflow-dependent functions are mocked.
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluate import evaluate_model, should_promote, load_production_model, register_and_promote


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
        assert len(cm) == 2  # 2 classes
        assert len(cm[0]) == 2

    def test_evaluate_good_model(self, wine_data):
        X, y, _ = wine_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        result = evaluate_model(model, X_test, y_test)
        assert result['accuracy'] > 0.7


class TestShouldPromote:
    """Tests for should_promote."""

    def test_better_candidate_is_promoted(self):
        assert should_promote({'accuracy': 0.76}, {'accuracy': 0.70}) is True

    def test_worse_candidate_is_rejected(self):
        assert should_promote({'accuracy': 0.70}, {'accuracy': 0.76}) is False

    def test_equal_candidate_is_promoted(self):
        assert should_promote({'accuracy': 0.76}, {'accuracy': 0.76}) is True

    def test_other_metric(self):
        candidate = {'accuracy': 0.9, 'f1_score': 0.5}
        production = {'accuracy': 0.8, 'f1_score': 0.6}
        assert should_promote(candidate, production, metric='f1_score') is False


class TestLoadProductionModel:
    """Tests for load_production_model (mocked MLflow)."""

    @patch('evaluate.mlflow')
    @patch('evaluate.MlflowClient')
    def test_loads_model_when_production_exists(self, mock_client_cls, mock_mlflow):
        mock_client_cls.return_value.search_model_versions.return_value = [
            MagicMock(current_stage='Archived'), MagicMock(current_stage='Production')
        ]
        model = load_production_model('wine_quality_model')
        mock_mlflow.sklearn.load_model.assert_called_once_with("models:/wine_quality_model/Production")
        assert model is mock_mlflow.sklearn.load_model.return_value

    @patch('evaluate.mlflow')
    @patch('evaluate.MlflowClient')
    def test_returns_none_when_no_production(self, mock_client_cls, mock_mlflow):
        mock_client_cls.return_value.search_model_versions.return_value = [MagicMock(current_stage='Staging')]
        assert load_production_model('wine_quality_model') is None
        mock_mlflow.sklearn.load_model.assert_not_called()

    @patch('evaluate.mlflow')
    @patch('evaluate.MlflowClient')
    def test_returns_none_when_model_not_registered(self, mock_client_cls, mock_mlflow):
        mock_client_cls.return_value.search_model_versions.return_value = []
        assert load_production_model('wine_quality_model') is None


class TestRegisterAndPromote:
    """Tests for register_and_promote (mocked MLflow)."""

    @patch('evaluate.mlflow')
    @patch('evaluate.MlflowClient')
    def test_registers_and_transitions(self, mock_client_cls, mock_mlflow):
        mock_mlflow.register_model.return_value.version = '3'

        version = register_and_promote('run-123', 'wine_quality_model')

        assert version == '3'
        mock_mlflow.register_model.assert_called_once_with("runs:/run-123/model", 'wine_quality_model')
        mock_client_cls.return_value.transition_model_version_stage.assert_called_once_with(
            name='wine_quality_model',
            version='3',
            stage='Production',
            archive_existing_versions=True
        )

    @patch('evaluate.mlflow')
    @patch('evaluate.MlflowClient')
    def test_promotion_error_is_raised(self, mock_client_cls, mock_mlflow):
        mock_client_cls.return_value.transition_model_version_stage.side_effect = Exception("Connection error")
        with pytest.raises(Exception, match="Connection error"):
            register_and_promote('run-123')
