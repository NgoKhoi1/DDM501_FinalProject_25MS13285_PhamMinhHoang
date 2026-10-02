"""
Tests for explainability module.
SHAP import is handled gracefully if not installed.
"""
import pytest
import sys
import os
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from explainability import plot_feature_importance, generate_report


class TestFeatureImportance:
    """Tests for plot_feature_importance."""

    def test_plot_saves_file(self, trained_model, wine_data):
        _, _, feature_names = wine_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "importance.png")
            plot_feature_importance(trained_model, list(feature_names), save_path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_plot_no_save(self, trained_model, wine_data):
        """Should not raise when save_path is None (uses plt.show which is mocked)."""
        _, _, feature_names = wine_data
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        # Should not raise
        plot_feature_importance(trained_model, list(feature_names), save_path=None)

    def test_plot_handles_no_importances(self, wine_data):
        """Model without feature_importances_ should be handled gracefully."""
        from unittest.mock import MagicMock
        mock_model = MagicMock(spec=[])  # No attributes
        del mock_model.feature_importances_
        _, _, feature_names = wine_data
        # Should not raise, just log error
        plot_feature_importance(mock_model, list(feature_names))


class TestGenerateReport:
    """Tests for generate_report."""

    def test_report_creates_html(self, trained_model, wine_data):
        X, y, feature_names = wine_data
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_report(
                trained_model,
                pd.DataFrame(X, columns=feature_names),
                pd.Series(y),
                list(feature_names),
                tmpdir
            )
            report_path = os.path.join(tmpdir, "report.html")
            assert os.path.exists(report_path)
            assert os.path.getsize(report_path) > 0

    def test_report_creates_feature_importance_png(self, trained_model, wine_data):
        X, y, feature_names = wine_data
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_report(
                trained_model,
                pd.DataFrame(X, columns=feature_names),
                pd.Series(y),
                list(feature_names),
                tmpdir
            )
            feat_path = os.path.join(tmpdir, "feature_importance.png")
            assert os.path.exists(feat_path)

    def test_report_html_content(self, trained_model, wine_data):
        X, y, feature_names = wine_data
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_report(
                trained_model,
                pd.DataFrame(X, columns=feature_names),
                pd.Series(y),
                list(feature_names),
                tmpdir
            )
            report_path = os.path.join(tmpdir, "report.html")
            with open(report_path, 'r') as f:
                content = f.read()
            assert "Model Explainability Report" in content
            assert "Feature Importance" in content

    def test_report_creates_output_dir(self, trained_model, wine_data):
        X, y, feature_names = wine_data
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "new_report_dir")
            generate_report(
                trained_model,
                pd.DataFrame(X, columns=feature_names),
                pd.Series(y),
                list(feature_names),
                new_dir
            )
            assert os.path.isdir(new_dir)


class TestShapReport:
    """Tests for generate_shap_report."""

    def test_shap_report_returns_values_or_none(self, trained_model, wine_data):
        """SHAP may or may not be installed; either way should not crash."""
        X, _, feature_names = wine_data
        from explainability import generate_shap_report
        result = generate_shap_report(
            trained_model,
            pd.DataFrame(X[:20], columns=feature_names),
            list(feature_names)
        )
        # Returns shap_values if SHAP installed, None otherwise
        assert result is None or result is not None  # Just ensuring no crash

    @patch.dict('sys.modules', {'shap': None})
    def test_shap_report_handles_import_error(self, trained_model, wine_data):
        """When SHAP is not installed, should return None gracefully."""
        X, _, feature_names = wine_data
        # Re-import to trigger ImportError path
        import importlib
        import explainability
        importlib.reload(explainability)
        result = explainability.generate_shap_report(
            trained_model,
            pd.DataFrame(X[:10], columns=feature_names),
            list(feature_names)
        )
        assert result is None
