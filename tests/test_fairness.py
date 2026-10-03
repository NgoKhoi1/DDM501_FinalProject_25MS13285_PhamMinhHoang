"""
Tests for fairness module.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fairness import group_metrics, fairness_gaps, fairness_report


class FixedModel:
    """Model that returns predetermined predictions."""

    def __init__(self, predictions):
        self.predictions = np.array(predictions)

    def predict(self, X):
        return self.predictions


@pytest.fixture
def two_groups():
    """Group a: all 4 predictions correct. Group b: 2 of 4 correct, good wines are missed."""
    y = pd.Series([1, 1, 0, 0, 1, 1, 0, 0])
    groups = pd.Series(['a'] * 4 + ['b'] * 4)
    model = FixedModel([1, 1, 0, 0, 0, 0, 0, 0])
    return model, pd.DataFrame({'x': range(8)}), y, groups


class TestGroupMetrics:
    def test_per_group_values(self, two_groups):
        metrics = group_metrics(*two_groups)
        assert list(metrics.index) == ['a', 'b']
        assert metrics.loc['a', 'n'] == 4
        assert metrics.loc['a', 'accuracy'] == 1.0
        assert metrics.loc['b', 'accuracy'] == 0.5
        assert metrics.loc['a', 'selection_rate'] == 0.5
        assert metrics.loc['b', 'selection_rate'] == 0.0
        assert metrics.loc['a', 'tpr'] == 1.0
        assert metrics.loc['b', 'tpr'] == 0.0
        assert metrics.loc['b', 'fpr'] == 0.0

    def test_group_without_positives_has_nan_tpr(self):
        y = pd.Series([0, 0])
        metrics = group_metrics(FixedModel([0, 1]), pd.DataFrame({'x': [1, 2]}), y, pd.Series(['a', 'a']))
        assert np.isnan(metrics.loc['a', 'tpr'])
        assert metrics.loc['a', 'fpr'] == 0.5


class TestFairnessGaps:
    def test_gaps(self, two_groups):
        gaps = fairness_gaps(group_metrics(*two_groups))
        assert gaps == {
            'accuracy_gap': 0.5,
            'demographic_parity_gap': 0.5,
            'equal_opportunity_gap': 1.0,
            'false_positive_rate_gap': 0.0,
        }

    def test_single_group_has_no_gap(self):
        y = pd.Series([1, 0, 1, 0])
        metrics = group_metrics(FixedModel([1, 0, 0, 0]), pd.DataFrame({'x': range(4)}), y, pd.Series(['a'] * 4))
        assert all(gap == 0 for gap in fairness_gaps(metrics).values())


class TestFairnessReport:
    def test_report_is_flat_and_numeric(self, two_groups):
        report = fairness_report(*two_groups)
        assert report['accuracy_a'] == 1.0
        assert report['accuracy_b'] == 0.5
        assert report['accuracy_gap'] == 0.5
        assert all(isinstance(v, float) for v in report.values())

    def test_real_model_on_wine_colors(self, trained_model, wine_data):
        """End to end on the real data: both colors are reported and gaps stay within [0, 1]."""
        from data_pipeline import FEATURE_NAMES, TARGET, ingest_data, clean_data
        df = clean_data(ingest_data())
        report = fairness_report(trained_model, df[FEATURE_NAMES].to_numpy(), df[TARGET], df['color'])
        assert {'accuracy_red', 'accuracy_white'} <= set(report)
        assert all(0.0 <= v <= 1.0 for v in report.values())
