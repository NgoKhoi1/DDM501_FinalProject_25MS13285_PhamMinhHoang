"""
Fairness Module.
Compares model behaviour across groups of the data (here: wine color).

The dataset has no personal attributes, so the group of interest is the product
type: a model trained mostly on one color must not be systematically worse on the other.
"""
import logging
import pandas as pd
from typing import Any, Dict
from sklearn.metrics import accuracy_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def group_metrics(model: Any, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Per-group sample count, accuracy, selection rate (share predicted positive), TPR and FPR."""
    y_pred = pd.Series(model.predict(X), index=y.index)
    rows = {}
    for group in sorted(groups.unique()):
        mask = groups == group
        tn, fp, fn, tp = confusion_matrix(y[mask], y_pred[mask], labels=[0, 1]).ravel()
        rows[group] = {
            'n': int(mask.sum()),
            'accuracy': accuracy_score(y[mask], y_pred[mask]),
            'selection_rate': y_pred[mask].mean(),
            'base_rate': y[mask].mean(),
            'tpr': tp / (tp + fn) if (tp + fn) else float('nan'),
            'fpr': fp / (fp + tn) if (fp + tn) else float('nan'),
        }
    return pd.DataFrame.from_dict(rows, orient='index')


def fairness_gaps(metrics: pd.DataFrame) -> Dict[str, float]:
    """Largest difference between groups for each criterion (0 = perfectly even)."""
    def gap(column):
        return float(metrics[column].max() - metrics[column].min())

    return {
        'accuracy_gap': gap('accuracy'),
        # Demographic parity: do groups receive positive predictions at the same rate?
        'demographic_parity_gap': gap('selection_rate'),
        # Equal opportunity: are truly good wines recognised equally often in each group?
        'equal_opportunity_gap': gap('tpr'),
        # Equalized odds also needs the false positive rates to match
        'false_positive_rate_gap': gap('fpr'),
    }


def fairness_report(model: Any, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> Dict[str, float]:
    """Flat dict of per-group accuracy and the gaps between groups, ready to log as MLflow metrics."""
    metrics = group_metrics(model, X, y, groups)
    report = {f'accuracy_{group}': row['accuracy'] for group, row in metrics.iterrows()}
    report.update(fairness_gaps(metrics))
    logger.info(f"Fairness report: {report}")
    return report
