"""
Explainability Module.
Three complementary views of what drives the model:
- impurity-based feature importance (built into the random forest),
- permutation importance (model-agnostic, measured on held-out data),
- SHAP values (per-prediction contributions, summarised over a sample).
"""
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Any
from sklearn.inspection import permutation_importance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_shap_report(model: Any, X: pd.DataFrame, feature_names: List[str]) -> Any:
    """Use SHAP TreeExplainer to compute SHAP values."""
    logger.info("Generating SHAP values.")
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        logger.info("SHAP values computed successfully.")
        return shap_values
    except ImportError:
        logger.warning("SHAP library not installed. Cannot generate SHAP report.")
        return None


def plot_feature_importance(model: Any, feature_names: List[str], save_path: str = None) -> None:
    """Plot and optionally save feature importance bar chart."""
    logger.info("Plotting feature importance.")
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(feature_names)), importances[indices], align="center")
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlim([-1, len(feature_names)])
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()

        plt.close()
    except AttributeError:
        logger.error("Model does not have feature_importances_ attribute.")


def compute_permutation_importance(model: Any, X: pd.DataFrame, y: pd.Series, feature_names: List[str],
                                   n_repeats: int = 10) -> pd.DataFrame:
    """Drop in accuracy when a feature is shuffled, averaged over n_repeats. Sorted, most important first."""
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, scoring='accuracy')
    return pd.DataFrame(
        {'importance_mean': result.importances_mean, 'importance_std': result.importances_std},
        index=feature_names
    ).sort_values('importance_mean', ascending=False)


def plot_permutation_importance(importance: pd.DataFrame, save_path: str) -> None:
    """Horizontal bar chart of permutation importance with its spread."""
    ordered = importance.iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Permutation Importance (drop in accuracy)")
    plt.barh(ordered.index, ordered['importance_mean'], xerr=ordered['importance_std'])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_shap_summary(model: Any, X: pd.DataFrame, save_path: str) -> None:
    """SHAP summary (beeswarm) plot for the positive class of a tree model."""
    import shap
    explanation = shap.TreeExplainer(model)(X)
    if explanation.values.ndim == 3:
        # (samples, features, classes): keep the positive class
        explanation = explanation[:, :, 1]
    plt.figure(figsize=(10, 6))
    shap.summary_plot(explanation, X, show=False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_report(model: Any, X: pd.DataFrame, y: pd.Series, feature_names: List[str], output_dir: str) -> None:
    """Generate HTML report with feature importance, permutation importance and SHAP summary."""
    logger.info(f"Generating comprehensive report in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    plot_feature_importance(model, feature_names, os.path.join(output_dir, "feature_importance.png"))

    importance = compute_permutation_importance(model, X, y, feature_names)
    importance.to_csv(os.path.join(output_dir, "permutation_importance.csv"))
    plot_permutation_importance(importance, os.path.join(output_dir, "permutation_importance.png"))

    try:
        plot_shap_summary(model, X, os.path.join(output_dir, "shap_summary.png"))
        shap_section = '<img src="shap_summary.png" alt="SHAP Summary" width="600">'
    except Exception as e:
        logger.warning(f"Failed to generate SHAP summary plot: {e}")
        shap_section = f"<p>SHAP summary not available: {e}</p>"

    html_content = f"""
    <html>
    <head><title>Model Explainability Report</title></head>
    <body>
        <h1>Model Explainability Report</h1>

        <h2>Feature Importance (Global, impurity-based)</h2>
        <img src="feature_importance.png" alt="Feature Importance" width="600">

        <h2>Permutation Importance (Global, model-agnostic)</h2>
        <img src="permutation_importance.png" alt="Permutation Importance" width="600">
        {importance.to_html(float_format='{:.4f}'.format)}

        <h2>SHAP Summary (per-prediction contributions, positive class)</h2>
        {shap_section}
    </body>
    </html>
    """

    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w") as f:
        f.write(html_content)

    logger.info(f"Report generated successfully at {report_path}")
