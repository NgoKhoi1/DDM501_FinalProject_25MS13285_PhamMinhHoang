"""
Explainability Module.
Provides SHAP reports and feature importance visualizations.
"""
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Any

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

def generate_report(model: Any, X: pd.DataFrame, y: pd.Series, feature_names: List[str], output_dir: str) -> None:
    """Generate comprehensive HTML report with feature importance, SHAP summary."""
    logger.info(f"Generating comprehensive report in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate feature importance plot
    feat_imp_path = os.path.join(output_dir, "feature_importance.png")
    plot_feature_importance(model, feature_names, feat_imp_path)
    
    shap_msg = "SHAP values not available (library not installed)."
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        shap_plot_path = os.path.join(output_dir, "shap_summary.png")
        
        plt.figure(figsize=(10, 6))
        # Depending on multi-class, shap_values might be a list. Fallback to default summary plot
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, X, show=False)
            
        plt.savefig(shap_plot_path)
        plt.close()
        shap_msg = "SHAP summary plot generated."
    except Exception as e:
        logger.warning(f"Failed to generate SHAP summary plot: {e}")
        shap_plot_path = None
        
    html_content = f"""
    <html>
    <head><title>Model Explainability Report</title></head>
    <body>
        <h1>Model Explainability Report</h1>
        
        <h2>Feature Importance (Global)</h2>
        <img src="feature_importance.png" alt="Feature Importance" width="600">
        
        <h2>SHAP Summary</h2>
        <p>{shap_msg}</p>
    """
    
    if shap_plot_path:
         html_content += '\n        <img src="shap_summary.png" alt="SHAP Summary" width="600">'
         
    html_content += """
    </body>
    </html>
    """
    
    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w") as f:
        f.write(html_content)
        
    logger.info(f"Report generated successfully at {report_path}")

if __name__ == '__main__':
    # Test script
    pass
