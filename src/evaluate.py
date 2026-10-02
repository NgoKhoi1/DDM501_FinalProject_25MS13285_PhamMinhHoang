"""
Evaluation Module.
Handles model evaluation, metrics comparison and registry promotion.
"""
import logging
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test) -> Dict[str, Any]:
    """Return dict with accuracy, f1_score, precision, recall, confusion_matrix."""
    logger.info("Evaluating model performance.")
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    logger.info(f"Evaluation metrics: accuracy={metrics['accuracy']:.4f}")
    return metrics

def get_production_model_metrics(model_name: str) -> Dict[str, float]:
    """Fetch current production model's metrics from MLflow."""
    logger.info(f"Fetching production model metrics for {model_name}")
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        prod_versions = [v for v in versions if v.current_stage == 'Production']
        if not prod_versions:
            logger.warning(f"No Production version found for model {model_name}")
            return {}
        
        latest_prod = prod_versions[0]
        run_id = latest_prod.run_id
        run = mlflow.get_run(run_id)
        return run.data.metrics
    except Exception as e:
        logger.error(f"Error fetching production metrics: {e}")
        return {}

def compare_with_production(candidate_metrics: Dict[str, float], production_model_name: str = 'wine_quality_model', metric: str = 'accuracy', threshold: float = 0.01) -> bool:
    """Compare candidate vs current production model in MLflow registry."""
    logger.info("Comparing candidate model with production.")
    prod_metrics = get_production_model_metrics(production_model_name)
    
    if not prod_metrics:
        logger.info("No production model found. Candidate is automatically better.")
        return True
        
    candidate_val = candidate_metrics.get(metric, 0)
    prod_val = prod_metrics.get(metric, 0)
    
    logger.info(f"Candidate {metric}: {candidate_val}, Production {metric}: {prod_val}")
    if candidate_val >= prod_val + threshold:
        logger.info(f"Candidate improves {metric} by at least {threshold}.")
        return True
    
    logger.info(f"Candidate does not improve {metric} by {threshold}.")
    return False

def promote_model(model_name: str, version: int, stage: str = 'Production') -> None:
    """Transition model version to Production stage in MLflow registry, archive old versions."""
    logger.info(f"Promoting {model_name} version {version} to {stage}")
    client = MlflowClient()
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True
        )
        logger.info(f"Successfully promoted model to {stage}.")
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")

if __name__ == '__main__':
    # Test functions
    pass
