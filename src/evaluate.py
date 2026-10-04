"""
Evaluation Module.
Handles model evaluation, comparison with production and registry promotion.
"""
import logging
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from typing import Dict, Any
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = 'wine_quality_model'


def evaluate_model(model, X_test, y_test) -> Dict[str, Any]:
    """Return dict with accuracy, f1_score, precision, recall, confusion_matrix."""
    logger.info("Evaluating model performance.")
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    logger.info(f"Evaluation metrics: accuracy={metrics['accuracy']:.4f}")
    return metrics


def load_production_model(model_name: str = MODEL_NAME):
    """Load the current Production model from the MLflow registry, or None if there is none yet."""
    versions = MlflowClient().search_model_versions(f"name='{model_name}'")
    if not any(v.current_stage == 'Production' for v in versions):
        logger.info(f"No Production version found for model {model_name}")
        return None
    return mlflow.sklearn.load_model(f"models:/{model_name}/Production")


def should_promote(candidate_metrics: Dict[str, float], production_metrics: Dict[str, float],
                   metric: str = 'accuracy') -> bool:
    """Candidate is promoted when it is at least as good as production, both measured on the same test set."""
    candidate_val = candidate_metrics[metric]
    prod_val = production_metrics[metric]
    logger.info(f"Candidate {metric}: {candidate_val:.4f}, Production {metric}: {prod_val:.4f}")
    return candidate_val >= prod_val


def register_and_promote(run_id: str, model_name: str = MODEL_NAME, stage: str = 'Production') -> str:
    """Register the run's model and move it to the given stage, archiving the previous one. Returns the version."""
    version = mlflow.register_model(f"runs:/{run_id}/model", model_name).version
    logger.info(f"Promoting {model_name} version {version} to {stage}")
    MlflowClient().transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True
    )
    return version
