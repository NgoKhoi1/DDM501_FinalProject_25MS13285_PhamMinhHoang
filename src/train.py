"""
Training Module.
Trains the model and logs to MLflow.
"""
import logging
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Import modules from same directory
from evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = 'wine_quality_experiment'
DEFAULT_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'random_state': 42
}


def train_model(X_train, y_train, params: Optional[Dict[str, Any]] = None) -> RandomForestClassifier:
    """Train RandomForestClassifier, return model."""
    if params is None:
        params = DEFAULT_PARAMS
    logger.info(f"Training RandomForestClassifier with params: {params}")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def build_serving_pipeline(scaler: StandardScaler, model: RandomForestClassifier) -> Pipeline:
    """Bundle the fitted scaler and model so the served model takes raw features."""
    return Pipeline([('scaler', scaler), ('model', model)])


def log_to_mlflow(pipeline: Pipeline, X_test, y_test, params: Dict[str, Any],
                  tags: Optional[Dict[str, Any]] = None) -> str:
    """Log params, test metrics, confusion matrix and the model to MLflow. Returns the run id."""
    logger.info("Logging to MLflow...")
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)

        metrics = evaluate_model(pipeline, X_test, y_test)
        cm = metrics.pop('confusion_matrix')
        mlflow.log_metrics(metrics)
        logger.info(f"Logged metrics: {metrics}")

        # Artifacts: Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model", input_example=X_test.head(5))
        logger.info("MLflow logging complete.")
        return run.info.run_id
