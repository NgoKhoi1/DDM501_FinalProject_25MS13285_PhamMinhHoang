"""
Training Module.
Trains the model (grid search with cross-validation) and logs to MLflow.
"""
import logging
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Import modules from same directory
from evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = 'wine_quality_experiment'
RANDOM_STATE = 42
DEFAULT_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'random_state': RANDOM_STATE
}
PARAM_GRID = {
    'n_estimators': [100, 300],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [1, 3],
}


def train_model(X_train, y_train, params: Optional[Dict[str, Any]] = None) -> RandomForestClassifier:
    """Train RandomForestClassifier, return model."""
    if params is None:
        params = DEFAULT_PARAMS
    logger.info(f"Training RandomForestClassifier with params: {params}")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def tune_model(X_train, y_train, param_grid: Optional[Dict[str, List[Any]]] = None, cv: int = 5) -> GridSearchCV:
    """Grid search with shuffled stratified k-fold cross-validation.

    Returns the fitted search: `best_estimator_` is refit on the whole training set,
    `cv_results_` holds the cross-validated accuracy of every candidate.
    """
    search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid or PARAM_GRID,
        # Shuffle: the training rows are ordered by wine color, unshuffled folds would be single-color
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        scoring='accuracy',
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    logger.info(f"Best params: {search.best_params_} (CV accuracy {search.best_score_:.4f})")
    return search


def build_serving_pipeline(scaler: StandardScaler, model: RandomForestClassifier) -> Pipeline:
    """Bundle the fitted scaler and model so the served model takes raw features."""
    return Pipeline([('scaler', scaler), ('model', model)])


def log_to_mlflow(pipeline: Pipeline, X_test, y_test, params: Dict[str, Any],
                  tags: Optional[Dict[str, Any]] = None,
                  extra_metrics: Optional[Dict[str, float]] = None,
                  artifacts_dir: Optional[str] = None,
                  search: Optional[GridSearchCV] = None) -> str:
    """Log params, test metrics, confusion matrix and the model to MLflow. Returns the run id.

    extra_metrics: additional metrics (e.g. fairness) logged on the run.
    artifacts_dir: local directory uploaded as run artifacts (e.g. the explainability report).
    search: fitted grid search; every candidate is logged as a nested run with its CV accuracy.
    """
    logger.info("Logging to MLflow...")
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)

        metrics = evaluate_model(pipeline, X_test, y_test)
        cm = metrics.pop('confusion_matrix')
        mlflow.log_metrics({**metrics, **(extra_metrics or {})})
        logger.info(f"Logged metrics: {metrics}")

        if search is not None:
            results = search.cv_results_
            mlflow.log_metrics({
                'cv_accuracy_mean': results['mean_test_score'][search.best_index_],
                'cv_accuracy_std': results['std_test_score'][search.best_index_],
            })
            for candidate, mean, std in zip(results['params'], results['mean_test_score'], results['std_test_score']):
                with mlflow.start_run(nested=True, run_name='cv_candidate'):
                    mlflow.log_params(candidate)
                    mlflow.log_metrics({'cv_accuracy_mean': mean, 'cv_accuracy_std': std})

        # Artifacts: Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        if artifacts_dir:
            mlflow.log_artifacts(artifacts_dir)

        mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model", input_example=X_test.head(5))
        logger.info("MLflow logging complete.")
        return run.info.run_id
