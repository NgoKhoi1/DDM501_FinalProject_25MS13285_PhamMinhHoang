"""
Training Module.
Trains the model and logs to MLflow.
"""
import logging
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

# Import modules from same directory
from data_pipeline import ingest_data, clean_data
from feature_engineering import fit_transform_features, transform_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(X_train, y_train, params: Optional[Dict[str, Any]] = None) -> RandomForestClassifier:
    """Train RandomForestClassifier, return model."""
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'random_state': 42
        }
    logger.info(f"Training RandomForestClassifier with params: {params}")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

def log_to_mlflow(model, X_train, y_train, X_test, y_test, params, feature_names):
    """Log everything to MLflow."""
    logger.info("Logging to MLflow...")
    
    # MLflow settings
    os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID', 'minio')
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', 'minio123')
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000')
    os.environ['MLFLOW_S3_IGNORE_TLS'] = os.getenv('MLFLOW_S3_IGNORE_TLS', 'true')
    
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment('wine_quality_experiment')
    
    with mlflow.start_run():
        # Log params
        mlflow.log_params(params)
        
        # Evaluate metrics
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted')
        }
        mlflow.log_metrics(metrics)
        logger.info(f"Logged metrics: {metrics}")
        
        # Artifacts: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax)
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close(fig)
        mlflow.log_artifact(cm_path)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name='wine_quality_model'
        )
        logger.info("MLflow logging complete.")

def run_training_pipeline():
    """Full pipeline that calls ingest→clean→FE→train→log."""
    logger.info("Starting training pipeline...")
    data = ingest_data()
    X, y = clean_data(data['X'], data['y'])
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_tf, scaler = fit_transform_features(X_train_raw, data['feature_names'])
    X_test_tf = transform_features(X_test_raw, scaler)
    
    params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2, 'random_state': 42}
    model = train_model(X_train_tf, y_train, params)
    
    log_to_mlflow(model, X_train_tf, y_train, X_test_tf, y_test, params, data['feature_names'])
    logger.info("Training pipeline completed successfully.")

if __name__ == '__main__':
    # run_training_pipeline()
    pass
