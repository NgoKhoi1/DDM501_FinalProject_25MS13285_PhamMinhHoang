"""
Feature Engineering Module.
Handles feature transformations and engineering.
"""
import logging
import os
import joblib
import pandas as pd
from typing import Tuple, List, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_feature_pipeline() -> Pipeline:
    """Return sklearn Pipeline with StandardScaler."""
    logger.info("Creating feature pipeline with StandardScaler.")
    return Pipeline([
        ('scaler', StandardScaler())
    ])

def fit_transform_features(X_train: pd.DataFrame, feature_names: List[str]) -> Tuple[pd.DataFrame, Pipeline]:
    """Fit scaler, return transformed X_train and fitted scaler. Also add engineered features."""
    logger.info("Fitting and transforming features.")
    
    X = X_train.copy()
    
    # Feature engineering: Add interaction features
    # Proxy for alcohol density: alcohol / total_phenols
    if 'alcohol' in X.columns and 'total_phenols' in X.columns:
        X['alcohol_phenols_ratio'] = X['alcohol'] / (X['total_phenols'] + 1e-5)
    # Acidity ratio: malic_acid / ash
    if 'malic_acid' in X.columns and 'ash' in X.columns:
        X['acidity_ratio'] = X['malic_acid'] / (X['ash'] + 1e-5)
        
    pipeline = create_feature_pipeline()
    X_scaled = pipeline.fit_transform(X)
    
    X_transformed = pd.DataFrame(X_scaled, columns=X.columns)
    return X_transformed, pipeline

def transform_features(X: pd.DataFrame, scaler: Pipeline) -> pd.DataFrame:
    """Transform with existing scaler."""
    logger.info("Transforming features with existing scaler.")
    X_new = X.copy()
    
    if 'alcohol' in X_new.columns and 'total_phenols' in X_new.columns:
        X_new['alcohol_phenols_ratio'] = X_new['alcohol'] / (X_new['total_phenols'] + 1e-5)
    if 'malic_acid' in X_new.columns and 'ash' in X_new.columns:
        X_new['acidity_ratio'] = X_new['malic_acid'] / (X_new['ash'] + 1e-5)
        
    X_scaled = scaler.transform(X_new)
    return pd.DataFrame(X_scaled, columns=X_new.columns)

def save_scaler(scaler: Pipeline, path: str) -> None:
    """Save scaler to disk with joblib."""
    logger.info(f"Saving scaler to {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(scaler, path)

def load_scaler(path: str) -> Pipeline:
    """Load scaler from disk."""
    logger.info(f"Loading scaler from {path}")
    return joblib.load(path)

if __name__ == '__main__':
    # Test feature engineering
    df = pd.DataFrame({'alcohol': [13.0, 14.0], 'total_phenols': [2.5, 3.0], 'malic_acid': [1.5, 2.0], 'ash': [2.0, 2.5]})
    X_trans, scaler = fit_transform_features(df, df.columns.tolist())
    save_scaler(scaler, 'test_scaler.pkl')
    loaded_scaler = load_scaler('test_scaler.pkl')
    print("Test successful, transformed shape:", X_trans.shape)
