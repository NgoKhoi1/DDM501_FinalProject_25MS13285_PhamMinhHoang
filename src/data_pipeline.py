"""
Data Pipeline Module.
Handles data ingestion, cleaning, and persistence.
"""
import logging
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any
from sklearn.datasets import load_wine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ingest_data() -> Dict[str, Any]:
    """Load wine dataset from sklearn, return X, y, feature_names as dict."""
    logger.info("Ingesting wine dataset from sklearn.")
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    logger.info(f"Loaded dataset with shape {X.shape}")
    return {
        "X": X,
        "y": y,
        "feature_names": data.feature_names
    }

def clean_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Handle nulls (fill median), remove outliers (IQR method), type casting to float64, schema validation."""
    logger.info("Cleaning data: missing values, outliers, types, schema.")
    
    # Schema validation
    expected_features = 13
    if X.shape[1] != expected_features:
        logger.warning(f"Expected {expected_features} features, got {X.shape[1]}")
    
    # Type casting
    X = X.astype(np.float64)
    y = y.astype(np.int64)
    
    # Handle nulls
    if X.isnull().sum().sum() > 0:
        logger.info("Filling missing values with median.")
        X = X.fillna(X.median())
        
    # Remove outliers using IQR
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    
    # Mask for non-outliers
    mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
    
    X_cleaned = X[mask].reset_index(drop=True)
    y_cleaned = y[mask].reset_index(drop=True)
    
    logger.info(f"Data shape after cleaning: {X_cleaned.shape}")
    return X_cleaned, y_cleaned

def save_data_artifact(X: pd.DataFrame, y: pd.Series, feature_names: List[str], path: str) -> None:
    """Save as CSV for versioning."""
    logger.info(f"Saving data artifact to {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df = X.copy()
    df['target'] = y
    df.to_csv(path, index=False)
    logger.info("Data artifact saved successfully.")

def load_data_artifact(path: str) -> Dict[str, Any]:
    """Load from CSV."""
    logger.info(f"Loading data artifact from {path}")
    df = pd.read_csv(path)
    y = df['target']
    X = df.drop(columns=['target'])
    feature_names = X.columns.tolist()
    return {
        "X": X,
        "y": y,
        "feature_names": feature_names
    }

if __name__ == '__main__':
    # Test data pipeline
    data = ingest_data()
    X_c, y_c = clean_data(data['X'], data['y'])
    save_data_artifact(X_c, y_c, data['feature_names'], 'test_artifact.csv')
    loaded = load_data_artifact('test_artifact.csv')
    print("Test successful, loaded shape:", loaded['X'].shape)
