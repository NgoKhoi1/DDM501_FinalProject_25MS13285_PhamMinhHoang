"""
Feature Engineering Module.
Fits the feature scaler and persists it as an artifact.
"""
import logging
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the training features only."""
    logger.info("Fitting StandardScaler on training features.")
    return StandardScaler().fit(X_train)


def transform_features(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Transform with an already fitted scaler."""
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)


def save_scaler(scaler: StandardScaler, path: str) -> None:
    """Save scaler to disk with joblib."""
    logger.info(f"Saving scaler to {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: str) -> StandardScaler:
    """Load scaler from disk."""
    logger.info(f"Loading scaler from {path}")
    return joblib.load(path)
