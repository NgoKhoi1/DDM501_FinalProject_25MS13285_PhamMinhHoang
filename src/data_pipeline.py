"""
Data Pipeline Module.
Handles data ingestion, cleaning, splitting and persistence.

Dataset: UCI Wine Quality (Cortez et al., 2009), one CSV per wine color.
Task: binary classification, is the wine good (quality >= 6) or not.
"""
import logging
import os
import pandas as pd
from typing import Sequence, Tuple
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = os.getenv('DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
COLORS = ('red', 'white')
FEATURE_NAMES = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
    'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'
]
TARGET = 'target'
GOOD_QUALITY_THRESHOLD = 6


def ingest_data(colors: Sequence[str] = COLORS) -> pd.DataFrame:
    """Load the raw wine CSVs for the given colors: 11 features + quality + color."""
    frames = []
    for color in colors:
        path = os.path.join(DATA_DIR, f'winequality-{color}.csv')
        logger.info(f"Ingesting {path}")
        df = pd.read_csv(path, sep=';')
        df.columns = [c.replace(' ', '_') for c in df.columns]
        df['color'] = color
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded dataset with shape {data.shape}")
    return data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Schema validation, type casting, nulls (fill median), duplicates, binary target."""
    logger.info("Cleaning data: schema, types, missing values, duplicates, target.")

    # Schema validation
    missing = [c for c in FEATURE_NAMES + ['quality', 'color'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()

    # Type casting
    df[FEATURE_NAMES] = df[FEATURE_NAMES].astype('float64')

    # Handle nulls
    if df[FEATURE_NAMES].isnull().values.any():
        logger.info("Filling missing values with median.")
        df[FEATURE_NAMES] = df[FEATURE_NAMES].fillna(df[FEATURE_NAMES].median())

    # Duplicated rows would leak between train and test
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Removed {before - len(df)} duplicated rows")

    # Binary target: good wine or not
    df[TARGET] = (df['quality'] >= GOOD_QUALITY_THRESHOLD).astype('int64')
    df = df.drop(columns=['quality'])

    logger.info(f"Data shape after cleaning: {df.shape}")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split, done per color so a color's test rows never depend on the other colors."""
    train_parts, test_parts = [], []
    for _, group in df.groupby('color', sort=True):
        train, test = train_test_split(
            group, test_size=test_size, random_state=random_state, stratify=group[TARGET]
        )
        train_parts.append(train)
        test_parts.append(test)
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    logger.info(f"Split data: {len(train_df)} train rows, {len(test_df)} test rows")
    return train_df, test_df


def save_data_artifact(df: pd.DataFrame, path: str) -> None:
    """Save as CSV for versioning."""
    logger.info(f"Saving data artifact to {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)


def load_data_artifact(path: str) -> pd.DataFrame:
    """Load from CSV."""
    logger.info(f"Loading data artifact from {path}")
    return pd.read_csv(path)
