import pytest
import sys
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_train_model(wine_data):
    X, y, _ = wine_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    assert hasattr(model, 'predict')

def test_model_accuracy(wine_data):
    X, y, _ = wine_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    assert acc > 0.85

def test_model_prediction_shape(trained_model, wine_data):
    X, y, _ = wine_data
    preds = trained_model.predict(X[:5])
    assert preds.shape == (5,)

def test_model_feature_importance(trained_model):
    importances = trained_model.feature_importances_
    assert importances is not None
    assert np.isclose(importances.sum(), 1.0)

def test_evaluate_model(wine_data):
    X, y, _ = wine_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    precision = precision_score(y_test, preds, average='weighted')
    recall = recall_score(y_test, preds, average='weighted')
    
    assert acc is not None
    assert f1 is not None
    assert precision is not None
    assert recall is not None

def test_model_classes(trained_model):
    assert len(trained_model.classes_) == 3

def test_model_reproducibility(wine_data):
    X, y, _ = wine_data
    model1 = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    model2 = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    assert np.array_equal(model1.predict(X), model2.predict(X))
