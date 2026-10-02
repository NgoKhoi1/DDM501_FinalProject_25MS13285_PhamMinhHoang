import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import tempfile
import json
from airflow import DAG
from airflow.operators.python import PythonOperator
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow.tracking import MlflowClient

# MLflow env vars
os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID', 'minio')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', 'minio123')
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000')
os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def data_ingestion(**kwargs):
    os.makedirs('/tmp/ml_pipeline', exist_ok=True)
    wine = load_wine()
    df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    df.to_csv('/tmp/ml_pipeline/raw_data.csv', index=False)

def data_cleaning(**kwargs):
    df = pd.read_csv('/tmp/ml_pipeline/raw_data.csv')
    df = df.dropna()
    # Simple cleaning logic
    df.to_csv('/tmp/ml_pipeline/clean_data.csv', index=False)

def feature_engineering(**kwargs):
    df = pd.read_csv('/tmp/ml_pipeline/clean_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_scaled['target'] = y
    
    os.makedirs('/tmp/ml_pipeline/artifacts', exist_ok=True)
    joblib.dump(scaler, '/tmp/ml_pipeline/artifacts/scaler.pkl')
    X_scaled.to_csv('/tmp/ml_pipeline/transformed_data.csv', index=False)

def model_training(**kwargs):
    df = pd.read_csv('/tmp/ml_pipeline/transformed_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    mlflow.set_experiment("wine_quality_experiment")
    with mlflow.start_run() as run:
        model.fit(X, y)
        preds = model.predict(X)
        
        accuracy = accuracy_score(y, preds)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        
        # Save run_id for next step
        with open('/tmp/ml_pipeline/run_info.json', 'w') as f:
            json.dump({'run_id': run.info.run_id}, f)

def model_evaluation_and_registration(**kwargs):
    df = pd.read_csv('/tmp/ml_pipeline/transformed_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    with open('/tmp/ml_pipeline/run_info.json', 'r') as f:
        run_info = json.load(f)
    run_id = run_info['run_id']
    
    client = MlflowClient()
    
    # Load newly trained model
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    preds = model.predict(X)
    accuracy = accuracy_score(y, preds)
    
    # Check if a production model exists
    model_name = "wine_quality_model"
    try:
        latest_versions = client.get_latest_versions(model_name, stages=["Production"])
        if latest_versions:
            prod_version = latest_versions[0].version
            prod_model_uri = f"models:/{model_name}/Production"
            prod_model = mlflow.sklearn.load_model(prod_model_uri)
            prod_preds = prod_model.predict(X)
            prod_accuracy = accuracy_score(y, prod_preds)
            if accuracy > prod_accuracy:
                promote = True
            else:
                promote = False
        else:
            promote = True
    except Exception as e:
        promote = True
        
    if promote:
        model_version = mlflow.register_model(model_uri, model_name)
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True
        )

with DAG(
    dag_id='ml_training_pipeline',
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training', 'wine_quality'],
) as dag:
    
    t1 = PythonOperator(task_id='data_ingestion', python_callable=data_ingestion)
    t2 = PythonOperator(task_id='data_cleaning', python_callable=data_cleaning)
    t3 = PythonOperator(task_id='feature_engineering', python_callable=feature_engineering)
    t4 = PythonOperator(task_id='model_training', python_callable=model_training)
    t5 = PythonOperator(task_id='model_evaluation_and_registration', python_callable=model_evaluation_and_registration)
    
    t1 >> t2 >> t3 >> t4 >> t5
