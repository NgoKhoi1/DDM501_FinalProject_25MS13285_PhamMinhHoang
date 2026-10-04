import os
import sys
import json
from datetime import datetime, timedelta

import mlflow
import requests
from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator

# Pipeline logic lives in src/ (mounted next to the dags folder)
sys.path.insert(0, os.getenv('SRC_DIR', '/opt/airflow/src'))

from data_pipeline import (  # noqa: E402
    COLORS, FEATURE_NAMES, TARGET,
    ingest_data, clean_data, split_data, save_data_artifact, load_data_artifact
)
from feature_engineering import fit_scaler, transform_features, save_scaler, load_scaler  # noqa: E402
from train import tune_model, build_serving_pipeline, log_to_mlflow  # noqa: E402
from fairness import fairness_report  # noqa: E402
from explainability import generate_report  # noqa: E402
from evaluate import evaluate_model, load_production_model, should_promote, register_and_promote  # noqa: E402

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
API_URL = os.getenv('API_URL', 'http://api:8000')
EVIDENTLY_URL = os.getenv('EVIDENTLY_URL', 'http://evidently:8001')

WORK_DIR = '/tmp/ml_pipeline'
RAW_PATH = f'{WORK_DIR}/raw_data.csv'
CLEAN_PATH = f'{WORK_DIR}/clean_data.csv'
TRAIN_PATH = f'{WORK_DIR}/train_data.csv'
TEST_PATH = f'{WORK_DIR}/test_data.csv'
SCALER_PATH = f'{WORK_DIR}/artifacts/scaler.pkl'
RUN_INFO_PATH = f'{WORK_DIR}/run_info.json'
REPORT_DIR = f'{WORK_DIR}/report'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}


def data_ingestion(dag_run, **kwargs):
    # Trigger with {"colors": ["red"]} to train on a subset; default is all the data
    colors = dag_run.conf.get('colors', list(COLORS))
    save_data_artifact(ingest_data(colors), RAW_PATH)


def data_cleaning(**kwargs):
    save_data_artifact(clean_data(load_data_artifact(RAW_PATH)), CLEAN_PATH)


def feature_engineering(**kwargs):
    train_df, test_df = split_data(load_data_artifact(CLEAN_PATH))
    save_data_artifact(train_df, TRAIN_PATH)
    save_data_artifact(test_df, TEST_PATH)
    save_scaler(fit_scaler(train_df[FEATURE_NAMES]), SCALER_PATH)


def model_training(dag_run, **kwargs):
    train_df = load_data_artifact(TRAIN_PATH)
    test_df = load_data_artifact(TEST_PATH)
    scaler = load_scaler(SCALER_PATH)

    # Hyperparameter tuning with 5-fold cross-validation on the training set
    search = tune_model(transform_features(train_df[FEATURE_NAMES], scaler), train_df[TARGET])
    model = search.best_estimator_
    pipeline = build_serving_pipeline(scaler, model)

    # Responsible AI: explainability report (on a sample of the test set) and fairness across wine colors
    sample = test_df.sample(min(300, len(test_df)), random_state=42)
    generate_report(
        model, transform_features(sample[FEATURE_NAMES], scaler), sample[TARGET],
        FEATURE_NAMES, f'{REPORT_DIR}/explainability'
    )
    fairness = fairness_report(pipeline, test_df[FEATURE_NAMES], test_df[TARGET], test_df['color'])

    run_id = log_to_mlflow(
        pipeline, test_df[FEATURE_NAMES], test_df[TARGET], search.best_params_,
        tags={
            'colors': ','.join(sorted(train_df['color'].unique())),
            'train_rows': len(train_df),
            'triggered_by': dag_run.conf.get('triggered_by', 'manual')
        },
        extra_metrics=fairness,
        artifacts_dir=REPORT_DIR,
        search=search
    )
    # Save run_id for next step
    with open(RUN_INFO_PATH, 'w') as f:
        json.dump({'run_id': run_id}, f)


def model_evaluation_and_registration(**kwargs):
    """Returns True (pushed to XCom) when the candidate was promoted to Production."""
    test_df = load_data_artifact(TEST_PATH)
    X_test, y_test = test_df[FEATURE_NAMES], test_df[TARGET]

    with open(RUN_INFO_PATH, 'r') as f:
        run_id = json.load(f)['run_id']

    candidate = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    candidate_metrics = evaluate_model(candidate, X_test, y_test)

    # Candidate and production are compared on the same test set
    production = load_production_model()
    promote = True
    if production is not None:
        production_metrics = evaluate_model(production, X_test, y_test)
        promote = should_promote(candidate_metrics, production_metrics)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric('production_accuracy', production_metrics['accuracy'])

    if promote:
        register_and_promote(run_id)
    return promote


def deploy_model(ti, **kwargs):
    if not ti.xcom_pull(task_ids='model_evaluation_and_registration'):
        raise AirflowSkipException("Candidate was not promoted, nothing to deploy")

    # Serving API picks up the new Production model
    requests.post(f"{API_URL}/model/reload", timeout=60).raise_for_status()

    # Drift is now measured against the data the new model was trained on
    train_df = load_data_artifact(TRAIN_PATH)
    requests.post(
        f"{EVIDENTLY_URL}/reference",
        json={
            'data': train_df[FEATURE_NAMES].to_dict(orient='records'),
            'feature_names': FEATURE_NAMES,
            'description': f"Training data ({','.join(sorted(train_df['color'].unique()))} wine)"
        },
        timeout=60
    ).raise_for_status()


with DAG(
    dag_id='ml_training_pipeline',
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,  # tasks share WORK_DIR
    tags=['ml', 'training', 'wine_quality'],
) as dag:

    t1 = PythonOperator(task_id='data_ingestion', python_callable=data_ingestion)
    t2 = PythonOperator(task_id='data_cleaning', python_callable=data_cleaning)
    t3 = PythonOperator(task_id='feature_engineering', python_callable=feature_engineering)
    t4 = PythonOperator(task_id='model_training', python_callable=model_training)
    t5 = PythonOperator(task_id='model_evaluation_and_registration', python_callable=model_evaluation_and_registration)
    t6 = PythonOperator(task_id='deploy_model', python_callable=deploy_model)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6
