import os
import json
import logging
import pandas as pd
import requests
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
from fastapi import FastAPI
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Evidently Drift Monitor API")

REFERENCE_DATA_PATH = os.getenv('REFERENCE_DATA_PATH', '/app/monitoring/evidently/reference_data.csv')
INFERENCE_LOG_PATH = os.getenv('INFERENCE_LOG_PATH', '/app/logs/inference_log.jsonl')
AIRFLOW_URL = os.getenv('AIRFLOW_URL', 'http://airflow-webserver:8080')
AIRFLOW_USER = os.getenv('AIRFLOW_USER', 'admin')
AIRFLOW_PASS = os.getenv('AIRFLOW_PASS', 'admin')

def load_data():
    if not os.path.exists(REFERENCE_DATA_PATH):
        logger.error(f"Reference data not found at {REFERENCE_DATA_PATH}")
        return None, None
        
    reference_data = pd.read_csv(REFERENCE_DATA_PATH)
    
    if not os.path.exists(INFERENCE_LOG_PATH):
        logger.error(f"Inference log not found at {INFERENCE_LOG_PATH}")
        return reference_data, None
        
    current_data_records = []
    with open(INFERENCE_LOG_PATH, 'r') as f:
        for line in f:
            if line.strip():
                current_data_records.append(json.loads(line))
                
    if not current_data_records:
        return reference_data, None
        
    current_data = pd.DataFrame(current_data_records)
    # Drop columns not in reference data (like timestamp, model_version, prediction)
    features = [col for col in reference_data.columns if col != 'target']
    
    # We might only have features in current_data, and prediction maps to target
    if 'prediction' in current_data.columns:
        current_data['target'] = current_data['prediction']
        
    cols_to_keep = features + ['target']
    current_data = current_data[[col for col in cols_to_keep if col in current_data.columns]]
    
    return reference_data, current_data

def trigger_airflow_dag():
    dag_id = "ml_training_pipeline"
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns"
    try:
        response = requests.post(
            url,
            auth=(AIRFLOW_USER, AIRFLOW_PASS),
            json={"conf": {"triggered_by": "evidently_drift_monitor"}}
        )
        if response.status_code == 200:
            logger.info("Successfully triggered Airflow DAG.")
        else:
            logger.error(f"Failed to trigger DAG: {response.status_code} {response.text}")
    except Exception as e:
        logger.error(f"Error triggering DAG: {e}")

@app.get("/monitor")
def run_monitor():
    reference_data, current_data = load_data()
    if current_data is None or len(current_data) == 0:
        return {"status": "skipped", "reason": "No current data available"}
        
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    
    report_dict = report.as_dict()
    
    # Find dataset drift metric
    drift_metric = None
    for metric in report_dict['metrics']:
        if metric['metric'] == 'DatasetDriftMetric':
            drift_metric = metric['result']
            break
            
    is_drifted = False
    if drift_metric:
        drift_share = drift_metric['drift_share']
        is_drifted = drift_metric['dataset_drift']
        logger.info(f"Dataset drift share: {drift_share}")
        
        if is_drifted or drift_share > 0.3:
            logger.warning("Data drift detected! Triggering retraining pipeline...")
            trigger_airflow_dag()
    
    # Generate HTML report
    report_path = "/app/monitoring/evidently/drift_report.html"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report.save_html(report_path)
    
    return {
        "status": "success",
        "drift_detected": is_drifted,
        "report_generated": report_path
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run_monitor()
    else:
        uvicorn.run(app, host="0.0.0.0", port=8001)
