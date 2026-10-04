"""
============================================
INITIAL TRAINING
============================================

Triggers the Airflow training DAG on red wine only and waits for it to finish.
The DAG trains the model, registers it in MLflow (Production), reloads the API
and uploads the training data to Evidently as drift reference.
"""

import os
import sys
import time
import requests

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://localhost:8080")
AIRFLOW_AUTH = (os.getenv("AIRFLOW_USER", "admin"), os.getenv("AIRFLOW_PASS", "admin"))
DAG_ID = "ml_training_pipeline"
DAG_RUNS_URL = f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns"


def trigger_dag(conf):
    """Trigger the training DAG, return the dag_run_id."""
    response = requests.post(DAG_RUNS_URL, json={"conf": conf}, auth=AIRFLOW_AUTH, timeout=30)
    response.raise_for_status()
    return response.json()["dag_run_id"]


def wait_for_dag_run(dag_run_id, timeout=600):
    """Poll the DAG run until it finishes, return its final state ('success' or 'failed')."""
    deadline = time.time() + timeout
    state = None
    while time.time() < deadline:
        response = requests.get(f"{DAG_RUNS_URL}/{dag_run_id}", auth=AIRFLOW_AUTH, timeout=30)
        response.raise_for_status()
        if response.json()["state"] != state:
            state = response.json()["state"]
            print(f"  DAG run state: {state}")
        if state in ("success", "failed"):
            return state
        time.sleep(5)
    raise TimeoutError(f"DAG run {dag_run_id} did not finish within {timeout}s")


if __name__ == "__main__":
    print("🎯 Triggering initial training (red wine only)...")
    dag_run_id = trigger_dag({"colors": ["red"], "triggered_by": "initial_training"})
    print(f"  dag_run_id: {dag_run_id}")
    if wait_for_dag_run(dag_run_id) != "success":
        print(f"❌ Training failed, see {AIRFLOW_URL}/dags/{DAG_ID}/grid")
        sys.exit(1)
    print("✅ Model trained, promoted to Production and loaded by the API.")
