import requests
import time
import random
import argparse
import os
import numpy as np
from sklearn.datasets import load_wine
from colorama import init, Fore

import sys

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

init(autoreset=True)

API_URL = os.getenv("API_URL", "http://localhost:8000")
EVIDENTLY_URL = os.getenv("EVIDENTLY_URL", "http://localhost:8001")
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://localhost:8080")
AIRFLOW_AUTH = (os.getenv("AIRFLOW_USER", "admin"), os.getenv("AIRFLOW_PASS", "admin"))
DAG_ID = "ml_training_pipeline"

N_NORMAL = 100
N_DRIFT = 250


def send_request(features):
    try:
        response = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def send_batch(samples, label, dry_run):
    """Send samples to the API, return the number of successful requests."""
    success = 0
    for i, sample in enumerate(samples, start=1):
        if dry_run or send_request(sample):
            success += 1
        if i % 50 == 0 or i == len(samples):
            print(f"Sent {i}/{len(samples)} {label} requests")
        time.sleep(0.01)
    return success


def analyze(window_size):
    """Run Evidently on the last `window_size` inferences captured from the API."""
    time.sleep(2)  # the API forwards inferences to Evidently in the background
    response = requests.post(f"{EVIDENTLY_URL}/analyze", json={"window_size": window_size}, timeout=120)
    response.raise_for_status()
    result = response.json()
    print(f"  samples analyzed : {result['current_samples']}")
    print(f"  drifted features : {result['drifted_count']}/{result['total_features']}")
    print(f"  dataset drift    : {result['drift_detected']}")
    print(f"  report           : {EVIDENTLY_URL}{result['report_url']}")
    if result['current_samples'] < window_size:
        print(Fore.YELLOW + f"  Warning: expected {window_size} samples, the API did not capture all inferences.")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Generate the traffic without sending any request")
    args = parser.parse_args()

    X = load_wine().data

    if not args.dry_run:
        # Start from an empty window so only this run's traffic is analyzed
        requests.delete(f"{EVIDENTLY_URL}/production-data", timeout=10).raise_for_status()

    print(Fore.CYAN + f"Step 1: Send {N_NORMAL} requests with normal data")
    normal = [random.choice(X).tolist() for _ in range(N_NORMAL)]
    normal_success = send_batch(normal, "normal", args.dry_run)
    print(Fore.GREEN + f"Summary Step 1: {normal_success}/{N_NORMAL} normal requests successful.")
    if not args.dry_run:
        baseline = analyze(N_NORMAL)
        if baseline["drift_detected"]:
            print(Fore.YELLOW + "Unexpected: drift flagged on normal traffic.")
    print()

    print(Fore.CYAN + f"Step 2: Send {N_DRIFT} requests with perturbed features (Drift)")
    drifted = []
    for _ in range(N_DRIFT):
        sample = random.choice(X)
        scale = random.uniform(1.5, 3.0)
        noise = np.random.normal(0, np.std(sample) * 0.5, len(sample))
        drifted.append((sample * scale + noise).tolist())
    drift_success = send_batch(drifted, "drift", args.dry_run)
    print(Fore.GREEN + f"Summary Step 2: {drift_success}/{N_DRIFT} drift requests successful.\n")

    if args.dry_run:
        print("Dry run: skipping Evidently check and Airflow verification.")
        return

    print(Fore.CYAN + "Step 3: Run Evidently Check")
    result = analyze(N_DRIFT)
    print()

    print(Fore.CYAN + "Step 4: Verify Trigger")
    if not result["drift_detected"]:
        print(Fore.RED + "No drift detected on drifted traffic.")
        sys.exit(1)
    if not result["retrain_triggered"]:
        print(Fore.RED + "Drift detected but Evidently failed to trigger the Airflow DAG (see Evidently logs).")
        sys.exit(1)

    # Confirm the DAG run really exists in Airflow
    run_url = f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns/{result['dag_run_id']}"
    response = requests.get(run_url, auth=AIRFLOW_AUTH, timeout=10)
    response.raise_for_status()
    print(Fore.GREEN + "Data drift detected -> Evidently triggered the Airflow retraining DAG.")
    print(f"  dag_run_id : {result['dag_run_id']}")
    print(f"  state      : {response.json()['state']}")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as e:
        print(Fore.RED + f"Request failed: {e}")
        sys.exit(1)
