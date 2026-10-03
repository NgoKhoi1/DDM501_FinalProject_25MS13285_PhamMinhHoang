"""
Drift demo: the model is trained on red wine, then production traffic shifts to white wine.

Step 1: normal traffic (red wine)   -> Evidently reports no drift
Step 2: drifted traffic (white wine)
Step 3: Evidently detects the drift and triggers the Airflow retraining DAG
Step 4: the DAG retrains on red + white wine and the API serves the new model
"""
import argparse
import csv
import os
import random
import sys
import time

import requests
from colorama import init, Fore

from training import AIRFLOW_URL, DAG_ID, wait_for_dag_run

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

init(autoreset=True)

API_URL = os.getenv("API_URL", "http://localhost:8000")
EVIDENTLY_URL = os.getenv("EVIDENTLY_URL", "http://localhost:8001")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

N_NORMAL = 100
N_DRIFT = 250


def load_samples(color, n):
    """Sample n rows of features (quality column dropped) from the wine CSV of the given color."""
    with open(os.path.join(DATA_DIR, f"winequality-{color}.csv"), newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))[1:]
    return [[float(v) for v in row[:-1]] for row in random.sample(rows, n)]


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


def model_version():
    response = requests.get(f"{API_URL}/health", timeout=10)
    response.raise_for_status()
    return response.json()["model_version"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Generate the traffic without sending any request")
    args = parser.parse_args()

    if not args.dry_run:
        version_before = model_version()
        print(f"API is serving model version {version_before}\n")
        # Start from an empty window so only this run's traffic is analyzed
        requests.delete(f"{EVIDENTLY_URL}/production-data", timeout=10).raise_for_status()

    print(Fore.CYAN + f"Step 1: Send {N_NORMAL} requests with normal data (red wine)")
    normal_success = send_batch(load_samples("red", N_NORMAL), "normal", args.dry_run)
    print(Fore.GREEN + f"Summary Step 1: {normal_success}/{N_NORMAL} normal requests successful.")
    if not args.dry_run:
        baseline = analyze(N_NORMAL)
        if baseline["drift_detected"]:
            print(Fore.YELLOW + "Unexpected: drift flagged on normal traffic.")
    print()

    print(Fore.CYAN + f"Step 2: Send {N_DRIFT} requests with drifted data (white wine)")
    drift_success = send_batch(load_samples("white", N_DRIFT), "drift", args.dry_run)
    print(Fore.GREEN + f"Summary Step 2: {drift_success}/{N_DRIFT} drift requests successful.\n")

    if args.dry_run:
        print("Dry run: skipping Evidently check and Airflow verification.")
        return

    print(Fore.CYAN + "Step 3: Run Evidently Check")
    result = analyze(N_DRIFT)
    if not result["drift_detected"]:
        print(Fore.RED + "No drift detected on drifted traffic.")
        sys.exit(1)
    if not result["retrain_triggered"]:
        print(Fore.RED + "Drift detected but Evidently failed to trigger the Airflow DAG (see Evidently logs).")
        sys.exit(1)
    print(Fore.GREEN + "Data drift detected -> Evidently triggered the Airflow retraining DAG.")
    print(f"  dag_run_id : {result['dag_run_id']}\n")

    print(Fore.CYAN + "Step 4: Verify Retraining")
    if wait_for_dag_run(result["dag_run_id"]) != "success":
        print(Fore.RED + f"Retraining failed, see {AIRFLOW_URL}/dags/{DAG_ID}/grid")
        sys.exit(1)
    version_after = model_version()
    if version_after == version_before:
        print(Fore.YELLOW + f"Retraining finished but the new model was not promoted (still version {version_before}).")
    else:
        print(Fore.GREEN + f"Retrained on red + white wine: API now serves model version {version_after} "
                           f"(was {version_before}).")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as e:
        print(Fore.RED + f"Request failed: {e}")
        sys.exit(1)
