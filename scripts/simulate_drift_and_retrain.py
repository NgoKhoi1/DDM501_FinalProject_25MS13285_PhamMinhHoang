import requests
import time
import random
import argparse
import numpy as np
from sklearn.datasets import load_wine
from colorama import init, Fore, Style

import sys

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

init(autoreset=True)

API_URL = "http://localhost:8000/predict"
EVIDENTLY_URL = "http://localhost:8001/analyze"
AIRFLOW_DAG_URL = "http://localhost:8080/api/v1/dags/ml_training_pipeline/dagRuns"

def send_request(features):
    try:
        payload = {"features": features}
        response = requests.post(API_URL, json=payload, timeout=2)
        return response.status_code == 200
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run without sending actual requests")
    args = parser.parse_args()

    data = load_wine()
    X = data.data.tolist()

    print(Fore.CYAN + "Step 1: Send 100 requests with normal data")
    normal_success = 0
    for i in range(100):
        sample = random.choice(X)
        if not args.dry_run:
            if send_request(sample): normal_success += 1
        else:
            normal_success += 1
        if (i+1) % 20 == 0:
            print(f"Sent {i+1}/100 normal requests")
        time.sleep(0.01)
    
    print(Fore.GREEN + f"Summary Step 1: {normal_success}/100 normal requests successful.\n")

    print(Fore.CYAN + "Step 2: Send 200+ requests with perturbed features (Drift)")
    drift_success = 0
    for i in range(250):
        sample = np.array(random.choice(X))
        scale = random.uniform(1.5, 3.0)
        noise = np.random.normal(0, np.std(sample)*0.5, len(sample))
        drifted_sample = (sample * scale + noise).tolist()
        
        if not args.dry_run:
            if send_request(drifted_sample): drift_success += 1
        else:
            drift_success += 1
        if (i+1) % 50 == 0:
            print(f"Sent {i+1}/250 drift requests")
        time.sleep(0.01)

    print(Fore.GREEN + f"Summary Step 2: {drift_success}/250 drift requests successful.\n")

    print(Fore.CYAN + "Step 3: Run Evidently Check")
    drift_detected = False
    if not args.dry_run:
        try:
            response = requests.post(EVIDENTLY_URL, timeout=5)
            if response.status_code == 200:
                result = response.json()
                drift_detected = result.get("dataset_drift", True)
                print("Evidently analysis completed.")
            else:
                print(Fore.YELLOW + f"Evidently returned status code {response.status_code}")
                drift_detected = True
        except Exception as e:
            print(Fore.RED + f"Could not connect to Evidently: {e}")
            drift_detected = True
    else:
        print("Dry run: Skipping Evidently check.")
        drift_detected = True

    print(Fore.CYAN + "Step 4: Verify Trigger")
    if drift_detected:
        print(Fore.RED + "Data drift detected! Triggering Airflow DAG...")
        if not args.dry_run:
            try:
                auth = ("airflow", "airflow")
                response = requests.post(AIRFLOW_DAG_URL, json={}, auth=auth, timeout=5)
                if response.status_code in [200, 201, 202, 204]:
                    print(Fore.GREEN + "DAG triggered successfully!")
                else:
                    print(Fore.YELLOW + f"Failed to trigger DAG, status: {response.status_code}")
            except Exception as e:
                print(Fore.RED + f"Could not connect to Airflow: {e}")
        else:
            print(Fore.GREEN + "Dry run: DAG triggered successfully!")
    else:
        print(Fore.GREEN + "No drift detected. No action required.")

if __name__ == "__main__":
    main()
