# AGENT DIRECTIVE: END-TO-END MLOPS FINAL PROJECT PIPELINE (DDM501)

## 1. PROJECT OBJECTIVE & OVERVIEW
Build a complete, production-ready, automated End-to-End MLOps Pipeline satisfying the requirements of course DDM501 and the instructor's architecture design:
- **Core Stack:** Python, Apache Airflow, MLflow, FastAPI, Docker & Docker Compose, Prometheus, Grafana, Evidently AI, GitHub Actions (Self-hosted runner).
- **Core Loop:**
  `Data -> Clean -> Feature Engineering -> Train (MLflow) -> Model Registry -> Serve (FastAPI) -> Monitor (Prometheus + Grafana + Evidently AI) -> Trigger Retrain on Drift`.

---

## 2. SYSTEM ARCHITECTURE & COMPONENTS

### A. Data & Training Pipeline (Apache Airflow + MLflow)
1. **DAG Workflow:**
   - `data_ingestion`: Ingest raw dataset (store artifacts/versioning).
   - `data_cleaning`: Handle nulls, outliers, type casting, schema validation.
   - `feature_engineering`: Transformations, scalers/encoders (persisted as artifacts).
   - `model_training`: Train model, log hyperparameters, metrics, and model artifacts to **MLflow Tracking Server**.
   - `model_evaluation_and_registration`: Compare candidate model against current production model or threshold. If passed, promote to `Staging`/`Production` in **MLflow Model Registry**.
2. **Automated Retraining Trigger:**
   - Expose Airflow DAG execution via Airflow REST API (`POST /api/v1/dags/{dag_id}/dagRuns`) or file/sensor-based trigger to allow automated retraining.

### B. Model Serving (FastAPI + Docker)
1. **API Endpoints:**
   - `GET /health`: Healthcheck endpoint for Docker Compose & Kubernetes/load balancers.
   - `POST /predict`: Input payload parsing, run inference using latest model from MLflow Registry / local artifact mount, return prediction and prediction_id.
   - `GET /metrics`: Expose Prometheus metrics (latency, total requests, error rates, prediction distribution).
2. **Inference Logging:**
   - Log input features and model outputs (with timestamps) into a database (PostgreSQL/SQLite) or persistent log file/volume for drift analysis.

### C. Monitoring Stack (Prometheus + Grafana + Evidently AI)
1. **Prometheus:**
   - Scrape `/metrics` from FastAPI service at defined intervals (`scrape_interval: 5s`).
2. **Grafana:**
   - Pre-configured dashboard (JSON provisioned) showing:
     - API Latency (p50, p95, p99).
     - Request count / throughput / error rate (2xx vs 5xx).
     - Model prediction distribution.
3. **Evidently AI (Drift Detection & Automated Retrain Trigger):**
   - Implement an automated drift monitoring job/service:
     - Compare **Reference Dataset** (training data) against **Current Inference Dataset** (logged production traffic).
     - Run `DataDriftPreset` / `TargetDriftPreset` using Evidently AI.
     - Generate drift test suites (`Suite.run()` or `Report.run()`).
   - **Crucial Instructor Requirement:**
     - When `dataset_drift` is detected (`drift_share` exceeds defined threshold, e.g. > 30% of features drift):
       - Log drift metrics / alerts.
       - Automatically trigger the **Airflow Retraining DAG** via Airflow REST API.

### D. CI/CD Pipeline (GitHub Actions - Self-Hosted Runner)
1. **Setup:**
   - Configured specifically with `runs-on: self-hosted`.
2. **Pipeline Steps:**
   - **Linting & Code Quality:** flake8 / black / ruff.
   - **Automated Testing:** `pytest` (Unit tests, Integration tests, Data quality & Model validation tests). Ensure code coverage $\ge 80\%$.
   - **Build & Package:** Build Docker image for FastAPI serving service.
   - **Deploy:** Deploy or restart services locally using Docker Compose.

---

## 3. KEY FEATURE: DRIFT SIMULATION & AUTOMATED RETRAINING

To demonstrate the full feedback loop during the final demo, implement a drift simulation script:
1. **Simulation Script (`scripts/simulate_drift_and_retrain.py`):**
   - **Step 1 (Normal Traffic):** Send 100 requests to `POST /predict` sampled from the normal test distribution.
   - **Step 2 (Simulate Drift):** Send 200+ requests with intentionally perturbed/shifted feature distributions (e.g., adding Gaussian noise $\mathcal{N}(\mu, \sigma^2)$, multiplying numeric values by a scaling factor, or shifting categorical frequencies).
   - **Step 3 (Run Evidently Check):** Run drift evaluation on newly collected inference data.
   - **Step 4 (Verify Trigger):** Verify that Evidently detects data drift, flags `dataset_drift = True`, sends an alert, and successfully invokes the Airflow REST API to trigger DAG retraining automatically.

---

## 4. PROJECT DIRECTORY STRUCTURE
Ensure the repository is organized cleanly as follows:

```text
├── .github/
│   └── workflows/
│       └── ci-cd.yml                # GitHub Actions workflow (runs-on: self-hosted)
├── airflow/
│   ├── dags/
│   │   └── ml_pipeline_dag.py       # Data -> Clean -> Feature Eng -> Train -> Register
│   └── docker-compose.airflow.yml
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml
│   ├── grafana/
│   │   ├── provisioning/
│   │   └── dashboards/
│   └── evidently/
│       ├── drift_monitor.py         # Drift detection & Airflow trigger logic
│       └── reference_data.csv
├── serving/
│   ├── app/
│   │   ├── main.py                  # FastAPI server with /predict, /health, /metrics
│   │   ├── config.py
│   │   └── schemas.py
│   └── Dockerfile
├── src/
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── explainability.py            # SHAP / Responsible AI report
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── scripts/
│   └── simulate_drift_and_retrain.py# Script to generate drifted data & trigger retrain
├── docker-compose.yml               # Multi-service setup (FastAPI, Prometheus, Grafana, MLflow)
├── requirements.txt
├── Makefile                         # Helper commands: make up, make test, make simulate-drift
└── README.md                        # Complete architecture diagram & setup instructions
```

---

## 5. STEP-BY-STEP IMPLEMENTATION TASKS FOR AGENT

1. **Phase 1: Environment & Dataset Selection**
   - Select a realistic tabular dataset (e.g., California Housing, Bank Churn, or Fraud Detection).
   - Prepare clean mock baseline and define MLflow tracking configurations.
2. **Phase 2: Airflow DAG & MLflow Integration**
   - Write clean, modular Python functions for each task in `src/`.
   - Build Airflow DAG with proper task dependencies: `ingest >> clean >> fe >> train >> register`.
3. **Phase 3: Model Serving & Prometheus Metrics**
   - Implement FastAPI with Prometheus Client instrumentation (`Counter`, `Histogram`).
   - Create local Dockerfile and container configuration.
4. **Phase 4: Monitoring (Prometheus + Grafana + Evidently AI) & Retrain Hook**
   - Provision Prometheus scrape targets and Grafana dashboards.
   - Implement `drift_monitor.py` using Evidently AI: calculate Data Drift, output drift HTML report, and make an HTTP POST call to trigger Airflow DAG if drift is detected.
5. **Phase 5: Drift Simulation Script**
   - Create the script `scripts/simulate_drift_and_retrain.py` to seamlessly demonstrate drift detection and automatic retraining during live grading.
6. **Phase 6: Testing & CI/CD**
   - Add unit and integration tests with pytest (coverage > 80%).
   - Create `.github/workflows/ci-cd.yml` configured for `runs-on: self-hosted`.
7. **Phase 7: Documentation**
   - Update `README.md` with:
     - Architecture diagram (ASCII / Mermaid).
     - Instructions to start the services with `docker-compose`.
     - Step-by-step guide to run the live drift demo.

---

## 6. DELIVERABLE ACCEPTANCE CRITERIA
- [ ] `docker compose up` starts all required services: Serving API, MLflow, Prometheus, Grafana, Airflow.
- [ ] Airflow DAG successfully trains model and saves artifacts in MLflow.
- [ ] Model can be called via FastAPI `/predict`.
- [ ] Prometheus scrapes `/metrics` and Grafana reflects live traffic.
- [ ] Running `simulate_drift_and_retrain.py` sends shifted data $\to$ Evidently flags Data Drift $\to$ Airflow DAG is automatically triggered to retrain the model.
- [ ] CI/CD pipeline runs successfully on a self-hosted runner with tests passing and code coverage $> 80\%$.