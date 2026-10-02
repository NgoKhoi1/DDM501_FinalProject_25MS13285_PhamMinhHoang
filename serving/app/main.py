import os
import json
import logging
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
import pandas as pd
import mlflow
from typing import List
from .config import settings
from .schemas import PredictionRequest, BatchPredictionRequest, PredictionResponse, HealthResponse

os.environ["MLFLOW_TRACKING_URI"] = settings.MLFLOW_TRACKING_URI
os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_SECRET_ACCESS_KEY
os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.MLFLOW_S3_ENDPOINT_URL
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Wine Quality Prediction API")

PREDICTION_COUNTER = Counter("predictions_total", "Total predictions made")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")

model = None
model_version = "unknown"

def load_model():
    global model, model_version
    try:
        model_uri = f"models:/{settings.MODEL_NAME}/{settings.MODEL_STAGE}"
        model = mlflow.sklearn.load_model(model_uri)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(settings.MODEL_NAME, stages=[settings.MODEL_STAGE])
        if versions:
            model_version = versions[0].version
        logger.info(f"Loaded model version: {model_version}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

@app.on_event("startup")
def startup_event():
    load_model()
    Instrumentator().instrument(app).expose(app)
    os.makedirs(os.path.dirname(settings.INFERENCE_LOG_PATH), exist_ok=True)

def log_inference(features: dict, prediction: int, version: str):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": version,
        "prediction": prediction
    }
    log_entry.update(features)
    
    with open(settings.INFERENCE_LOG_PATH, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    PREDICTION_COUNTER.inc()
    with PREDICTION_LATENCY.time():
        features = request.dict()
        df = pd.DataFrame([features])
        prediction = int(model.predict(df)[0])
        background_tasks.add_task(log_inference, features, prediction, model_version)
        return PredictionResponse(prediction=prediction, model_version=model_version)

@app.post("/batch_predict")
def batch_predict(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    predictions = []
    features_list = [item.dict() for item in request.data]
    df = pd.DataFrame(features_list)
    preds = model.predict(df)
    
    for i, pred in enumerate(preds):
        prediction = int(pred)
        predictions.append({"prediction": prediction, "model_version": model_version})
        background_tasks.add_task(log_inference, features_list[i], prediction, model_version)
        PREDICTION_COUNTER.inc()
        
    return {"predictions": predictions}

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="healthy", model_version=model_version)

@app.get("/model/info")
def model_info():
    return {"model_name": settings.MODEL_NAME, "version": model_version, "stage": settings.MODEL_STAGE}

@app.post("/model/reload")
def reload_model():
    load_model()
    return {"status": "success", "model_version": model_version}
