import os

class Settings:
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    MODEL_NAME = os.getenv('MODEL_NAME', 'wine_quality_model')
    MODEL_STAGE = os.getenv('MODEL_STAGE', 'Production')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', 'minio')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'minio123')
    MLFLOW_S3_ENDPOINT_URL = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000')
    INFERENCE_LOG_PATH = os.getenv('INFERENCE_LOG_PATH', '/app/logs/inference_log.jsonl')
    
settings = Settings()
