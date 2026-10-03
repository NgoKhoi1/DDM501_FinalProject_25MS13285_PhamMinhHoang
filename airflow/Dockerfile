FROM apache/airflow:2.8.4-python3.10

# ML dependencies for the training DAG.
# numpy / scikit-learn are pinned to the same versions as api/requirements.txt
# so a model pickled by the DAG loads in the API (and the other way round).
RUN pip install --no-cache-dir \
    mlflow==2.16.2 \
    scikit-learn==1.5.2 \
    numpy==1.26.4 \
    shap==0.46.0
