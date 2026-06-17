import os
import sys
import time
import psutil
import numpy as np
import pandas as pd
from fastapi import FastAPI, Response, Request, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest, CONTENT_TYPE_LATEST
import mlflow.sklearn
from sklearn.metrics import accuracy_score

app = FastAPI(title="Heart Disease Model Serving & Monitoring API")

# Record start time for throughput calculation
startup_time = time.time()

# --- DEFINE PROMETHEUS METRICS (10 Metrics) ---
prediction_count = Counter(
    "prediction_count_total", 
    "Total number of predictions made by the model"
)
prediction_latency = Histogram(
    "prediction_latency_seconds", 
    "Time taken to run model inference"
)
request_count = Counter(
    "request_count_total", 
    "Total number of requests received by the API"
)
error_count = Counter(
    "error_count_total", 
    "Total number of errors encountered by the API"
)
prediction_success = Counter(
    "prediction_success_total",
    "Total number of successful predictions"
)
prediction_failed = Counter(
    "prediction_failed_total",
    "Total number of failed predictions"
)
cpu_usage = Gauge(
    "cpu_usage_percent", 
    "CPU utilization in percent"
)
memory_usage = Gauge(
    "memory_usage_percent", 
    "Memory utilization in percent"
)
disk_usage = Gauge(
    "disk_usage_percent", 
    "Disk space utilization in percent"
)
model_accuracy = Gauge(
    "model_accuracy_ratio", 
    "Test accuracy of the active model computed from test dataset"
)
throughput = Gauge(
    "api_throughput_requests_per_second", 
    "Throughput of the API in requests per second"
)
response_time = Summary(
    "api_response_time_seconds", 
    "Overall API response time in seconds"
)

# --- LOAD MODEL (No Dummy/Fallback - Fail Fast) ---
MODEL_PATH = "model"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = mlflow.sklearn.load_model(MODEL_PATH)
        print(f"[OK] Model loaded successfully from '{MODEL_PATH}/'.")
    except Exception as e:
        print(f"[ERROR] Failed to load model from '{MODEL_PATH}/': {e}")
        print("[ERROR] Please ensure a valid MLflow model exists in the 'model/' directory.")
        sys.exit(1)
else:
    print(f"[ERROR] Model directory '{MODEL_PATH}/' not found.")
    print("[ERROR] Please run modelling.py first to train and save the model.")
    sys.exit(1)

# --- COMPUTE REAL ACCURACY FROM TEST DATA ---
computed_accuracy = None
TEST_DATA_PATHS = [
    os.path.join("Membangun_Model", "dataset_preprocessed", "test.csv"),
    os.path.join("Membangun_model", "dataset_preprocessed", "test.csv"),
    os.path.join("dataset_preprocessed", "test.csv"),
    os.path.join("Eksperimen_SML_Kevin_Adiputra", "preprocessing", "dataset_preprocessed", "test.csv"),
    os.path.join("Eksperimen_SML_Kevin_Adiputra", "preprocessing", "dataset_preprocessing", "test.csv"),
]

test_data_path = None
for path in TEST_DATA_PATHS:
    if os.path.exists(path):
        test_data_path = path
        break

if test_data_path is not None:
    try:
        test_df = pd.read_csv(test_data_path)
        X_test = test_df.drop(columns=["target"])
        y_test = test_df["target"]
        y_pred = model.predict(X_test)
        computed_accuracy = accuracy_score(y_test, y_pred)
        model_accuracy.set(computed_accuracy)
        print(f"[OK] Model accuracy computed from test data: {computed_accuracy:.4f}")
    except Exception as e:
        print(f"[WARNING] Could not compute accuracy from test data: {e}")
        print("[WARNING] model_accuracy_ratio metric will remain at 0 until test data is available.")
else:
    print("[WARNING] Test dataset not found. model_accuracy_ratio will remain at 0.")
    print(f"[WARNING] Searched paths: {TEST_DATA_PATHS}")


# Define request schema matching the 22 preprocessed feature columns
class InferenceRequest(BaseModel):
    age: float
    trestbps: float
    chol: float
    thalach: float
    oldpeak: float
    chol_bps_ratio: float
    hr_age_ratio: float
    sex: int
    fbs: int
    exang: int
    ca: int
    # Encoded features (OHE output from preprocessor)
    cp_1: int = 0
    cp_2: int = 0
    cp_3: int = 0
    restecg_1: int = 0
    restecg_2: int = 0
    slope_1: int = 0
    slope_2: int = 0
    thal_1: int = 0
    thal_2: int = 0
    thal_3: int = 0
    age_group_1: int = 0
    age_group_2: int = 0


@app.middleware("http")
async def log_response_time(request: Request, call_next):
    """Middleware to measure the overall response time of the API."""
    request_count.inc()
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response_time.observe(process_time)
        return response
    except Exception as e:
        error_count.inc()
        raise e


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Heart Disease Serving & Monitoring System",
        "model_accuracy": computed_accuracy,
        "docs_url": "/docs"
    }


@app.post("/predict")
def predict(request: InferenceRequest):
    """Inference endpoint that returns target predictions using the real trained model."""
    start_inference_time = time.time()
    try:
        # Convert request body to dataframe matching expected shape
        input_data = pd.DataFrame([request.model_dump()])
        
        # Make prediction using real model
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][int(pred)]
        
        # Log latency
        inference_latency = time.time() - start_inference_time
        prediction_latency.observe(inference_latency)
        
        # Increment counters
        prediction_count.inc()
        prediction_success.inc()
        
        return {
            "prediction": int(pred),
            "probability": float(prob),
            "label": "Heart Disease" if pred == 1 else "Normal",
            "latency_seconds": inference_latency
        }
    except Exception as e:
        error_count.inc()
        prediction_failed.inc()
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")


@app.get("/metrics")
def metrics():
    """Prometheus metrics scrape endpoint. All metrics are from real inference."""
    # Update system resource stats
    cpu_usage.set(psutil.cpu_percent())
    memory_usage.set(psutil.virtual_memory().percent)
    disk_usage.set(psutil.disk_usage("/").percent)
    
    # Calculate throughput from real request counter
    uptime = time.time() - startup_time
    total_requests = request_count._value.get()
    throughput.set(total_requests / (uptime + 1e-5))
    
    # model_accuracy is already set at startup from real test data
    # No hardcoded values — accuracy comes from accuracy_score(y_test, y_pred)
    
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
