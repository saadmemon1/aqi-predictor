import os
import joblib
from fastapi import FastAPI, HTTPException
import hopsworks
from pydantic import BaseModel
import pandas as pd
from src.pipelines.feature_pipeline import run_feature_pipeline

import time

app = FastAPI(title="AQI Predictor API")

model = None
explainer = None
fs = None
last_load_error = None

def load_model(retries=5, delay=3):
    global model, explainer, fs, last_load_error
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        last_load_error = "HOPSWORKS_API_KEY environment variable not found."
        print(f"Warning: {last_load_error}")
        return False
        
    for attempt in range(1, retries + 1):
        try:
            print(f"Attempting to login to Hopsworks (attempt {attempt}/{retries})...")
            project = hopsworks.login(api_key_value=api_key)
            fs = project.get_feature_store()
            mr = project.get_model_registry()
            
            print("Downloading model from Hopsworks...")
            aqi_model = mr.get_model("aqi_prediction_model", version=1)
            model_dir = aqi_model.download()
            
            model = joblib.load(os.path.join(model_dir, "aqi_model.pkl"))
            explainer = joblib.load(os.path.join(model_dir, "shap_explainer.pkl"))
            print("Model loaded successfully.")
            last_load_error = None
            return True
        except Exception as e:
            last_load_error = f"Attempt {attempt} failed: {str(e)}"
            print(f"Error loading model on attempt {attempt}: {e}")
            if attempt < retries:
                time.sleep(delay)
    return False

@app.on_event("startup")
def startup_load_model():
    # Attempt to load model on startup in the background or synchronously
    load_model(retries=3, delay=5)

@app.post("/run-feature-pipeline")
def api_run_feature_pipeline():
    """Triggered hourly by cron-job.org"""
    success = run_feature_pipeline()
    if success:
        return {"status": "success", "message": "Feature pipeline executed successfully"}
    else:
        raise HTTPException(status_code=500, detail="Feature pipeline failed")

@app.get("/predict")
def predict_aqi():
    """Fetches latest features from Hopsworks and returns 72h AQI prediction."""
    global model, fs, last_load_error
    if model is None or fs is None:
        load_model(retries=1, delay=0)
        
    if model is None or fs is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model or Feature Store not loaded. Last error: {last_load_error}"
        )
        
    try:
        # Get latest features
        feature_view = fs.get_feature_view("aqi_features_view", version=1)
        # Fetch the last row
        # In Hopsworks, getting a single feature vector:
        # We can use get_feature_vector(entry) if we know the primary key, but since we want the latest:
        # We can fetch the batch data and get the latest
        batch_data = feature_view.get_batch_data()
        batch_data.sort_values(by="timestamp", ascending=False, inplace=True)
        latest_features = batch_data.head(1).drop(['date', 'timestamp', 'target_aqi_24h', 'target_aqi_48h'], axis=1, errors='ignore')
        
        # Predict
        prediction = model.predict(latest_features)[0]
        
        return {
            "predicted_aqi_72h": float(prediction),
            "latest_timestamp": float(batch_data.iloc[0]['timestamp']),
            "features_used": latest_features.to_dict(orient="records")[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "healthy"}

@app.get("/test-dns")
def test_dns():
    import socket
    import urllib.request
    results = {}
    for host in ["google.com", "api.open-meteo.com", "c.app.hopsworks.ai", "huggingface.co"]:
        try:
            ip = socket.gethostbyname(host)
            results[host] = {"ip": ip, "status": "resolved"}
            try:
                urllib.request.urlopen(f"https://{host}", timeout=5)
                results[host]["http"] = "success"
            except Exception as e:
                results[host]["http"] = f"failed: {str(e)}"
        except Exception as e:
            results[host] = {"status": f"failed: {str(e)}"}
    return results
