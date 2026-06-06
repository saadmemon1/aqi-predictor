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
            project = hopsworks.login(host="eu-west.cloud.hopsworks.ai", api_key_value=api_key)
            fs = project.get_feature_store()
            mr = project.get_model_registry()
            
            print("Downloading best model from Hopsworks...")
            aqi_model = mr.get_best_model("aqi_prediction_model", "R2", "max")
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
        feature_view = fs.get_feature_view("aqi_features_view", version=2)
        # Fetch the last row
        # In Hopsworks, getting a single feature vector:
        # We can use get_feature_vector(entry) if we know the primary key, but since we want the latest:
        # We can fetch the batch data and get the latest
        batch_data = feature_view.get_batch_data()
        batch_data.sort_values(by="timestamp", ascending=False, inplace=True)
        latest_features = batch_data.head(1).drop(['date', 'timestamp', 'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h'], axis=1, errors='ignore')
        
        # Align column order with the training features expected by the model
        if hasattr(model, "feature_names_in_"):
            latest_features = latest_features[list(model.feature_names_in_)]
        
        features_dict = latest_features.to_dict(orient="records")[0]
        print(f"API Debug - Features passed to model: {features_dict}")
        
        # Predict raw values
        raw_pred = model.predict(latest_features)[0]
        
        # Check if raw_pred is a multi-output array/list
        if hasattr(raw_pred, "__len__") and len(raw_pred) >= 3:
            pred_24h = max(0.0, float(raw_pred[0]))
            pred_48h = max(0.0, float(raw_pred[1]))
            pred_72h = max(0.0, float(raw_pred[2]))
        else:
            # Fallback for single-output model
            single_val = max(0.0, float(raw_pred))
            pred_24h = single_val
            pred_48h = single_val
            pred_72h = single_val
            
        # Calculate SHAP values for feature contribution explanation
        shap_contrib = {}
        try:
            if explainer is not None:
                sv = explainer.shap_values(latest_features)
                if isinstance(sv, list):
                    sv = sv[0]
                if len(sv.shape) == 2:
                    sv = sv[0]
                for col, val in zip(latest_features.columns, sv):
                    shap_contrib[col] = float(val)
        except Exception as shap_err:
            print(f"Error generating SHAP explanation: {shap_err}")
        
        return {
            "predicted_aqi_24h": pred_24h,
            "predicted_aqi_48h": pred_48h,
            "predicted_aqi_72h": pred_72h,
            "latest_timestamp": float(batch_data.iloc[0]['timestamp']),
            "features_used": latest_features.to_dict(orient="records")[0],
            "shap_explanation": shap_contrib
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "healthy"}
