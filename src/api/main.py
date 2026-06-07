import os
import joblib
from fastapi import FastAPI, HTTPException
import hopsworks
from pydantic import BaseModel
import pandas as pd
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
        # Get latest features from Hopsworks Feature Group
        fg = fs.get_feature_group("aqi_features", version=1)
        
        # Read all data via Feature Query Service (~30K rows, takes ~1 second)
        df = fg.read()
        df.sort_values(by="timestamp", ascending=False, inplace=True)
        
        # Ensure we only pass the exactly expected 18 features in the correct order
        model_features = [
            'temperature', 'humidity', 'precipitation', 'pressure', 
            'wind_speed', 'wind_direction', 'pm10', 'pm2_5', 
            'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 
            'ozone', 'aqi', 'hour', 'day_of_week', 'month', 
            'aqi_24h_rolling_mean', 'aqi_change_rate'
        ]
        latest_features = df.head(1)[model_features]
        
        # Align column order with model expectations
        if hasattr(model, "feature_names_in_"):
            latest_features = latest_features[list(model.feature_names_in_)]
        
        features_dict = latest_features.to_dict(orient="records")[0]
        
        # Predict
        raw_pred = model.predict(latest_features)[0]
        
        if hasattr(raw_pred, "__len__") and len(raw_pred) >= 3:
            pred_24h = max(0.0, float(raw_pred[0]))
            pred_48h = max(0.0, float(raw_pred[1]))
            pred_72h = max(0.0, float(raw_pred[2]))
        else:
            single_val = max(0.0, float(raw_pred))
            pred_24h = pred_48h = pred_72h = single_val
            
        # SHAP explanation
        shap_contrib = {}
        try:
            if explainer is not None:
                sv = explainer.shap_values(latest_features)
                if isinstance(sv, list):
                    sv = sv[0]
                if len(sv.shape) == 3:
                    sv = sv[0, :, 0]
                elif len(sv.shape) == 2:
                    sv = sv[0]
                for col, val in zip(latest_features.columns, sv):
                    shap_contrib[col] = float(val)
        except Exception as shap_err:
            print(f"Error generating SHAP explanation: {shap_err}")
            import traceback
            traceback.print_exc()
        
        return {
            "current_aqi": float(features_dict.get('aqi', 0.0)),
            "predicted_aqi_24h": pred_24h,
            "predicted_aqi_48h": pred_48h,
            "predicted_aqi_72h": pred_72h,
            "latest_timestamp": float(df.iloc[0]['timestamp']),
            "features_used": features_dict,
            "shap_explanation": shap_contrib
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "healthy"}
