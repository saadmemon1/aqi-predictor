# Serverless AQI Predictor

A 100% serverless machine learning system that predicts Air Quality Index (AQI) 72 hours ahead for Karachi.

## Features
- **Data Source**: OpenMeteo API (Free, no API key required for weather & AQI data)
- **Feature Store & Registry**: Hopsworks Serverless Free Tier
- **Automation**: `cron-job.org` triggers hourly feature updates
- **Hosting**: Hugging Face Docker Spaces (FastAPI + Streamlit in one container)
- **Models**: Random Forest, Ridge Regression, XGBoost
- **Interpretability**: SHAP

## Architecture

1. **Backfill & Train (Kaggle)**
   - Notebook `01_Data_Backfill_and_EDA.ipynb` fetches historical data and writes it to Hopsworks.
   - Notebook `02_Model_Training_Pipeline.ipynb` trains the models and uploads the best model to Hopsworks.

2. **Hourly Feature Updates**
   - `cron-job.org` sends a POST request to `/run-feature-pipeline` on the FastAPI server every hour.
   - FastAPI executes `src/pipelines/feature_pipeline.py` which pushes the latest 24 hours of data to Hopsworks.

3. **Inference & UI**
   - Streamlit connects to FastAPI (`/predict`).
   - FastAPI pulls the latest feature vector from Hopsworks, predicts the AQI, and returns the result to Streamlit.

## Quickstart

### 1. Requirements
Ensure you have Python 3.12+ and install requirements:
```bash
pip install -r requirements.txt
```

### 2. Hopsworks Setup
Create an account on [Hopsworks](https://app.hopsworks.ai) and retrieve your API key.
Set it as an environment variable:
```bash
export HOPSWORKS_API_KEY="your-key-here"
```

### 3. Running Locally
Run the Docker-like start script which spins up FastAPI on `8000` and Streamlit on `7860`.
```bash
chmod +x start.sh
./start.sh
```

### 4. Deploying to Hugging Face
1. Create a new **Docker** space on Hugging Face.
2. Add `HOPSWORKS_API_KEY` to the Space Secrets.
3. Push this repository to the Space.
4. Set up `cron-job.org` to ping your Space URL `/run-feature-pipeline` every hour.
