import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# Configure page
st.set_page_config(page_title="Karachi AQI Predictor", page_icon="🌬️", layout="wide")

# Custom CSS for modern aesthetics
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }
    
    /* Cards */
    div[data-testid="stMetricValue"] {
        font-size: 3rem !important;
        font-weight: 800;
        color: #00E676;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        background: -webkit-linear-gradient(#eee, #333);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        color: white !important;
    }
    
    /* Text overrides for gradient background */
    p, span, div {
        color: #e0e0e0;
    }
    
    .alert-danger {
        background: rgba(255, 50, 50, 0.2);
        border-left: 5px solid #ff3232;
        padding: 15px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
    
    .alert-warning {
        background: rgba(255, 165, 0, 0.2);
        border-left: 5px solid #ffa500;
        padding: 15px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
    
    .alert-success {
        background: rgba(0, 230, 118, 0.2);
        border-left: 5px solid #00E676;
        padding: 15px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; font-size: 4em; margin-bottom: 0px;'>Karachi AQI Predictor 🌬️</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2em; color: #b0bec5; margin-bottom: 40px;'>Predicting Air Quality 72 Hours Ahead using Serverless Machine Learning</p>", unsafe_allow_html=True)

# Fetch prediction from FastAPI backend
@st.cache_data(ttl=300) # Cache for 5 minutes when successful
def fetch_prediction():
    response = requests.get("http://127.0.0.1:8000/predict")
    if response.status_code == 200:
        return response.json()
    else:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise Exception(f"Backend Error ({response.status_code}): {detail}")

with st.spinner("Fetching latest real-time predictions..."):
    try:
        data = fetch_prediction()
        error_message = None
    except Exception as e:
        data = None
        error_message = str(e)

if data:
    pred_aqi = max(0.0, data['predicted_aqi_72h'])
    features = data['features_used']
    last_updated = datetime.fromtimestamp(data['latest_timestamp'] / 1000.0).strftime('%Y-%m-%d %H:%M')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Determine color and alert HTML based on severity
        if pred_aqi > 150:
            aqi_color = "#ff3232" # Red
            alert_html = "<div class='alert-danger'>⚠️ HAZARDOUS ALERT: AQI is predicted to be very poor!</div>"
        elif pred_aqi > 100:
            aqi_color = "#ffa500" # Orange
            alert_html = "<div class='alert-warning'>🚧 WARNING: AQI is predicted to be poor. Sensitive groups should be careful.</div>"
        else:
            aqi_color = "#00E676" # Green
            alert_html = "<div class='alert-success'>✅ GOOD: Air quality is predicted to be acceptable.</div>"

        st.markdown(f"""
        <div class='metric-card' style='text-align: center; margin-bottom: 20px;'>
            <div style='font-size: 1.1rem; color: #b0bec5; font-weight: 600; margin-bottom: 10px;'>Predicted AQI (in 72 hours)</div>
            <div style='font-size: 3.5rem; font-weight: 800; color: {aqi_color}; margin-bottom: 15px;'>{pred_aqi:.0f}</div>
            {alert_html}
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    st.markdown("### 🔍 Current Environmental Factors")
    
    # Display features used for prediction
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("Temperature", f"{features.get('temperature', 0):.1f} °C")
    with mcol2:
        st.metric("Humidity", f"{features.get('humidity', 0):.0f} %")
    with mcol3:
        st.metric("PM2.5", f"{features.get('pm2_5', 0):.1f} μg/m³")
    with mcol4:
        st.metric("PM10", f"{features.get('pm10', 0):.1f} μg/m³")
        
    st.markdown("<br><p style='text-align: center; color: #90a4ae;'>Data sourced from OpenMeteo & Hopsworks Feature Store.</p>", unsafe_allow_html=True)

else:
    st.error(f"Failed to fetch predictions. Details: {error_message}")
    st.info("💡 Please verify that: \n1. Your `HOPSWORKS_API_KEY` is added to the Space secrets (Settings -> Variables and Secrets).\n2. The Space is fully built and started.")
    
    st.markdown("### 🛠️ Network Connectivity Diagnostics")
    if st.button("Run Connection Diagnostics"):
        with st.spinner("Testing DNS resolution and outbound HTTP access..."):
            try:
                diag_response = requests.get("http://127.0.0.1:8000/test-dns")
                if diag_response.status_code == 200:
                    st.json(diag_response.json())
                else:
                    st.error(f"Failed to get diagnostics from backend (HTTP {diag_response.status_code}): {diag_response.text}")
            except Exception as e:
                st.error(f"Could not connect to backend diagnostics endpoint: {str(e)}")
