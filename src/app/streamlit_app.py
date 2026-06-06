import streamlit as st
import requests
import pandas as pd
from datetime import datetime

def calculate_pm25_aqi(c):
    if c <= 12.0:
        return ((50 - 0) / (12.0 - 0)) * (c - 0) + 0
    elif c <= 35.4:
        return ((100 - 51) / (35.4 - 12.1)) * (c - 12.1) + 51
    elif c <= 55.4:
        return ((150 - 101) / (55.4 - 35.5)) * (c - 35.5) + 101
    elif c <= 150.4:
        return ((200 - 151) / (150.4 - 55.5)) * (c - 55.5) + 151
    elif c <= 250.4:
        return ((300 - 201) / (250.4 - 150.5)) * (c - 150.5) + 201
    elif c <= 350.4:
        return ((400 - 301) / (350.4 - 250.5)) * (c - 250.5) + 301
    else:
        return ((500 - 401) / (500.4 - 350.5)) * (c - 350.5) + 401

def calculate_pm10_aqi(c):
    if c <= 54.0:
        return ((50 - 0) / (54.0 - 0)) * (c - 0) + 0
    elif c <= 154.0:
        return ((100 - 51) / (154.0 - 55.0)) * (c - 55.0) + 51
    elif c <= 254.0:
        return ((150 - 101) / (254.0 - 155.0)) * (c - 155.0) + 101
    elif c <= 354.0:
        return ((200 - 151) / (354.0 - 255.0)) * (c - 255.0) + 151
    elif c <= 424.0:
        return ((300 - 201) / (424.0 - 355.0)) * (c - 355.0) + 201
    elif c <= 504.0:
        return ((400 - 301) / (504.0 - 425.0)) * (c - 425.0) + 301
    else:
        return ((500 - 401) / (604.0 - 505.0)) * (c - 505.0) + 401

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
    import time
    max_retries = 5
    delay = 2.0
    last_ex = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get("http://127.0.0.1:8000/predict", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                last_ex = Exception(f"Backend Error ({response.status_code}): {detail}")
                # If it's a 503 error, the backend is still loading, we should wait and retry
                if response.status_code == 503 and attempt < max_retries:
                    time.sleep(delay)
                    continue
                raise last_ex
        except Exception as e:
            last_ex = e
            if attempt < max_retries:
                time.sleep(delay)
            else:
                raise last_ex

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
    
    # Advanced Model Explanation (SHAP)
    with st.expander("🛠️ Advanced Model Explanation (SHAP)"):
        st.markdown("""
        This section explains **why** the machine learning model arrived at this prediction using **SHAP (SHapley Additive exPlanations)**.
        - **Positive values** push the predicted AQI **up** (predicting worse air quality).
        - **Negative values** pull the predicted AQI **down** (predicting cleaner air quality).
        """)
        
        shap_data = data.get("shap_explanation", {})
        if shap_data:
            # Create a dataframe for the SHAP values
            shap_df = pd.DataFrame({
                "Factor": list(shap_data.keys()),
                "Influence on AQI": list(shap_data.values())
            })
            
            # Sort by absolute influence to show the most important drivers
            shap_df["Magnitude"] = shap_df["Influence on AQI"].abs()
            shap_df = shap_df.sort_values(by="Magnitude", ascending=False).drop("Magnitude", axis=1)
            
            # Clean up feature names
            shap_df["Factor"] = shap_df["Factor"].str.replace("_", " ").str.title()
            
            # Render horizontal bar chart
            st.bar_chart(
                shap_df,
                x="Factor",
                y="Influence on AQI",
                use_container_width=True
            )
            
            # Show DataFrame
            st.dataframe(
                shap_df.rename(columns={"Influence on AQI": "Influence on AQI (SHAP value)"}),
                hide_index=True,
                use_container_width=True
            )
            
            # Sub-index EPA AQI calculations
            st.markdown("---")
            st.markdown("### 📊 Current AQI Calculation Math (US EPA)")
            st.markdown("The current AQI is determined by calculating the individual sub-indices for PM2.5 and PM10. The overall current AQI is the **maximum** of these sub-indices:")
            
            pm25_val = features.get('pm2_5', 0.0)
            pm10_val = features.get('pm10', 0.0)
            
            pm25_aqi = calculate_pm25_aqi(pm25_val)
            pm10_aqi = calculate_pm10_aqi(pm10_val)
            overall_current_aqi = max(pm25_aqi, pm10_aqi)
            
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.metric(
                    label="PM2.5 Sub-Index", 
                    value=f"{pm25_aqi:.0f}", 
                    help=f"Calculated from current PM2.5 concentration of {pm25_val:.1f} μg/m³"
                )
            with cc2:
                st.metric(
                    label="PM10 Sub-Index", 
                    value=f"{pm10_aqi:.0f}", 
                    help=f"Calculated from current PM10 concentration of {pm10_val:.1f} μg/m³"
                )
            with cc3:
                st.metric(
                    label="Current Overall AQI", 
                    value=f"{overall_current_aqi:.0f}", 
                    help="Maximum of the individual pollutant sub-indices"
                )
                
            st.latex(r"\text{Overall AQI} = \max(\text{AQI}_{\text{PM2.5}}, \text{AQI}_{\text{PM10}}) = \max(" + f"{pm25_aqi:.0f}, {pm10_aqi:.0f}" + r") = " + f"{overall_current_aqi:.0f}")
        else:
            st.info("SHAP explanations are loading or not available for this registered model version.")
    
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
