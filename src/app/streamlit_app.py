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
    import time
    max_retries = 5
    delay = 2.0
    last_ex = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get("http://127.0.0.1:8000/predict", timeout=30)
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
    current_aqi = max(0.0, data.get('current_aqi', 0.0))
    pred_aqi_24h = max(0.0, data.get('predicted_aqi_24h', 0.0))
    pred_aqi_48h = max(0.0, data.get('predicted_aqi_48h', 0.0))
    pred_aqi_72h = max(0.0, data.get('predicted_aqi_72h', 0.0))
    features = data['features_used']
    last_updated = datetime.fromtimestamp(data['latest_timestamp'] / 1000.0).strftime('%Y-%m-%d %H:%M')
    
    # US EPA AQI Categories
    def get_aqi_status(aqi):
        if aqi > 300:
            return "#7e0023", "<div style='padding: 15px; border-radius: 5px; background: rgba(126, 0, 35, 0.2); border-left: 5px solid #7e0023; color: white; font-weight: bold;'>⚠️ HAZARDOUS</div>"
        elif aqi > 200:
            return "#8f3f97", "<div style='padding: 15px; border-radius: 5px; background: rgba(143, 63, 151, 0.2); border-left: 5px solid #8f3f97; color: white; font-weight: bold;'>⚠️ VERY UNHEALTHY</div>"
        elif aqi > 150:
            return "#ff0000", "<div style='padding: 15px; border-radius: 5px; background: rgba(255, 0, 0, 0.2); border-left: 5px solid #ff0000; color: white; font-weight: bold;'>⚠️ UNHEALTHY</div>"
        elif aqi > 100:
            return "#ff7e00", "<div style='padding: 15px; border-radius: 5px; background: rgba(255, 126, 0, 0.2); border-left: 5px solid #ff7e00; color: white; font-weight: bold;'>🚧 SENSITIVE GROUPS</div>"
        elif aqi > 50:
            return "#ffff00", "<div style='padding: 15px; border-radius: 5px; background: rgba(255, 255, 0, 0.2); border-left: 5px solid #ffff00; color: white; font-weight: bold;'>🟡 MODERATE</div>"
        else:
            return "#00e400", "<div style='padding: 15px; border-radius: 5px; background: rgba(0, 228, 0, 0.2); border-left: 5px solid #00e400; color: white; font-weight: bold;'>✅ GOOD</div>"

    color_curr, alert_curr = get_aqi_status(current_aqi)
    color_24h, alert_24h = get_aqi_status(pred_aqi_24h)
    color_48h, alert_48h = get_aqi_status(pred_aqi_48h)
    color_72h, alert_72h = get_aqi_status(pred_aqi_72h)

    st.markdown("<h3 style='text-align: center; color: white; margin-bottom: 5px;'>Live Dashboard & 3-Day Forecast</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #b0bec5; font-size: 0.9rem; margin-bottom: 20px;'>Data last updated: {last_updated} (UTC)</p>", unsafe_allow_html=True)
    
    # Show staleness warning if data is older than 24 hours
    data_age_hours = (datetime.utcnow() - datetime.utcfromtimestamp(data['latest_timestamp'] / 1000.0)).total_seconds() / 3600
    if data_age_hours > 24:
        st.warning(f"⚠️ Feature Store data is {data_age_hours:.0f} hours old. The Hopsworks offline materialization job may be queued. New data will appear once the job completes.")
    
    col0, col1, col2, col3 = st.columns(4)
    
    with col0:
        st.markdown(f"""
        <div class='metric-card' style='text-align: center; margin-bottom: 20px; border: 1px solid #00e400;'>
            <div style='font-size: 1.1rem; color: #b0bec5; font-weight: 600; margin-bottom: 10px;'>Current AQI</div>
            <div style='font-size: 3.5rem; font-weight: 800; color: {color_curr}; margin-bottom: 15px;'>{current_aqi:.0f}</div>
            {alert_curr}
        </div>
        """, unsafe_allow_html=True)

    with col1:
        st.markdown(f"""
        <div class='metric-card' style='text-align: center; margin-bottom: 20px;'>
            <div style='font-size: 1.1rem; color: #b0bec5; font-weight: 600; margin-bottom: 10px;'>Tomorrow (24h)</div>
            <div style='font-size: 3.5rem; font-weight: 800; color: {color_24h}; margin-bottom: 15px;'>{pred_aqi_24h:.0f}</div>
            {alert_24h}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card' style='text-align: center; margin-bottom: 20px;'>
            <div style='font-size: 1.1rem; color: #b0bec5; font-weight: 600; margin-bottom: 10px;'>Day After (48h)</div>
            <div style='font-size: 3.5rem; font-weight: 800; color: {color_48h}; margin-bottom: 15px;'>{pred_aqi_48h:.0f}</div>
            {alert_48h}
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-card' style='text-align: center; margin-bottom: 20px;'>
            <div style='font-size: 1.1rem; color: #b0bec5; font-weight: 600; margin-bottom: 10px;'>3 Days (72h)</div>
            <div style='font-size: 3.5rem; font-weight: 800; color: {color_72h}; margin-bottom: 15px;'>{pred_aqi_72h:.0f}</div>
            {alert_72h}
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
    

