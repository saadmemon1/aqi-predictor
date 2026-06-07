import os
import pandas as pd
import hopsworks
import openmeteo_requests
from retry_requests import retry
from datetime import datetime, timedelta

def run_feature_pipeline():
    print("Running feature pipeline...")
    # Setup Hopsworks
    # Get API key from environment
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        print("HOPSWORKS_API_KEY not found in environment.")
        return False
        
    project = hopsworks.login(host="eu-west.cloud.hopsworks.ai", api_key_value=api_key)
    fs = project.get_feature_store()

    # Define coordinates
    lat, lon = 24.832862, 67.033866
    
    # We fetch the last 3 days to correctly compute 24h rolling features
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    
    import requests
    session = requests.Session()
    retry_session = retry(session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # 1. Weather Data
    weather_url = 'https://api.open-meteo.com/v1/forecast' # Using forecast API for current/recent
    weather_params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'surface_pressure', 'wind_speed_10m', 'wind_direction_10m']
    }
    weather_response = openmeteo.weather_api(weather_url, params=weather_params)[0]
    hourly_weather = weather_response.Hourly()

    weather_data = {
        'date': pd.date_range(
            start=pd.to_datetime(hourly_weather.Time(), unit='s', utc=True),
            end=pd.to_datetime(hourly_weather.TimeEnd(), unit='s', utc=True),
            freq=pd.Timedelta(seconds=hourly_weather.Interval()),
            inclusive='left'
        ),
        'temperature': hourly_weather.Variables(0).ValuesAsNumpy(),
        'humidity': hourly_weather.Variables(1).ValuesAsNumpy(),
        'precipitation': hourly_weather.Variables(2).ValuesAsNumpy(),
        'pressure': hourly_weather.Variables(3).ValuesAsNumpy(),
        'wind_speed': hourly_weather.Variables(4).ValuesAsNumpy(),
        'wind_direction': hourly_weather.Variables(5).ValuesAsNumpy()
    }
    df_weather = pd.DataFrame(data=weather_data)

    # 2. Air Quality Data
    aqi_url = 'https://air-quality-api.open-meteo.com/v1/air-quality'
    aqi_params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ['pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'us_aqi']
    }
    aqi_response = openmeteo.weather_api(aqi_url, params=aqi_params)[0]
    hourly_aqi = aqi_response.Hourly()

    aqi_data = {
        'date': pd.date_range(
            start=pd.to_datetime(hourly_aqi.Time(), unit='s', utc=True),
            end=pd.to_datetime(hourly_aqi.TimeEnd(), unit='s', utc=True),
            freq=pd.Timedelta(seconds=hourly_aqi.Interval()),
            inclusive='left'
        ),
        'pm10': hourly_aqi.Variables(0).ValuesAsNumpy(),
        'pm2_5': hourly_aqi.Variables(1).ValuesAsNumpy(),
        'carbon_monoxide': hourly_aqi.Variables(2).ValuesAsNumpy(),
        'nitrogen_dioxide': hourly_aqi.Variables(3).ValuesAsNumpy(),
        'sulphur_dioxide': hourly_aqi.Variables(4).ValuesAsNumpy(),
        'ozone': hourly_aqi.Variables(5).ValuesAsNumpy(),
        'aqi': hourly_aqi.Variables(6).ValuesAsNumpy()
    }
    df_aqi = pd.DataFrame(data=aqi_data)

    # Merge
    df = pd.merge(df_weather, df_aqi, on='date', how='inner')
    df.dropna(inplace=True)
    
    # Feature Engineering
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    df['aqi_24h_rolling_mean'] = df['aqi'].rolling(window=24).mean()
    df['aqi_change_rate'] = df['aqi'].pct_change(periods=24)
    
    # In live serving, we don't know the future targets (that's what we predict).
    # But for historical feature insertion, we want to maintain the same schema.
    # We insert NaN for targets in the online feature store.
    df['target_aqi_24h'] = float('nan')
    df['target_aqi_48h'] = float('nan')
    df['target_aqi_72h'] = float('nan')
    
    df.dropna(subset=['aqi_change_rate'], inplace=True)
    
    df['timestamp'] = df['date'].astype('int64') // 10**6
    
    # We only insert the last 24 hours of data to avoid re-inserting everything
    df_recent = df.tail(24)

    # Upload to Hopsworks
    aqi_fg = fs.get_feature_group(name='aqi_features', version=1)
    aqi_fg.insert(df_recent)
    
    print(f"Successfully inserted {len(df_recent)} rows to Hopsworks.")
    return True

if __name__ == "__main__":
    run_feature_pipeline()
