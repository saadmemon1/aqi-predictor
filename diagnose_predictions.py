"""
Diagnostic script to trace exactly why the model predicts 0 (or negative).
Run with: .venv/bin/python diagnose_predictions.py
"""
import os
import joblib
import pandas as pd
import numpy as np
import hopsworks

print("=" * 70)
print("STEP 1: Connecting to Hopsworks")
print("=" * 70)
project = hopsworks.login(host="eu-west.cloud.hopsworks.ai", api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
mr = project.get_model_registry()

print("\n" + "=" * 70)
print("STEP 2: Download the model from registry")
print("=" * 70)
aqi_model = mr.get_best_model("aqi_prediction_model", "R2", "max")
print(f"Model version: {aqi_model.version}")
print(f"Model description: {aqi_model.description}")
print(f"Model metrics: {aqi_model.training_metrics}")

model_dir = aqi_model.download()
model = joblib.load(os.path.join(model_dir, "aqi_model.pkl"))

print(f"\nModel type: {type(model).__name__}")
if hasattr(model, "feature_names_in_"):
    print(f"Model expects {len(model.feature_names_in_)} features:")
    for i, name in enumerate(model.feature_names_in_):
        print(f"  [{i}] {name}")
else:
    print("WARNING: Model has no feature_names_in_ attribute!")

if hasattr(model, "n_features_in_"):
    print(f"n_features_in_: {model.n_features_in_}")

# Check if multi-output
print(f"\nIs MultiOutput model: {hasattr(model, 'estimators_') and isinstance(getattr(model, 'estimators_', None), list)}")

print("\n" + "=" * 70)
print("STEP 3: Fetch batch data from Feature View v2")
print("=" * 70)
feature_view = fs.get_feature_view("aqi_features_view", version=2)
batch_data = feature_view.get_batch_data()
print(f"Batch data shape: {batch_data.shape}")
print(f"Batch data columns ({len(batch_data.columns)}):")
for c in batch_data.columns:
    print(f"  - {c}")

batch_data.sort_values(by="timestamp", ascending=False, inplace=True)
latest_row = batch_data.head(1)
print(f"\nLatest row (raw) values:")
for c in latest_row.columns:
    val = latest_row[c].values[0]
    print(f"  {c} = {val} (type: {type(val).__name__})")

print("\n" + "=" * 70)
print("STEP 4: Prepare features (same as main.py)")
print("=" * 70)
latest_features = latest_row.drop(['date', 'timestamp', 'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h'], axis=1, errors='ignore')
print(f"After dropping date/timestamp/targets: {latest_features.shape}")
print(f"Columns: {list(latest_features.columns)}")
print(f"Values: {latest_features.to_dict(orient='records')[0]}")

# Check for NaN/inf
nan_cols = [c for c in latest_features.columns if latest_features[c].isna().any()]
inf_cols = [c for c in latest_features.columns if np.isinf(latest_features[c].values).any()]
print(f"\nColumns with NaN: {nan_cols if nan_cols else 'NONE'}")
print(f"Columns with Inf: {inf_cols if inf_cols else 'NONE'}")

if hasattr(model, "feature_names_in_"):
    missing_in_data = set(model.feature_names_in_) - set(latest_features.columns)
    extra_in_data = set(latest_features.columns) - set(model.feature_names_in_)
    print(f"\nFeatures expected by model but MISSING in data: {missing_in_data if missing_in_data else 'NONE'}")
    print(f"Features in data but NOT expected by model: {extra_in_data if extra_in_data else 'NONE'}")
    
    # Align
    latest_features = latest_features[list(model.feature_names_in_)]
    print(f"\nAfter alignment, shape: {latest_features.shape}")

print("\n" + "=" * 70)
print("STEP 5: Run prediction")
print("=" * 70)
raw_pred = model.predict(latest_features)
print(f"Raw prediction output: {raw_pred}")
print(f"Raw prediction shape: {raw_pred.shape}")
print(f"Raw prediction dtype: {raw_pred.dtype}")

pred_row = raw_pred[0]
print(f"\nFirst row prediction: {pred_row}")
if hasattr(pred_row, "__len__") and len(pred_row) >= 3:
    print(f"  pred_24h (raw): {pred_row[0]}")
    print(f"  pred_48h (raw): {pred_row[1]}")
    print(f"  pred_72h (raw): {pred_row[2]}")
    print(f"  pred_24h (clamped): {max(0.0, float(pred_row[0]))}")
    print(f"  pred_48h (clamped): {max(0.0, float(pred_row[1]))}")
    print(f"  pred_72h (clamped): {max(0.0, float(pred_row[2]))}")
else:
    print(f"  Single value prediction: {float(pred_row)}")
    print(f"  Clamped: {max(0.0, float(pred_row))}")

print("\n" + "=" * 70)
print("STEP 6: Cross-check with training data")
print("=" * 70)
print("Fetching a sample from training data to compare distributions...")
try:
    X_train, y_train = feature_view.training_data(description="Diagnostic check")
    X_train = X_train.drop(['date', 'timestamp', 'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h'], axis=1, errors='ignore')
    print(f"Training X shape: {X_train.shape}")
    print(f"Training y shape: {y_train.shape}")
    print(f"Training y columns: {list(y_train.columns)}")
    print(f"\nTraining y stats:")
    print(y_train.describe())
    print(f"\nTraining X stats (mean):")
    print(X_train.mean())
    print(f"\nInference features vs Training mean:")
    for c in latest_features.columns:
        inf_val = latest_features[c].values[0]
        train_mean = X_train[c].mean()
        train_std = X_train[c].std()
        z_score = (inf_val - train_mean) / train_std if train_std > 0 else 0
        flag = " ⚠️ OUTLIER" if abs(z_score) > 3 else ""
        print(f"  {c}: inference={inf_val:.4f}, train_mean={train_mean:.4f}, z-score={z_score:.2f}{flag}")
    
    # Test prediction on a known training sample
    print(f"\nPrediction on FIRST training sample:")
    sample = X_train.head(1)
    if hasattr(model, "feature_names_in_"):
        sample = sample[list(model.feature_names_in_)]
    sample_pred = model.predict(sample)
    print(f"  Input: {sample.to_dict(orient='records')[0]}")
    print(f"  Prediction: {sample_pred}")
    print(f"  Actual target: {y_train.head(1).to_dict(orient='records')[0]}")
    
    # Test on a random training sample
    print(f"\nPrediction on RANDOM training sample:")
    idx = len(X_train) // 2
    sample2 = X_train.iloc[[idx]]
    if hasattr(model, "feature_names_in_"):
        sample2 = sample2[list(model.feature_names_in_)]
    sample_pred2 = model.predict(sample2)
    print(f"  Prediction: {sample_pred2}")
    print(f"  Actual target: {y_train.iloc[[idx]].to_dict(orient='records')[0]}")
    
except Exception as e:
    print(f"Error fetching training data: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)
