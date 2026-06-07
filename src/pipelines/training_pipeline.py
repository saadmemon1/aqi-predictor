"""
Daily Training Pipeline
Fetches training data from the Hopsworks Feature Store,
trains XGBoost with hyperparameter tuning, evaluates performance,
and uploads the new model to the Hopsworks Model Registry.
"""
import os
import pandas as pd
import numpy as np
import hopsworks
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')
def fix_xgboost_base_score(model):
    """
    XGBoost 3.0+ serializes 'base_score' as a list-string like '[8.8E1,8.8E1,8.8E1]'.
    SHAP's XGBTreeModelLoader calls float() on this and crashes.
    
    Key insight: SHAP's _tree.py does 'from .other._ubjson import decode_ubjson_buffer',
    which creates a LOCAL reference in the _tree module namespace. We must patch THAT
    reference (shap_tree.decode_ubjson_buffer), not the source module.
    """
    import shap.explainers._tree as shap_tree

    def _fix_base_score_in_dict(d):
        """Fix base_score in a parsed XGBoost model dict."""
        if not isinstance(d, dict):
            return d
        try:
            bs = d.get("learner", {}).get("learner_model_param", {}).get("base_score")
            if bs is not None:
                if isinstance(bs, str) and bs.startswith("["):
                    first_val = bs.strip("[]").split(",")[0].strip()
                    d["learner"]["learner_model_param"]["base_score"] = first_val
                    print(f"  [patch] Fixed base_score from list-string to: {first_val}")
                elif isinstance(bs, list) and len(bs) > 0:
                    d["learner"]["learner_model_param"]["base_score"] = str(float(bs[0]))
                    print(f"  [patch] Fixed base_score from list to: {bs[0]}")
        except Exception as e:
            print(f"  [patch] Warning during base_score fix: {e}")
        return d

    # Patch the decode_ubjson_buffer reference IN shap.explainers._tree (not the source module)
    original_decode = shap_tree.decode_ubjson_buffer
    if not hasattr(original_decode, "_bs_patched"):
        def patched_decode(*args, **kwargs):
            result = original_decode(*args, **kwargs)
            return _fix_base_score_in_dict(result)
        patched_decode._bs_patched = True
        shap_tree.decode_ubjson_buffer = patched_decode
        print("Patched shap_tree.decode_ubjson_buffer (local ref in _tree.py).")

    print("Successfully monkey-patched SHAP to handle XGBoost 3.0+ base_score.")


def run_training_pipeline():
    print("=" * 60)
    print("STEP 1: Connecting to Hopsworks")
    print("=" * 60)
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        print("HOPSWORKS_API_KEY not found in environment.")
        return False

    project = hopsworks.login(host="eu-west.cloud.hopsworks.ai", api_key_value=api_key)
    fs = project.get_feature_store()

    print("=" * 60)
    print("STEP 2: Fetching training data from Feature Store")
    print("=" * 60)
    aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
    query = aqi_fg.select_all()
    feature_view = fs.get_or_create_feature_view(
        name="aqi_features_view",
        version=2,
        description="Read from AQI feature group",
        labels=["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"],
        query=query
    )
    X, y = feature_view.training_data(description="Training data for AQI prediction")

    # Clean up columns
    X = X.drop(['date', 'timestamp', 'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h'], axis=1, errors='ignore')
    X.sort_index(inplace=True)
    y.sort_index(inplace=True)

    # Drop rows where targets are NaN (these are the live-inserted rows with masked targets)
    valid_mask = y.notna().all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"Training data shape: X={X.shape}, y={y.shape}")

    # Chronological split (walk-forward validation, no data leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    print("=" * 60)
    print("STEP 3: Training models with hyperparameter tuning")
    print("=" * 60)

    models = {
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42)
    }

    param_grids = {
        "Ridge Regression": {
            'alpha': [1.0, 10.0]
        },
        "Random Forest": {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2]
        },
        "XGBoost": {
            'n_estimators': [100, 200],
            'max_depth': [5, 7],
            'learning_rate': [0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'min_child_weight': [3],
            'reg_alpha': [0.0],
            'reg_lambda': [1.0]
        }
    }

    results = {}
    best_overall_model = None
    best_overall_model_name = None
    best_overall_r2 = -float('inf')

    for name, base_model in models.items():
        print(f"\n--- Tuning and Training {name} ---")
        
        # Calculate total available parameter combinations to adjust n_iter safely
        n_configs = 1
        for vals in param_grids[name].values():
            n_configs *= len(vals)
        curr_n_iter = min(25, n_configs)
        print(f"Running search with {curr_n_iter} iterations (out of {n_configs} combinations)...")
        
        search = RandomizedSearchCV(
            base_model,
            param_grids[name],
            n_iter=curr_n_iter,
            cv=3,
            scoring='r2',
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f"Best params for {name}: {search.best_params_}")

        y_pred = best_model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'model': best_model}
        print(f"{name} Test Metrics -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

        if r2 > best_overall_r2:
            best_overall_r2 = r2
            best_overall_model = best_model
            best_overall_model_name = name

    print(f"\n🏆 Best Overall Model: {best_overall_model_name} with Test R2: {best_overall_r2:.2f}")

    # Patch XGBoost model configuration if it won, to prevent SHAP TreeExplainer crash
    if best_overall_model_name == "XGBoost":
        fix_xgboost_base_score(best_overall_model)

    print("=" * 60)
    print("STEP 4: Generating SHAP explanations")
    print("=" * 60)

    if best_overall_model_name in ["Random Forest", "XGBoost"]:
        explainer = shap.TreeExplainer(best_overall_model)
    else:
        explainer = shap.LinearExplainer(best_overall_model, X_train)

    print("SHAP explainer created successfully.")

    print("=" * 60)
    print("STEP 5: Uploading model to Hopsworks Model Registry")
    print("=" * 60)

    mr = project.get_model_registry()

    model_dir = "aqi_model_artifacts"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(best_overall_model, os.path.join(model_dir, "aqi_model.pkl"))
    joblib.dump(explainer, os.path.join(model_dir, "shap_explainer.pkl"))

    metrics = {
        "RMSE": results[best_overall_model_name]['RMSE'],
        "MAE": results[best_overall_model_name]['MAE'],
        "R2": results[best_overall_model_name]['R2']
    }

    aqi_model = mr.python.create_model(
        name="aqi_prediction_model",
        metrics=metrics,
        description=f"AQI prediction model using tuned {best_overall_model_name} (daily retrain)"
    )
    aqi_model.save(model_dir)

    print(f"Model uploaded successfully! Metrics: {metrics}")
    print("=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_training_pipeline()
    if not success:
        exit(1)
