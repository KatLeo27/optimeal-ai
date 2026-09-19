"""
OptiMeal AI - Demand Prediction Model Training
Trains, evaluates, benchmarks, and persists meal demand forecasting models.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import load_processed_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join("models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "demand_model.joblib")
METRICS_SAVE_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

# Operational features known prior to meal preparation
CATEGORICAL_FEATURES = ["Meal", "Food_Category", "Canteen_Section", "Day_of_Week"]
NUMERICAL_FEATURES = [
    "expected_attendance",
    "temperature",
    "rainfall",
    "holiday",
    "special_event",
    "previous_day_consumption",
    "Is_Weekend",
    "Month",
    "Day"
]
TARGET_COLUMN = "meals_consumed"

def build_preprocessing_pipeline():
    """
    Constructs ColumnTransformer for encoding categorical and scaling numerical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor

def evaluate_model(name: str, model_pipeline, X_train, y_train, X_test, y_test) -> dict:
    """
    Train pipeline and calculate MAE, RMSE, and R2 evaluation metrics.
    """
    model_pipeline.fit(X_train, y_train)
    y_pred_train = model_pipeline.predict(X_train)
    y_pred_test = model_pipeline.predict(X_test)
    
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    
    metrics = {
        "model_name": name,
        "test_mae": round(float(mae_test), 2),
        "test_rmse": round(float(rmse_test), 2),
        "test_r2": round(float(r2_test), 4),
        "train_mae": round(float(mae_train), 2),
        "train_rmse": round(float(rmse_train), 2),
        "train_r2": round(float(r2_train), 4)
    }
    logger.info(f"[{name}] Test MAE: {mae_test:.2f} | RMSE: {rmse_test:.2f} | R²: {r2_test:.4f}")
    return metrics

def train_and_select_best_model():
    """
    Loads data, trains candidate models, benchmarks metrics, and saves the best model.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_processed_data()
    
    # Feature matrix & target vector
    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = df[TARGET_COLUMN]
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    logger.info(f"Dataset split: {len(X_train)} training rows, {len(X_test)} testing rows.")
    
    # Candidate architectures
    candidates = {
        "Linear Regression (Baseline)": Pipeline([
            ("preprocessor", build_preprocessing_pipeline()),
            ("regressor", LinearRegression())
        ]),
        "Random Forest Regressor": Pipeline([
            ("preprocessor", build_preprocessing_pipeline()),
            ("regressor", RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_split=4, random_state=42))
        ]),
        "Gradient Boosting Regressor": Pipeline([
            ("preprocessor", build_preprocessing_pipeline()),
            ("regressor", GradientBoostingRegressor(n_estimators=160, learning_rate=0.08, max_depth=5, random_state=42))
        ])
    }
    
    all_metrics = {}
    best_model_name = None
    best_score = -float("inf")
    best_pipeline = None
    
    for name, pipeline in candidates.items():
        metrics = evaluate_model(name, pipeline, X_train, y_train, X_test, y_test)
        all_metrics[name] = metrics
        
        # Select best model using R2 score (and lowest RMSE)
        if metrics["test_r2"] > best_score:
            best_score = metrics["test_r2"]
            best_model_name = name
            best_pipeline = pipeline
            
    logger.info(f"Best selected model: {best_model_name} with R² = {best_score:.4f}")
    
    # Extract feature importances if available
    feature_importance_dict = {}
    if hasattr(best_pipeline.named_steps["regressor"], "feature_importances_"):
        ohe = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        all_feature_names = cat_feature_names + NUMERICAL_FEATURES
        importances = best_pipeline.named_steps["regressor"].feature_importances_
        for feat, imp in zip(all_feature_names, importances):
            feature_importance_dict[feat] = round(float(imp), 4)
            
    # Package artifact bundle
    model_artifact = {
        "pipeline": best_pipeline,
        "best_model_name": best_model_name,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "target": TARGET_COLUMN,
        "metrics": all_metrics,
        "feature_importances": feature_importance_dict,
        "trained_at": datetime.now().isoformat()
    }
    
    joblib.dump(model_artifact, MODEL_SAVE_PATH)
    logger.info(f"Saved best model artifact bundle to {MODEL_SAVE_PATH}")
    
    # Save standalone JSON metrics for documentation & inspection
    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(all_metrics, f, indent=4)
        
    return model_artifact

if __name__ == "__main__":
    train_and_select_best_model()
