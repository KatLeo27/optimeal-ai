"""
OptiMeal AI - Demand Prediction Engine
Handles model inference, input validation, point prediction, and uncertainty estimation.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple
from src.train_demand_model import MODEL_SAVE_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DemandPredictor:
    """
    Inference wrapper for meal demand prediction pipeline.
    """
    def __init__(self, model_path: str = MODEL_SAVE_PATH):
        self.model_path = model_path
        self.artifact = self._load_model()
        self.pipeline = self.artifact["pipeline"] if self.artifact else None
        self.metrics = self.artifact.get("metrics", {}) if self.artifact else {}
        self.model_name = self.artifact.get("best_model_name", "Demand Predictor") if self.artifact else ""

    def _load_model(self) -> dict:
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found at {self.model_path}. Training on the fly...")
            from src.train_demand_model import train_and_select_best_model
            return train_and_select_best_model()
        try:
            artifact = joblib.load(self.model_path)
            logger.info("Successfully loaded demand prediction model artifact.")
            return artifact
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}")
            from src.train_demand_model import train_and_select_best_model
            return train_and_select_best_model()

    def predict_demand(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make demand prediction and calculate estimated bounds.
        
        Input format:
        {
            "Meal": "Lunch",
            "Food_Category": "Vegetables",
            "Canteen_Section": "A",
            "Day_of_Week": "Tuesday",
            "expected_attendance": 600,
            "temperature": 29.5,
            "rainfall": 0.0,
            "holiday": 0,
            "special_event": 0,
            "previous_day_consumption": 570,
            "Is_Weekend": 0,
            "Month": 7,
            "Day": 15
        }
        """
        # Convert input dictionary to DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Ensure correct data types
        num_cols = [
            "expected_attendance", "temperature", "rainfall",
            "holiday", "special_event", "previous_day_consumption",
            "Is_Weekend", "Month", "Day"
        ]
        for col in num_cols:
            if col in df_input.columns:
                df_input[col] = pd.to_numeric(df_input[col], errors="coerce").fillna(0)
                
        # Run inference
        pred_value = float(self.pipeline.predict(df_input)[0])
        pred_value = max(0, round(pred_value))
        
        # Calculate uncertainty margin using model's test RMSE or percentage error
        test_rmse = 20.0
        if self.metrics and self.model_name in self.metrics:
            test_rmse = self.metrics[self.model_name].get("test_rmse", 20.0)
            
        # 90% confidence bound approximate (+/- 1.645 * RMSE)
        lower_bound = max(0, round(pred_value - 1.645 * test_rmse))
        upper_bound = round(pred_value + 1.645 * test_rmse)
        
        return {
            "predicted_consumption": pred_value,
            "lower_bound_90pct": lower_bound,
            "upper_bound_90pct": upper_bound,
            "confidence_margin": round(1.645 * test_rmse, 1),
            "model_used": self.model_name
        }

if __name__ == "__main__":
    predictor = DemandPredictor()
    sample_input = {
        "Meal": "Lunch",
        "Food_Category": "Vegetables",
        "Canteen_Section": "A",
        "Day_of_Week": "Tuesday",
        "expected_attendance": 650,
        "temperature": 30.0,
        "rainfall": 0.0,
        "holiday": 0,
        "special_event": 0,
        "previous_day_consumption": 610,
        "Is_Weekend": 0,
        "Month": 7,
        "Day": 15
    }
    result = predictor.predict_demand(sample_input)
    print("Sample Inference Result:")
    print(result)
