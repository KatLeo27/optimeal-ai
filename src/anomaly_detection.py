"""
OptiMeal AI - Food Waste Anomaly Detection Engine
Implements statistical rolling Z-score and unsupervised Isolation Forest anomaly detection
with explainable root-cause diagnosis.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from sklearn.ensemble import IsolationForest
from src.data_loader import load_processed_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def detect_statistical_anomalies(df: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    """
    Detect waste anomalies using a rolling 7-day Z-score on daily aggregated waste.
    """
    daily = df.groupby(["Date", "Day_of_Week"]).agg(
        daily_waste_kg=("Waste_Weight_kg", "sum"),
        daily_cost_loss=("Cost_Loss", "sum"),
        meals_prepared=("meals_prepared", "sum"),
        meals_consumed=("meals_consumed", "sum"),
        surplus_meals=("surplus_meals", "sum"),
        is_holiday=("holiday", "max"),
        is_event=("special_event", "max"),
        rainfall=("rainfall", "max")
    ).reset_index()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.sort_values(by="Date").reset_index(drop=True)
    
    # 7-day rolling statistics
    daily["rolling_mean"] = daily["daily_waste_kg"].rolling(window=7, min_periods=3).mean()
    daily["rolling_std"] = daily["daily_waste_kg"].rolling(window=7, min_periods=3).std().fillna(1.0)
    
    # Z-score computation
    daily["z_score"] = ((daily["daily_waste_kg"] - daily["rolling_mean"]) / daily["rolling_std"]).fillna(0.0)
    
    # Flag positive anomalies (unusually high waste)
    daily["is_statistical_anomaly"] = daily["z_score"] >= z_threshold
    
    # Diagnostic explanations
    reasons = []
    for _, row in daily.iterrows():
        if row["is_statistical_anomaly"]:
            excess_pct = round(((row["daily_waste_kg"] - row["rolling_mean"]) / max(1.0, row["rolling_mean"])) * 100, 1)
            reason = f"Waste was +{excess_pct}% ({row['z_score']:.1f}σ) above the 7-day baseline."
            if row["surplus_meals"] > (row["meals_prepared"] * 0.15):
                reason += f" Severe overproduction ({row['surplus_meals']} surplus meals)."
            if row["is_holiday"]:
                reason += " Holiday attendance drop."
            if row["rainfall"] > 15:
                reason += f" Heavy rainfall ({row['rainfall']} mm)."
            reasons.append(reason)
        else:
            reasons.append("Normal operational range.")
            
    daily["anomaly_explanation"] = reasons
    return daily

def detect_isolation_forest_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Detect multi-dimensional anomalies across waste, cost, surplus, and attendance
    using unsupervised Isolation Forest.
    """
    df_feat = df.copy()
    feature_cols = [
        "Waste_Weight_kg", "Cost_Loss", "meals_prepared",
        "meals_consumed", "surplus_meals", "expected_attendance",
        "temperature", "rainfall"
    ]
    
    # Fill any missing values with column medians
    X = df_feat[feature_cols].fillna(df_feat[feature_cols].median())
    
    iso_forest = IsolationForest(
        n_estimators=120,
        contamination=contamination,
        random_state=42
    )
    
    preds = iso_forest.fit_predict(X)
    # Isolation forest outputs -1 for outliers and 1 for inliers
    df_feat["is_iso_anomaly"] = (preds == -1)
    df_feat["iso_anomaly_score"] = iso_forest.decision_function(X).round(3)
    
    logger.info(f"Isolation Forest flagged {df_feat['is_iso_anomaly'].sum()} anomalous records out of {len(df_feat)}.")
    return df_feat

def summarize_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive anomaly summary report.
    """
    stat_df = detect_statistical_anomalies(df)
    anomalous_days = stat_df[stat_df["is_statistical_anomaly"]]
    
    return {
        "total_anomalous_days": len(anomalous_days),
        "total_days_evaluated": len(stat_df),
        "anomaly_rate_pct": round((len(anomalous_days) / max(1, len(stat_df))) * 100, 1),
        "highest_z_score_day": anomalous_days.sort_values(by="z_score", ascending=False).iloc[0].to_dict() if len(anomalous_days) > 0 else None,
        "anomalous_days_dataframe": anomalous_days
    }

if __name__ == "__main__":
    raw_proc = load_processed_data()
    summary = summarize_anomalies(raw_proc)
    print(f"Anomalous Days Found: {summary['total_anomalous_days']} / {summary['total_days_evaluated']}")
    print("\nSample Anomalous Days:")
    print(summary["anomalous_days_dataframe"][["Date", "daily_waste_kg", "z_score", "anomaly_explanation"]].head(5))
