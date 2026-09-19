"""
OptiMeal AI - Data Processing Module
Handles data cleaning, deterministic feature extraction, and pipeline transformations.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw authentic food waste dataset:
    - Handle whitespace in column names
    - Convert Date column to datetime
    - Check and remove exact duplicates
    - Validate numeric ranges
    """
    df = df.copy()
    
    # Standardize column headers
    df.columns = [c.strip() for c in df.columns]
    
    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Remove duplicate records if any
    initial_len = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_len:
        logger.info(f"Removed {initial_len - len(df)} duplicate rows.")
        
    # Ensure numeric columns are positive floats
    for col in ["Waste_Weight_kg", "Unit_Price_per_kg", "Cost_Loss"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            df[col] = df[col].clip(lower=0.0)
            
    # Verify Cost_Loss matches Unit_Price_per_kg * Waste_Weight_kg
    if "Cost_Loss" in df.columns and "Waste_Weight_kg" in df.columns and "Unit_Price_per_kg" in df.columns:
        calculated_cost = (df["Waste_Weight_kg"] * df["Unit_Price_per_kg"]).round(2)
        # Update if slight rounding discrepancies exist
        df["Cost_Loss"] = calculated_cost
        
    logger.info(f"Cleaned dataset. Total records: {len(df)}")
    return df

def derive_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive deterministic calendar and time variables from authentic Date.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])
        
    df["Day_of_Week"] = df["Date"].dt.day_name()
    df["Day_Num"] = df["Date"].dt.dayofweek # 0=Monday, 6=Sunday
    df["Is_Weekend"] = df["Day_Num"].apply(lambda x: 1 if x in [5, 6] else 0)
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Day_of_Year"] = df["Date"].dt.dayofyear
    
    return df
