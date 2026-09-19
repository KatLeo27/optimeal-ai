"""
OptiMeal AI - Data Loader Module
Loads and validates authentic source datasets.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = os.path.join("data", "raw", "Dataset Propely.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "food_waste_processed.csv")

def load_raw_data(filepath=RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the authentic raw food waste dataset.
    
    Parameters:
    - filepath: Path to the raw CSV file.
    
    Returns:
    - pd.DataFrame: Raw dataset.
    """
    if not os.path.exists(filepath):
        logger.error(f"Raw data file not found at: {filepath}")
        raise FileNotFoundError(f"Source dataset not found at {filepath}. Please ensure 'Dataset Propely.csv' is placed in 'data/raw/'.")
    
    df = pd.read_csv(filepath)
    logger.info(f"Loaded raw dataset successfully with shape {df.shape} from {filepath}")
    return df

def load_processed_data(filepath=PROCESSED_DATA_PATH) -> pd.DataFrame:
    """
    Load processed dataset. If not found, attempts to process from raw.
    
    Parameters:
    - filepath: Path to processed CSV file.
    
    Returns:
    - pd.DataFrame: Processed and feature-engineered dataset.
    """
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        logger.info(f"Loaded processed dataset with shape {df.shape} from {filepath}")
        return df
    else:
        logger.warning(f"Processed dataset not found at {filepath}. Triggering processing pipeline...")
        from src.feature_engineering import generate_and_save_processed_data
        return generate_and_save_processed_data()

def inspect_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Generate diagnostic summary of dataset health and statistics.
    """
    summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "date_range": (str(df["Date"].min()), str(df["Date"].max())) if "Date" in df.columns else None,
        "meals": df["Meal"].unique().tolist() if "Meal" in df.columns else [],
        "categories": df["Food_Category"].unique().tolist() if "Food_Category" in df.columns else []
    }
    return summary

if __name__ == "__main__":
    raw_df = load_raw_data()
    print("Raw Data Overview:")
    print(raw_df.info())
    print("\nFirst 3 rows:")
    print(raw_df.head(3))
