"""
OptiMeal AI - Feature Engineering & Data Augmentation Module
Generates statistically grounded complementary operational variables to support demand modeling,
preserving full provenance between authentic, derived, and synthetic features.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import logging
from src.data_loader import load_raw_data, RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.data_processing import clean_raw_data, derive_temporal_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def augment_operational_features(df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """
    Augment authentic food waste records with realistic operational variables:
    expected attendance, actual attendance, meals prepared, meals consumed,
    weather metrics, event indicators, and lagged consumption.
    
    All variables are generated with fixed seed for exact reproducibility.
    """
    np.random.seed(random_seed)
    df = df.copy()
    
    # Base attendance capacity by meal type
    meal_base_attendance = {
        "Breakfast": 260,
        "Lunch": 620,
        "Dinner": 410
    }
    
    # Section multipliers (assuming Section A & B are main dining halls, C & D are executive/annex)
    section_multipliers = {
        "A": 1.15,
        "B": 1.05,
        "C": 0.85,
        "D": 0.95
    }
    
    # Category popularity adjustment for consumption
    category_consumption_rate = {
        "Rice": 0.96,
        "Vegetables": 0.93,
        "Soup": 0.91,
        "Meat": 0.97
    }
    
    # Track operational dates
    unique_dates = df["Date"].sort_values().unique()
    
    # Pre-generate day-level environmental conditions for consistency across same day
    day_conditions = {}
    for date_val in unique_dates:
        dt = pd.to_datetime(date_val)
        day_of_year = dt.dayofyear
        
        # Summer temperature in Celsius (June-August)
        base_temp = 28.0 + 3.5 * np.sin(2 * np.pi * (day_of_year - 170) / 365)
        temp = round(float(np.random.normal(base_temp, 2.0)), 1)
        
        # Summer thunderstorm/rain probability (~20% of days)
        is_raining = np.random.rand() < 0.22
        rain = round(float(np.random.exponential(14.0) if is_raining else 0.0), 1)
        
        # Public holidays & institutional special events (conferences/festivals)
        is_holiday = int(np.random.rand() < 0.05)
        is_special_event = int(np.random.rand() < 0.08 and not is_holiday)
        
        day_conditions[date_val] = {
            "temperature": temp,
            "rainfall": rain,
            "holiday": is_holiday,
            "special_event": is_special_event
        }
        
    expected_attendances = []
    actual_attendances = []
    meals_prepared_list = []
    meals_consumed_list = []
    temps = []
    rains = []
    holidays = []
    special_events = []
    
    for idx, row in df.iterrows():
        meal = row["Meal"]
        section = row.get("Canteen_Section", "A")
        cat = row.get("Food_Category", "Rice")
        is_weekend = row.get("Is_Weekend", 0)
        date_val = row["Date"]
        
        cond = day_conditions[date_val]
        temp = cond["temperature"]
        rain = cond["rainfall"]
        holiday = cond["holiday"]
        special_event = cond["special_event"]
        
        temps.append(temp)
        rains.append(rain)
        holidays.append(holiday)
        special_events.append(special_event)
        
        # Calculate expected attendance
        base = meal_base_attendance.get(meal, 400) * section_multipliers.get(section, 1.0)
        
        if is_weekend:
            base *= 0.55
        if holiday:
            base *= 0.35
        if special_event:
            base *= 1.35
        if rain > 15:
            base *= 0.92
            
        expected_att = int(np.clip(np.random.normal(base, base * 0.06), 40, 1200))
        expected_attendances.append(expected_att)
        
        # Actual attendance
        noise = np.random.normal(0, expected_att * 0.05)
        actual_att = int(np.clip(expected_att + noise, 30, expected_att * 1.35))
        actual_attendances.append(actual_att)
        
        # Traditional preparation strategy (Historical fixed buffer 7-14%)
        prep_buffer = np.random.uniform(0.07, 0.14)
        prepared = int(np.ceil(expected_att * (1 + prep_buffer)))
        meals_prepared_list.append(prepared)
        
        # Meals consumed based on attendance and category demand
        rate = category_consumption_rate.get(cat, 0.94) + np.random.uniform(-0.02, 0.02)
        consumed = min(int(round(actual_att * rate)), prepared)
        meals_consumed_list.append(consumed)
        
    df["temperature"] = temps
    df["rainfall"] = rains
    df["holiday"] = holidays
    df["special_event"] = special_events
    df["expected_attendance"] = expected_attendances
    df["actual_attendance"] = actual_attendances
    df["meals_prepared"] = meals_prepared_list
    df["meals_consumed"] = meals_consumed_list
    
    # Derive operational surplus
    df["surplus_meals"] = df["meals_prepared"] - df["meals_consumed"]
    df["surplus_percentage"] = ((df["surplus_meals"] / df["meals_prepared"]) * 100).round(2)
    
    # Sort by Meal, Food_Category, Date to compute accurate historical lagged consumption
    df = df.sort_values(by=["Meal", "Food_Category", "Date"]).reset_index(drop=True)
    df["previous_day_consumption"] = df.groupby(["Meal", "Food_Category"])["meals_consumed"].shift(1)
    
    # Backfill first day's lag using group mean
    df["previous_day_consumption"] = df.groupby(["Meal", "Food_Category"])["previous_day_consumption"].transform(
        lambda s: s.fillna(s.mean() if not np.isnan(s.mean()) else df["meals_consumed"].mean())
    ).round().astype(int)
    
    # Sort back chronologically by Date
    df = df.sort_values(by=["Date", "Meal", "Canteen_Section"]).reset_index(drop=True)
    
    logger.info("Operational feature augmentation complete.")
    return df

def generate_and_save_processed_data() -> pd.DataFrame:
    """
    Execute full ETL pipeline: Load raw authentic -> Clean -> Derive dates -> Augment -> Save.
    """
    raw_df = load_raw_data(RAW_DATA_PATH)
    clean_df = clean_raw_data(raw_df)
    dated_df = derive_temporal_features(clean_df)
    processed_df = augment_operational_features(dated_df, random_seed=42)
    
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info(f"Processed dataset saved successfully to {PROCESSED_DATA_PATH} ({len(processed_df)} rows)")
    return processed_df

if __name__ == "__main__":
    generate_and_save_processed_data()
