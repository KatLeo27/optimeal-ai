"""
FoodWise AI - Synthetic Data Generator
Generates realistic institutional food service operations data with realistic correlations,
seasonal dynamics, weather impacts, and food waste calculations.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_food_waste_data(num_days=730, random_seed=42, output_path="data/food_waste_data.csv"):
    """
    Generate realistic multi-year food service data.
    
    Parameters:
    - num_days: Number of days of historical operations (default 730 = ~2 years)
    - random_seed: Random seed for reproducibility
    - output_path: Destination path for CSV output
    """
    np.random.seed(random_seed)
    
    start_date = datetime(2024, 1, 1)
    meal_types = ["Breakfast", "Lunch", "Dinner"]
    menu_categories = ["Vegetarian", "Non-Vegetarian", "Vegan", "Mixed/Continental"]
    
    # Typical base attendance by meal type
    meal_base_attendance = {
        "Breakfast": 260,
        "Lunch": 620,
        "Dinner": 410
    }
    
    # Waste characteristics by category (avg kg per prepared/consumed unit)
    category_waste_factor = {
        "Vegetarian": 1.05,
        "Non-Vegetarian": 0.95,
        "Vegan": 1.12,
        "Mixed/Continental": 1.00
    }
    
    records = []
    
    # Track historical consumption to simulate realistic lag
    last_consumption_tracker = {
        ("Breakfast", "Vegetarian"): 240,
        ("Breakfast", "Non-Vegetarian"): 250,
        ("Breakfast", "Vegan"): 220,
        ("Breakfast", "Mixed/Continental"): 245,
        ("Lunch", "Vegetarian"): 580,
        ("Lunch", "Non-Vegetarian"): 610,
        ("Lunch", "Vegan"): 540,
        ("Lunch", "Mixed/Continental"): 590,
        ("Dinner", "Vegetarian"): 380,
        ("Dinner", "Non-Vegetarian"): 400,
        ("Dinner", "Vegan"): 360,
        ("Dinner", "Mixed/Continental"): 390,
    }
    
    for day_offset in range(num_days):
        current_date = start_date + timedelta(days=day_offset)
        day_of_week = current_date.strftime("%A")
        is_weekend = day_of_week in ["Saturday", "Sunday"]
        
        # Seasonality and weather
        day_of_year = current_date.timetuple().tm_yday
        # Temperature: sinusoidal curve peaking around July (day 195)
        base_temp = 20 + 10 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
        temperature = round(float(np.random.normal(base_temp, 3.5)), 1)
        
        # Rainfall: episodic with higher probability during monsoon/rainy season
        rain_prob = 0.25 if 150 <= day_of_year <= 270 else 0.12
        is_raining = np.random.rand() < rain_prob
        rainfall = round(float(np.random.exponential(12.0) if is_raining else 0.0), 1)
        
        # Public holidays (approx ~15 days a year)
        is_holiday = int(np.random.rand() < 0.04)
        
        # Special institutional events (festivals, guest lectures, conferences)
        is_special_event = int(np.random.rand() < 0.06 and not is_holiday)
        
        for meal in meal_types:
            # Randomly select primary menu category for the shift or cycle
            menu_cat = np.random.choice(menu_categories, p=[0.35, 0.40, 0.10, 0.15])
            
            # Base expected attendance
            base_att = meal_base_attendance[meal]
            
            # Weekend impact: lunch/dinner decrease, breakfast moderate
            if is_weekend:
                base_att *= 0.55
                
            # Holiday impact
            if is_holiday:
                base_att *= 0.35
                
            # Special event impact
            if is_special_event:
                base_att *= 1.35
                
            # Weather impact: heavy rain slightly lowers walk-in attendance
            if rainfall > 20:
                base_att *= 0.92
                
            expected_attendance = int(np.clip(np.random.normal(base_att, base_att * 0.08), 40, 1200))
            
            # Actual attendance fluctuates around expected
            attendance_noise = np.random.normal(0, expected_attendance * 0.06)
            actual_attendance = int(np.clip(expected_attendance + attendance_noise, 30, expected_attendance * 1.3))
            
            # Kitchen preparation strategy:
            # Kitchen prepares based on expected attendance + safety buffer (usually 5% to 15%)
            buffer_pct = np.random.uniform(0.06, 0.14)
            meals_prepared = int(np.ceil(expected_attendance * (1 + buffer_pct)))
            
            # Meals consumed:
            # Bound by actual attendance and food prepared
            consumption_rate = np.random.uniform(0.92, 0.98)
            raw_consumed = int(round(actual_attendance * consumption_rate))
            meals_consumed = min(raw_consumed, meals_prepared)
            
            # Food waste computation:
            # 1. Unserved Overproduction: each surplus meal contributes ~0.45 kg of unconsumed food
            unserved_meals = max(0, meals_prepared - meals_consumed)
            unserved_waste_kg = unserved_meals * np.random.uniform(0.38, 0.48)
            
            # 2. Plate waste / kitchen prep scraps: ~0.05 to 0.08 kg per meal served
            plate_waste_kg = meals_consumed * np.random.uniform(0.04, 0.07)
            
            # Category multiplier
            cat_factor = category_waste_factor[menu_cat]
            
            # Total waste in kg
            food_waste_kg = round(float((unserved_waste_kg + plate_waste_kg) * cat_factor), 2)
            
            # Previous day consumption lag for this meal + category combo
            prev_key = (meal, menu_cat)
            prev_consumption = last_consumption_tracker[prev_key]
            last_consumption_tracker[prev_key] = meals_consumed
            
            records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "day_of_week": day_of_week,
                "meal_type": meal,
                "menu_category": menu_cat,
                "expected_attendance": expected_attendance,
                "actual_attendance": actual_attendance,
                "meals_prepared": meals_prepared,
                "meals_consumed": meals_consumed,
                "food_waste_kg": food_waste_kg,
                "special_event": is_special_event,
                "holiday": is_holiday,
                "temperature": temperature,
                "rainfall": rainfall,
                "previous_day_consumption": prev_consumption
            })
            
    df = pd.DataFrame(records)
    
    # Ensure destination directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records saved to {output_path}")
    print(df.head())
    print("\nDataset Summary:")
    print(df.describe().T[["mean", "std", "min", "50%", "max"]])
    return df

if __name__ == "__main__":
    generate_food_waste_data()
