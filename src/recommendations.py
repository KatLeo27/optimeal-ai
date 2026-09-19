"""
OptiMeal AI - Meal Optimization & Recommendation Engine
Calculates optimal meal preparation targets, dynamic safety margins, surplus forecasts,
and operational decision rationale.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
from typing import Dict, Any

def optimize_meal_preparation(
    predicted_demand: int,
    expected_attendance: int,
    meal_type: str = "Lunch",
    food_category: str = "Vegetables",
    special_event: int = 0,
    holiday: int = 0,
    rainfall: float = 0.0,
    historical_buffer_rate: float = 0.12 # Traditional 12% overproduction
) -> Dict[str, Any]:
    """
    Calculate data-driven preparation recommendation balancing food availability with waste reduction.
    
    Parameters:
    - predicted_demand: ML predicted consumption count
    - expected_attendance: Kitchen manager's headcount estimate
    - meal_type: Breakfast, Lunch, Dinner
    - food_category: Meat, Rice, Soup, Vegetables
    - special_event: Binary event flag
    - holiday: Binary holiday flag
    - rainfall: Expected precipitation (mm)
    - historical_buffer_rate: Baseline traditional kitchen prep buffer
    
    Returns:
    - dict: Recommended prep quantity, buffer %, surplus, potential savings, and reasoning.
    """
    # 1. Calibrate dynamic safety buffer
    base_buffer = 0.04 # 4% base data-driven buffer
    
    # Adjust for operational volatility
    if special_event:
        base_buffer += 0.03 # +3% during events due to attendee variance
    if holiday:
        base_buffer -= 0.01 # -1% on holidays
    if rainfall > 15.0:
        base_buffer -= 0.01 # -1% for heavy rain
        
    # Food category sensitivity
    if food_category == "Meat":
        base_buffer += 0.01 # Meat carries higher guest satisfaction sensitivity
    elif food_category == "Soup":
        base_buffer -= 0.01 # Soup is easily held or batched
        
    # Clamp safety buffer between 3% and 8%
    optimized_buffer_pct = max(0.03, min(0.08, base_buffer))
    
    # 2. Recommended preparation
    recommended_prep = int(math.ceil(predicted_demand * (1 + optimized_buffer_pct)))
    
    # 3. Traditional prep comparison (for what-if and savings benchmarking)
    traditional_prep = int(math.ceil(expected_attendance * (1 + historical_buffer_rate)))
    
    # 4. Expected surplus and waste risk
    expected_surplus = max(0, recommended_prep - predicted_demand)
    traditional_surplus = max(0, traditional_prep - predicted_demand)
    
    meals_saved = max(0, traditional_prep - recommended_prep)
    estimated_waste_kg_saved = round(meals_saved * 0.42, 2) # avg ~0.42 kg waste avoided per surplus meal
    
    # Determine waste risk indicator
    surplus_ratio = expected_surplus / max(1, recommended_prep)
    if surplus_ratio < 0.05:
        risk_level = "Low"
        risk_color = "#10B981" # Green
    elif surplus_ratio < 0.08:
        risk_level = "Moderate"
        risk_color = "#F59E0B" # Amber
    else:
        risk_level = "Elevated"
        risk_color = "#EF4444" # Red
        
    # 5. Formulate transparent reasoning
    reasons = []
    reasons.append(
        f"Base demand forecast is {predicted_demand:,} portions based on attendance ({expected_attendance:,}) and temporal factors."
    )
    if special_event:
        reasons.append("Applied +3.0% buffer adjustment for special event attendance volatility.")
    if rainfall > 15.0:
        reasons.append(f"Reduced buffer by 1.0% due to expected precipitation ({rainfall} mm).")
    if food_category == "Meat":
        reasons.append("Maintained a protective buffer of 5.0% for high-protein entrée demand.")
        
    reasons.append(
        f"Dynamic safety buffer set at {optimized_buffer_pct*100:.1f}% (vs traditional {historical_buffer_rate*100:.1f}% fixed buffer), "
        f"preventing ~{meals_saved} overproduced meals while maintaining high service reliability."
    )
    
    rationale = " ".join(reasons)
    
    return {
        "predicted_demand": predicted_demand,
        "recommended_preparation": recommended_prep,
        "traditional_preparation": traditional_prep,
        "optimized_buffer_pct": round(optimized_buffer_pct * 100, 1),
        "expected_surplus": expected_surplus,
        "traditional_surplus": traditional_surplus,
        "meals_saved_estimate": meals_saved,
        "estimated_waste_kg_saved": estimated_waste_kg_saved,
        "waste_risk_level": risk_level,
        "waste_risk_color": risk_color,
        "operational_rationale": rationale
    }

if __name__ == "__main__":
    rec = optimize_meal_preparation(
        predicted_demand=580,
        expected_attendance=620,
        meal_type="Lunch",
        food_category="Vegetables",
        special_event=0,
        holiday=0
    )
    print("Optimization Output:")
    for k, v in rec.items():
        print(f"  {k}: {v}")
