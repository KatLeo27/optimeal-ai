"""
OptiMeal AI - Food Waste Analytics Engine
Computes statistical waste metrics, category distributions, temporal trends, and cost impacts.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
from src.data_loader import load_processed_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def compute_executive_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute headline food waste and meal preparation KPIs.
    """
    total_waste_kg = round(float(df["Waste_Weight_kg"].sum()), 2)
    total_cost_loss = round(float(df["Cost_Loss"].sum()), 2)
    
    unique_days = df["Date"].nunique()
    avg_daily_waste_kg = round(total_waste_kg / unique_days, 2) if unique_days > 0 else 0.0
    avg_daily_cost_loss = round(total_cost_loss / unique_days, 2) if unique_days > 0 else 0.0
    
    total_prepared = int(df["meals_prepared"].sum()) if "meals_prepared" in df.columns else 0
    total_consumed = int(df["meals_consumed"].sum()) if "meals_consumed" in df.columns else 0
    total_surplus = total_prepared - total_consumed
    
    surplus_rate = round((total_surplus / total_prepared * 100), 2) if total_prepared > 0 else 0.0
    
    return {
        "total_waste_kg": total_waste_kg,
        "total_cost_loss": total_cost_loss,
        "avg_daily_waste_kg": avg_daily_waste_kg,
        "avg_daily_cost_loss": avg_daily_cost_loss,
        "total_meals_prepared": total_prepared,
        "total_meals_consumed": total_consumed,
        "total_surplus_meals": total_surplus,
        "surplus_rate_pct": surplus_rate,
        "total_operating_days": unique_days,
        "total_observations": len(df)
    }

def get_waste_by_meal_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate waste weight and cost by meal type (Breakfast, Lunch, Dinner).
    """
    grp = df.groupby("Meal").agg(
        total_waste_kg=("Waste_Weight_kg", "sum"),
        avg_waste_kg=("Waste_Weight_kg", "mean"),
        total_cost_loss=("Cost_Loss", "sum"),
        records_count=("Meal", "count")
    ).reset_index()
    grp["waste_share_pct"] = (grp["total_waste_kg"] / grp["total_waste_kg"].sum() * 100).round(1)
    return grp.sort_values(by="total_waste_kg", ascending=False)

def get_waste_by_food_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate waste weight and cost by food category (Meat, Rice, Soup, Vegetables).
    """
    grp = df.groupby("Food_Category").agg(
        total_waste_kg=("Waste_Weight_kg", "sum"),
        avg_waste_kg=("Waste_Weight_kg", "mean"),
        total_cost_loss=("Cost_Loss", "sum"),
        avg_unit_price=("Unit_Price_per_kg", "mean"),
        records_count=("Food_Category", "count")
    ).reset_index()
    grp["cost_share_pct"] = (grp["total_cost_loss"] / grp["total_cost_loss"].sum() * 100).round(1)
    grp["waste_share_pct"] = (grp["total_waste_kg"] / grp["total_waste_kg"].sum() * 100).round(1)
    return grp.sort_values(by="total_cost_loss", ascending=False)

def get_waste_by_section(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate waste by canteen section.
    """
    grp = df.groupby("Canteen_Section").agg(
        total_waste_kg=("Waste_Weight_kg", "sum"),
        total_cost_loss=("Cost_Loss", "sum"),
        avg_waste_kg=("Waste_Weight_kg", "mean"),
        records_count=("Canteen_Section", "count")
    ).reset_index()
    return grp.sort_values(by="total_waste_kg", ascending=False)

def get_daily_waste_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily aggregated time-series with rolling 7-day averages.
    """
    daily = df.groupby("Date").agg(
        daily_waste_kg=("Waste_Weight_kg", "sum"),
        daily_cost_loss=("Cost_Loss", "sum"),
        daily_meals_prepared=("meals_prepared", "sum"),
        daily_meals_consumed=("meals_consumed", "sum"),
        daily_surplus_meals=("surplus_meals", "sum")
    ).reset_index()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.sort_values(by="Date").reset_index(drop=True)
    
    # 7-day rolling moving averages
    daily["waste_kg_7d_mavg"] = daily["daily_waste_kg"].rolling(window=7, min_periods=1).mean().round(2)
    daily["cost_loss_7d_mavg"] = daily["daily_cost_loss"].rolling(window=7, min_periods=1).mean().round(2)
    
    return daily

def get_highest_waste_days(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Identify and rank the highest waste operating days.
    """
    daily = df.groupby(["Date", "Day_of_Week"]).agg(
        total_waste_kg=("Waste_Weight_kg", "sum"),
        total_cost_loss=("Cost_Loss", "sum"),
        meals_prepared=("meals_prepared", "sum"),
        meals_consumed=("meals_consumed", "sum"),
        surplus_meals=("surplus_meals", "sum"),
        avg_temp=("temperature", "mean"),
        total_rain=("rainfall", "max"),
        is_event=("special_event", "max"),
        is_holiday=("holiday", "max")
    ).reset_index()
    
    daily["surplus_rate_pct"] = ((daily["surplus_meals"] / daily["meals_prepared"]) * 100).round(1)
    daily["total_waste_kg"] = daily["total_waste_kg"].round(2)
    daily["total_cost_loss"] = daily["total_cost_loss"].round(2)
    
    return daily.sort_values(by="total_waste_kg", ascending=False).head(top_n).reset_index(drop=True)

if __name__ == "__main__":
    data = load_processed_data()
    kpis = compute_executive_kpis(data)
    print("Executive KPIs:")
    for k, v in kpis.items():
        print(f"  {k}: {v}")
    print("\nTop 5 Highest Waste Days:")
    print(get_highest_waste_days(data, 5))
