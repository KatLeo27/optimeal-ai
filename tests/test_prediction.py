"""
Unit tests for Demand Prediction, Waste Analysis, Anomaly Detection, and Optimization Engines.
"""

import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_processed_data
from src.demand_prediction import DemandPredictor
from src.waste_analysis import compute_executive_kpis, get_waste_by_meal_type, get_waste_by_food_category
from src.anomaly_detection import detect_statistical_anomalies, detect_isolation_forest_anomalies
from src.recommendations import optimize_meal_preparation
from src.rag import SustainabilityRAGAssistant

class TestCoreModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_processed_data()
        cls.predictor = DemandPredictor()
        cls.rag = SustainabilityRAGAssistant()

    def test_demand_prediction(self):
        input_data = {
            "Meal": "Lunch",
            "Food_Category": "Vegetables",
            "Canteen_Section": "A",
            "Day_of_Week": "Wednesday",
            "expected_attendance": 500,
            "temperature": 28.0,
            "rainfall": 0.0,
            "holiday": 0,
            "special_event": 0,
            "previous_day_consumption": 480,
            "Is_Weekend": 0,
            "Month": 7,
            "Day": 10
        }
        res = self.predictor.predict_demand(input_data)
        self.assertIn("predicted_consumption", res)
        self.assertGreater(res["predicted_consumption"], 0)
        self.assertLessEqual(res["lower_bound_90pct"], res["predicted_consumption"])
        self.assertGreaterEqual(res["upper_bound_90pct"], res["predicted_consumption"])

    def test_waste_analysis_kpis(self):
        kpis = compute_executive_kpis(self.df)
        self.assertGreater(kpis["total_waste_kg"], 0)
        self.assertGreater(kpis["total_cost_loss"], 0)
        self.assertGreater(kpis["avg_daily_waste_kg"], 0)

        meal_df = get_waste_by_meal_type(self.df)
        self.assertEqual(len(meal_df), 3) # Breakfast, Lunch, Dinner

        cat_df = get_waste_by_food_category(self.df)
        self.assertGreaterEqual(len(cat_df), 4)

    def test_anomaly_detection(self):
        stat_anoms = detect_statistical_anomalies(self.df)
        self.assertIn("is_statistical_anomaly", stat_anoms.columns)
        self.assertIn("z_score", stat_anoms.columns)

        iso_anoms = detect_isolation_forest_anomalies(self.df)
        self.assertIn("is_iso_anomaly", iso_anoms.columns)

    def test_recommendation_optimization(self):
        rec = optimize_meal_preparation(
            predicted_demand=500,
            expected_attendance=540,
            meal_type="Lunch",
            food_category="Vegetables",
            special_event=0,
            holiday=0
        )
        self.assertGreaterEqual(rec["recommended_preparation"], 500)
        self.assertGreater(rec["optimized_buffer_pct"], 0)
        self.assertIn("operational_rationale", rec)

    def test_rag_assistant(self):
        res = self.rag.query_knowledge_base("What is the EPA food recovery hierarchy?")
        self.assertIn("answer", res)
        self.assertGreater(len(res["retrieved_sources"]), 0)

if __name__ == "__main__":
    unittest.main()
