"""
Unit tests for data loader and data processing modules.
"""

import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_raw_data, load_processed_data, inspect_dataset_summary
from src.data_processing import clean_raw_data, derive_temporal_features

class TestDataPipeline(unittest.TestCase):
    def test_load_raw_data(self):
        df = load_raw_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn("Date", df.columns)
        self.assertIn("Waste_Weight_kg", df.columns)

    def test_data_cleaning_and_derivation(self):
        df = load_raw_data()
        cleaned = clean_raw_data(df)
        dated = derive_temporal_features(cleaned)
        
        self.assertIn("Day_of_Week", dated.columns)
        self.assertIn("Is_Weekend", dated.columns)
        self.assertTrue((dated["Waste_Weight_kg"] >= 0).all())
        self.assertTrue((dated["Cost_Loss"] >= 0).all())

    def test_load_processed_data(self):
        df = load_processed_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("meals_consumed", df.columns)
        self.assertIn("expected_attendance", df.columns)
        self.assertIn("previous_day_consumption", df.columns)

if __name__ == "__main__":
    unittest.main()
