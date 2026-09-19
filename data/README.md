# OptiMeal AI — Dataset Provenance & Data Dictionary

## Overview
This document specifies the origin, lineage, categorization, and transformation methodology for all data utilized within **OptiMeal AI**.

In accordance with strict **Responsible AI** and data transparency standards, we explicitly distinguish between:
1. **Authentic Source Data**: Real-world recorded observations from institutional canteen operations.
2. **Derived Variables**: Exact mathematical and temporal transformations computed directly from authentic columns.
3. **Synthetic / Augmented Variables**: Statistically grounded operational features generated to support predictive demand optimization where turnstile or environmental telemetry was absent.

---

## 1. Authentic Source Dataset (`data/raw/Dataset Propely.csv`)
- **Source**: Operational food waste logs from institutional dining / canteen facilities.
- **Records**: 2,600 observations across 61 operating dates (June 11, 2025 to August 10, 2025).
- **Original Columns**:
  - `Date` (ISO Date string: `YYYY-MM-DD`): Recording date.
  - `Meal` (Categorical: `Breakfast`, `Lunch`, `Dinner`): Operational shift.
  - `Canteen_Section` (Categorical: `A`, `B`, `C`, `D`): Service station / dining quadrant.
  - `Food_Category` (Categorical: `Vegetables`, `Rice`, `Soup`, `Meat`): Specific food stream.
  - `Waste_Weight_kg` (Continuous float: `0.1` to `5.0` kg): Weighed discarded food per station batch.
  - `Unit_Price_per_kg` (Continuous float: `1.5` to `8.0` USD/EUR): Ingredient cost per kilogram.
  - `Cost_Loss` (Continuous float: `0.15` to `40.0` USD/EUR): Direct financial loss (`Waste_Weight_kg * Unit_Price_per_kg`).

---

## 2. Derived Variables (Computed Deterministically)
- `Day_of_Week` (String: `Monday` - `Sunday`): Extracted from authentic `Date`.
- `Is_Weekend` (Binary integer: `1` for Saturday/Sunday, `0` otherwise).
- `Month` (Integer: `6`, `7`, `8`).
- `Day` (Integer: `1` - `31`).
- `Section_Daily_Waste_kg`: Aggregation of waste weight across categories for a given date and section.
- `Meal_Daily_Cost_Loss`: Aggregated financial waste loss per meal session.

---

## 3. Synthetic / Augmented Variables (`data/processed/food_waste_processed.csv`)

### Why was augmentation necessary?
The authentic dataset provides precise food waste weight (kg), price ($), and categorical logs per station, but does not capture raw guest turnstile taps, historical kitchen prep sheets, or local weather sensor readings. To support the end-to-end Machine Learning demand prediction and optimization workflow, these variables were generated following observed operational distributions.

| Feature Name | Type | Range / Domain | Statistical Logic & Relationship |
| :--- | :--- | :--- | :--- |
| `expected_attendance` | Integer | 80 – 850 | Base capacity scaled by meal type (Lunch > Dinner > Breakfast), section, and weekend reduction (-45%). |
| `actual_attendance` | Integer | 60 – 900 | Gaussian fluctuation around expected attendance ($\sigma \approx 6\%$), modified by events and rainfall. |
| `meals_prepared` | Integer | 90 – 950 | Traditional kitchen prep policy: `expected_attendance * (1 + buffer)`, where historical buffer was 7–14%. |
| `meals_consumed` | Integer | 55 – 880 | Primary ML target. Function of actual attendance ($0.93 - 0.98$ consumption rate) bounded by `meals_prepared`. |
| `previous_day_consumption`| Integer | 50 – 850 | Lag-1 consumption for the corresponding meal type and food category. |
| `temperature` | Float | 22.0 – 35.0 °C | Realistic summer daily temperature profile. |
| `rainfall` | Float | 0.0 – 45.0 mm | Episodic precipitation with higher probability on monsoon/rainy days. |
| `special_event` | Binary (0/1) | 0 or 1 | Institutional conferences, banquets, or campus events (+35% attendance boost). |
| `holiday` | Binary (0/1) | 0 or 1 | Recognized public holidays (-65% attendance drop). |

### Reproducibility
All augmented features are generated using a fixed random seed (`seed=42`) via `src/feature_engineering.py`.
