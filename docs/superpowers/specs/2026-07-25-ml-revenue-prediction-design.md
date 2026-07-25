# ML Revenue Prediction — Design Spec

**Date:** 2026-07-25
**Feature:** Predict `total_amount` per trip using PySpark MLlib

---

## Goal

Train two regression models (Linear Regression baseline + Gradient Boosted Trees) on preprocessed Yellow Taxi data to predict total trip revenue (`total_amount`). Compare RMSE and R² between models to demonstrate improvement.

---

## Target

`total_amount` — fare + tip + tolls + surcharges (full amount the passenger pays).

---

## Features

| Feature | Type | Transformation |
|---------|------|----------------|
| `trip_distance` | numeric | used as-is |
| `pickup_hour` | numeric | used as-is |
| `day_of_week` | numeric | used as-is |
| `passenger_count` | numeric | used as-is |
| `PULocationID` | categorical | StringIndexer → numeric index |
| `DOLocationID` | categorical | StringIndexer → numeric index |
| `pay_credit_card` | binary | already encoded |
| `pay_cash` | binary | already encoded |

All assembled into a single `features` vector via VectorAssembler.

---

## Pipeline

1. `StringIndexer` on `PULocationID` → `pu_idx`, `DOLocationID` → `do_idx`
2. `VectorAssembler` combines all 8 features into `features` column
3. 80/20 train/test split (`seed=42`)
4. Train **LinearRegression** (baseline): `maxIter=100, regParam=0.1`
5. Train **GBTRegressor** (main model): `maxIter=50, maxDepth=5, seed=42`
6. Evaluate both on test set: **RMSE** and **R²** via `RegressionEvaluator`
7. Save both models to `output/models/lr/` and `output/models/gbt/`
8. Print comparison table

---

## Evaluation Metrics

- **RMSE** (root mean squared error) — primary metric, in dollars
- **R²** (coefficient of determination) — how much variance the model explains

Expected baseline (LR): RMSE ~$4–6, R² ~0.85
Expected main (GBT): RMSE ~$2–4, R² ~0.92+

---

## Files

| File | Responsibility |
|------|----------------|
| `work/ml_features.py` | `build_features(df)` — StringIndexer + VectorAssembler, returns transformed DataFrame |
| `work/04_ml.py` | Load clean data, train both models, evaluate, save, print comparison |
| `tests/test_ml.py` | Unit tests: feature vector has correct length, no nulls, target column present |
| `cloud/04_ml_cloud.py` | Same as `04_ml.py` with S3 input/output paths, no local Spark config |

---

## Constraints

- Uses only features available in the already-preprocessed `taxi_clean/` Parquet — no new preprocessing step needed.
- Models saved as PySpark ML PipelineModel format (portable, loadable for inference).
- No hyperparameter tuning (CrossValidator) — out of scope for this timeline.
