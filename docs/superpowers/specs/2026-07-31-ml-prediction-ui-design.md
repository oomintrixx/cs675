# ML Prediction UI — Design Spec

**Date:** 2026-07-31
**Feature:** Streamlit UI to predict `total_amount` from trained taxi ML models

---

## Goal

Build a local Streamlit web app that lets a user fill in trip feature values (pickup/dropoff zone, distance, time, passenger count, payment method) and get a predicted `total_amount` from the best-performing trained model (LR / GBT / RF).

---

## Problem: current saved models are not inference-ready

`work/04_ml.py` currently trains on a `features` vector produced by `work/ml_features.py::build_features()`, which internally fits a `Pipeline([StringIndexer(PULocationID), StringIndexer(DOLocationID), VectorAssembler])` and returns an already-transformed DataFrame. Only the bare regressor (`lr_model`, `gbt_model`, `rf_model`) is saved to `output/models/{lr,gbt,rf}/` — the fitted `StringIndexer` zone-ID → index mapping is never persisted.

This means a UI that loads only the saved regressor cannot correctly turn a raw `PULocationID` into the `pu_idx` the model was trained on.

**Fix:** retrain so the full pipeline (indexers + assembler + regressor) is fit and saved as one `PipelineModel`. Inference then becomes: `PipelineModel.load(path).transform(raw_single_row_df)`.

---

## Changes to training code

### `work/ml_features.py`

Add a new function alongside the existing `build_features` (left untouched — still covered by its existing tests in `tests/test_ml.py`):

```python
def build_pipeline(regressor) -> Pipeline:
    """
    Returns an untrained Pipeline: StringIndexer(PULocationID) + StringIndexer(DOLocationID)
    + VectorAssembler(FEATURE_COLS) + the given regressor stage.
    Callers fit() this once on raw training data and save the resulting
    PipelineModel — feature transform and model travel together for inference.
    """
```

### `work/04_ml.py`

Rewrite to, for each of LR / GBT / RF:
1. Build `build_pipeline(regressor)`
2. `pipeline.fit(train_df)` → single `PipelineModel`
3. `pipeline_model.transform(test_df)` → evaluate RMSE / R²
4. `pipeline_model.write().overwrite().save(OUTPUT_PATH + "models/<name>/")`

After training all three, write `output/models/metrics.json`:
```json
{
  "lr":  {"rmse": 4.82, "r2": 0.9123},
  "gbt": {"rmse": 2.41, "r2": 0.9654},
  "rf":  {"rmse": 2.55, "r2": 0.9601}
}
```

This overwrites the existing (non-functional-for-inference) contents of `output/models/lr|gbt|rf/`.

Out of scope: `cloud/04_ml_cloud.py` is not updated in this pass (UI is local-only per design decision below); it can get the same treatment later if the UI needs to serve cloud-trained models.

---

## Zone lookup data

Download NYC TLC's public `taxi_zone_lookup.csv` (same `d37ci6vzurychx.cloudfront.net` domain already used by `scripts/download_full_data.sh`) via a new `scripts/download_zone_lookup.sh`, saved to `data/taxi_zone_lookup.csv`. Columns: `LocationID, Borough, Zone, service_zone`. Loaded with plain pandas in the UI (no Spark needed for this small lookup) to populate the pickup/dropoff dropdowns as `"{Zone}, {Borough}"` labels mapped back to `LocationID`.

---

## Streamlit UI (`ui/predict_app.py`)

**Startup (cached via `@st.cache_resource`):**
- `get_spark("predict_ui")` from `work.spark_helper`
- Read `output/models/metrics.json`, pick the model name with lowest `rmse` as "best"
- `PipelineModel.load(OUTPUT_PATH + f"models/{best}/")`
- Load `data/taxi_zone_lookup.csv` via pandas

If `metrics.json` or the model directory is missing, show an `st.error` telling the user to run `make run-ml` first, and stop.

**Form fields:**

| Field | Widget | Notes |
|---|---|---|
| Pickup zone | `st.selectbox`, labeled `"{Zone}, {Borough}"` | maps to `PULocationID` |
| Dropoff zone | `st.selectbox`, same labeling | maps to `DOLocationID` |
| Trip distance (mi) | `st.number_input`, min 0.1, default 2.5 | `trip_distance` |
| Pickup hour | `st.slider` 0–23 | `pickup_hour` |
| Day of week | `st.selectbox` Sunday…Saturday | mapped to Spark `dayofweek()` convention: Sunday=1 … Saturday=7 |
| Passenger count | `st.number_input`, int, 1–6, default 1 | `passenger_count` |
| Payment method | `st.radio`: Credit Card / Cash | derives `pay_credit_card` / `pay_cash` — mutually exclusive by construction, avoids invalid combinations |

**Predict action:**
1. Build a dict of the raw feature columns above (pure helper function `build_input_row(...)`, unit-testable)
2. `spark.createDataFrame([Row(**row)])`
3. `model.transform(df)` → collect `prediction`
4. Display with `st.metric("Predicted total amount", f"${prediction:.2f}")`
5. Caption showing which model is serving the prediction and its test RMSE/R² (e.g. "Served by GBT — RMSE $2.41, R² 0.965")

---

## Testing

- `tests/test_ml.py`: add tests for `build_pipeline()` — fitting it (with a cheap `LinearRegression` stage) on tiny sample data produces a `PipelineModel` whose `.transform()` yields a `prediction` column, handles an unseen zone ID via `handleInvalid="keep"` without erroring.
- Pure helper functions in the UI module (payment-method → flags, raw-input → row dict) get unit tests in `tests/test_ui.py` — no Spark session needed for these.
- Manual verification: run `make run-ui`, exercise the form in a browser, confirm a prediction renders.

---

## Deployment scope

Local only — runs inside the existing `pyspark` Docker service, reads models from local `data/output/models/`. No S3/cloud model loading in this pass.

**Docker/Makefile changes:**
- `pyproject.toml`: add `streamlit` dependency
- `docker-compose.yml`: expose port `8501` on the `pyspark` service
- `Makefile`: add `run-ui` target running `streamlit run ui/predict_app.py --server.port 8501 --server.address 0.0.0.0` inside the container
- `README.md`: add a "Run the prediction UI" section

---

## Constraints

- No changes to `work/preprocess_steps.py` or the analytics pipeline.
- `build_features()` in `ml_features.py` stays as-is for backward compatibility with its existing tests; `build_pipeline()` is additive.
- Zone dropdowns cover all ~265 TLC zones regardless of whether they appear in the trained slice — `handleInvalid="keep"` on the StringIndexers means an unseen zone at inference time maps to a catch-all index rather than erroring.
