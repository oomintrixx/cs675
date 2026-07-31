# ML Prediction UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Streamlit app where a user fills in trip feature values and gets a predicted `total_amount` from the best-performing trained model.

**Architecture:** Retrain the three regression models as full `PipelineModel`s (StringIndexer + StringIndexer + VectorAssembler + regressor fit together) so a saved model can transform raw input directly. A new `work/predict_helpers.py` holds pure, unit-tested functions for form-to-feature-row mapping. `ui/predict_app.py` is a thin Streamlit wrapper that loads the best model (by RMSE from `metrics.json`), the TLC zone lookup CSV, renders the form, and displays the prediction.

**Tech Stack:** PySpark 3.5 MLlib (`Pipeline`, `PipelineModel`), Streamlit, pandas, pytest — all inside the existing `pyspark` Docker service.

## Global Constraints

- Local only — no S3/cloud model loading in this feature.
- `work/ml_features.py::build_features()` must remain unchanged (existing tests in `tests/test_ml.py` depend on it) — `build_pipeline()` is additive, not a replacement.
- Payment method is a single mutually-exclusive choice (Credit Card / Cash) — never expose `pay_credit_card`/`pay_cash` as two independent checkboxes.
- Day-of-week values passed into the model must use Spark's `dayofweek()` convention: Sunday=1 … Saturday=7.
- Zone dropdowns use the NYC TLC `taxi_zone_lookup.csv` downloaded from `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv` (same CDN already used by `scripts/download_full_data.sh`).
- `data/` is gitignored — downloaded CSVs and trained models are never committed.

---

### Task 1: Add `build_pipeline()` to `work/ml_features.py`

**Files:**
- Modify: `work/ml_features.py`
- Test: `tests/test_ml.py`

**Interfaces:**
- Consumes: `FEATURE_COLS` (existing list in `work/ml_features.py`)
- Produces: `build_pipeline(regressor) -> Pipeline` — an **untrained** `pyspark.ml.Pipeline` with stages `[pu_indexer, do_indexer, assembler, regressor]`. Task 2 calls `.fit(train_df)` on this to get one `PipelineModel` per model type.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ml.py` (reuses the existing `make_clean_rows` helper already defined at the top of the file):

```python
class TestBuildPipeline:
    def test_produces_prediction_column(self, spark):
        from work.ml_features import build_pipeline
        from pyspark.ml.regression import LinearRegression

        df = make_clean_rows(spark, n=5)
        pipeline = build_pipeline(
            LinearRegression(featuresCol="features", labelCol="total_amount")
        )
        model = pipeline.fit(df)
        result = model.transform(df)
        assert "prediction" in result.columns

    def test_handles_unseen_zone_id_at_inference(self, spark):
        from work.ml_features import build_pipeline
        from pyspark.ml.regression import LinearRegression
        from pyspark.sql import Row
        import datetime

        train_df = make_clean_rows(spark, n=5)  # PULocationID/DOLocationID always 161/236
        pipeline = build_pipeline(
            LinearRegression(featuresCol="features", labelCol="total_amount")
        )
        model = pipeline.fit(train_df)

        unseen_row = spark.createDataFrame([Row(
            pickup_date=datetime.date(2022, 1, 15),
            pickup_hour=10,
            day_of_week=7,
            time_of_day="morning",
            PULocationID=999,
            DOLocationID=999,
            trip_distance=2.5,
            distance_bucket="medium",
            fare_amount=12.0,
            fare_norm=0.5,
            tip_amount=2.0,
            total_amount=15.5,
            passenger_count=1,
            payment_type=1,
            pay_credit_card=1,
            pay_cash=0,
            pay_no_charge=0,
            dist_norm=0.5,
        )])
        result = model.transform(unseen_row)
        assert result.count() == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `docker compose exec pyspark bash -c "cd /home/jovyan && python -m pytest tests/test_ml.py -v -k TestBuildPipeline"`
Expected: `ImportError: cannot import name 'build_pipeline' from 'work.ml_features'`

- [ ] **Step 3: Implement `build_pipeline()`**

Add to `work/ml_features.py`, below the existing `build_features()` function (leave `build_features()` untouched):

```python
def build_pipeline(regressor) -> Pipeline:
    """
    Returns an untrained Pipeline: StringIndexer(PULocationID) + StringIndexer(DOLocationID)
    + VectorAssembler(FEATURE_COLS) + the given regressor stage.

    Unlike build_features(), this pipeline is meant to be fit once on raw
    training data and saved whole — feature transform and model travel
    together, so a loaded PipelineModel can predict directly from raw
    input columns (no separately-tracked StringIndexer mapping needed).
    """
    pu_indexer = StringIndexer(
        inputCol="PULocationID",
        outputCol="pu_idx",
        handleInvalid="keep",
    )
    do_indexer = StringIndexer(
        inputCol="DOLocationID",
        outputCol="do_idx",
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="features",
        handleInvalid="skip",
    )
    return Pipeline(stages=[pu_indexer, do_indexer, assembler, regressor])
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `docker compose exec pyspark bash -c "cd /home/jovyan && python -m pytest tests/test_ml.py -v"`
Expected: all tests pass, including the pre-existing `TestBuildFeatures` ones (unchanged).

- [ ] **Step 5: Commit**

```bash
git add work/ml_features.py tests/test_ml.py
git commit -m "feat: add build_pipeline() for inference-ready ML pipelines"
```

---

### Task 2: Retrain and save full `PipelineModel`s + `metrics.json`

**Files:**
- Modify: `work/04_ml.py`

**Interfaces:**
- Consumes: `build_pipeline(regressor)` from Task 1
- Produces: `data/output/models/{lr,gbt,rf}/` — each a full `PipelineModel` loadable via `PipelineModel.load(path)`. `data/output/models/metrics.json` — `{"lr": {"rmse": float, "r2": float}, "gbt": {...}, "rf": {...}}`, consumed by Task 4's `load_best_model_name()`.

This task is a script, not unit-tested directly — verification is running it against the real local data and inspecting outputs.

- [ ] **Step 1: Rewrite `work/04_ml.py`**

```python
# work/04_ml.py
import json

from work.spark_helper import get_spark
from work.constants import OUTPUT_PATH
from work.ml_features import build_pipeline

from pyspark.ml.regression import LinearRegression, GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = get_spark("04_ml")

# ── Load preprocessed data ──────────────────────────────────────────────────────
print("Loading clean taxi data...")
df = spark.read.parquet(OUTPUT_PATH + "taxi_clean/")
print(f"  Rows: {df.count():,}")

# ── Train / test split ──────────────────────────────────────────────────────────
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"  Train: {train_df.count():,} | Test: {test_df.count():,}")

evaluator_rmse = RegressionEvaluator(
    labelCol="total_amount", predictionCol="prediction", metricName="rmse"
)
evaluator_r2 = RegressionEvaluator(
    labelCol="total_amount", predictionCol="prediction", metricName="r2"
)

# maxBins=300 must exceed the ~262 distinct PU/DOLocationID zones
MODELS = {
    "lr": LinearRegression(
        featuresCol="features", labelCol="total_amount",
        maxIter=100, regParam=0.1, elasticNetParam=0.0,
    ),
    "gbt": GBTRegressor(
        featuresCol="features", labelCol="total_amount",
        maxIter=50, maxDepth=5, maxBins=300, seed=42,
    ),
    "rf": RandomForestRegressor(
        featuresCol="features", labelCol="total_amount",
        numTrees=100, maxDepth=8, maxBins=300, seed=42,
    ),
}

metrics = {}
print("\n── Model Comparison ─────────────────────────────────────────────────────")
print(f"{'Model':<30} {'RMSE':>10} {'R²':>10}")
print("-" * 52)
for name, regressor in MODELS.items():
    pipeline = build_pipeline(regressor)
    model = pipeline.fit(train_df)
    preds = model.transform(test_df)
    rmse = evaluator_rmse.evaluate(preds)
    r2 = evaluator_r2.evaluate(preds)
    metrics[name] = {"rmse": rmse, "r2": r2}
    print(f"{name:<30} ${rmse:>9.2f} {r2:>10.4f}")
    model.write().overwrite().save(OUTPUT_PATH + f"models/{name}/")

best = min(metrics, key=lambda k: metrics[k]["rmse"])
improvement = (metrics["lr"]["rmse"] - metrics[best]["rmse"]) / metrics["lr"]["rmse"] * 100
print(f"\nBest model: {best} — reduced RMSE by {improvement:.1f}% over LR baseline.")

metrics_path = OUTPUT_PATH + "models/metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nModels saved to {OUTPUT_PATH}models/, metrics written to {metrics_path}")
spark.stop()
```

- [ ] **Step 2: Run training**

```bash
make run-ml
```
Expected: comparison table prints for `lr`, `gbt`, `rf` with RMSE/R² values; final line `Models saved to /home/jovyan/data/output/models/, metrics written to /home/jovyan/data/output/models/metrics.json`.

- [ ] **Step 3: Verify the saved models are full pipelines, not bare regressors**

```bash
ls data/output/models/gbt/
cat data/output/models/metrics.json
```
Expected: `data/output/models/gbt/` now contains a `stages/` subdirectory (the `PipelineModel` save format — distinct from the old bare-regressor format, which had only `data/`/`metadata/`/`treesMetadata/` at the top level). `metrics.json` has three top-level keys (`lr`, `gbt`, `rf`), each with numeric `rmse` and `r2`.

- [ ] **Step 4: Commit**

```bash
git add work/04_ml.py
git commit -m "feat: train full inference-ready PipelineModels, write metrics.json"
```

---

### Task 3: Download NYC TLC zone lookup CSV

**Files:**
- Create: `scripts/download_zone_lookup.sh`
- Modify: `Makefile`

**Interfaces:**
- Produces: `data/taxi_zone_lookup.csv` with columns `LocationID, Borough, Zone, service_zone` — consumed by Task 5's `zone_dropdown_options()` and Task 6's Streamlit app.

- [ ] **Step 1: Create `scripts/download_zone_lookup.sh`**

```bash
#!/bin/bash
# scripts/download_zone_lookup.sh
# Downloads the NYC TLC taxi zone lookup table (LocationID -> Borough/Zone name).
set -e

mkdir -p data
curl -L "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv" \
     -o data/taxi_zone_lookup.csv

echo "Done. Saved to data/taxi_zone_lookup.csv"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/download_zone_lookup.sh
```

- [ ] **Step 3: Add a `download-zones` target to `Makefile`**

Add below the existing `download-data` target in `Makefile`:

```makefile
download-zones:
	./scripts/download_zone_lookup.sh
```

Also update the `.PHONY` line at the top of `Makefile` to include `download-zones`:

```makefile
.PHONY: up down shell test run-explore run-preprocess run-analytics run-ml download-data download-zones run-ui
```

- [ ] **Step 4: Run it and verify**

```bash
make download-zones
head -3 data/taxi_zone_lookup.csv
```
Expected: first line is the header `"LocationID","Borough","Zone","service_zone"`, followed by real zone rows (e.g. `1,"EWR","Newark Airport","EWR"`).

- [ ] **Step 5: Commit**

```bash
git add scripts/download_zone_lookup.sh Makefile
git commit -m "feat: add TLC zone lookup download script"
```

(`data/taxi_zone_lookup.csv` itself is gitignored via the existing `data/` rule — do not force-add it.)

---

### Task 4: Pure helper functions for the UI (TDD)

**Files:**
- Create: `work/predict_helpers.py`
- Test: `tests/test_predict_helpers.py`

**Interfaces:**
- Produces:
  - `DAY_NAME_TO_SPARK_DOW: dict[str, int]` — `{"Sunday": 1, ..., "Saturday": 7}`
  - `payment_flags(method: str) -> dict` — `{"pay_credit_card": int, "pay_cash": int}`, raises `ValueError` for unknown input
  - `build_input_row(pu_location_id: int, do_location_id: int, trip_distance: float, pickup_hour: int, day_name: str, passenger_count: int, payment_method: str) -> dict` — the raw feature dict a `PipelineModel` expects
  - `load_best_model_name(metrics: dict) -> str` — picks the key with the lowest `"rmse"`
  - `zone_dropdown_options(zone_df: pandas.DataFrame) -> list[tuple[str, int]]` — `[(label, LocationID), ...]` sorted by label, `zone_df` expected to have `LocationID`, `Borough`, `Zone` columns
- Consumed by: Task 5's `ui/predict_app.py`

No Spark session is required for any of these — they're plain Python/pandas, so `tests/test_predict_helpers.py` does not use the `spark` fixture.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_predict_helpers.py
import pytest
import pandas as pd


class TestPaymentFlags:
    def test_credit_card(self):
        from work.predict_helpers import payment_flags
        assert payment_flags("Credit Card") == {"pay_credit_card": 1, "pay_cash": 0}

    def test_cash(self):
        from work.predict_helpers import payment_flags
        assert payment_flags("Cash") == {"pay_credit_card": 0, "pay_cash": 1}

    def test_unknown_method_raises(self):
        from work.predict_helpers import payment_flags
        with pytest.raises(ValueError):
            payment_flags("Bitcoin")


class TestBuildInputRow:
    def test_maps_all_fields(self):
        from work.predict_helpers import build_input_row
        row = build_input_row(
            pu_location_id=161,
            do_location_id=236,
            trip_distance=2.5,
            pickup_hour=10,
            day_name="Monday",
            passenger_count=1,
            payment_method="Credit Card",
        )
        assert row == {
            "PULocationID": 161,
            "DOLocationID": 236,
            "trip_distance": 2.5,
            "pickup_hour": 10,
            "day_of_week": 2,
            "passenger_count": 1,
            "pay_credit_card": 1,
            "pay_cash": 0,
        }


class TestLoadBestModelName:
    def test_picks_lowest_rmse(self):
        from work.predict_helpers import load_best_model_name
        metrics = {
            "lr": {"rmse": 4.82, "r2": 0.91},
            "gbt": {"rmse": 2.41, "r2": 0.96},
            "rf": {"rmse": 2.55, "r2": 0.96},
        }
        assert load_best_model_name(metrics) == "gbt"


class TestZoneDropdownOptions:
    def test_formats_and_sorts_labels(self):
        from work.predict_helpers import zone_dropdown_options
        df = pd.DataFrame([
            {"LocationID": 236, "Borough": "Manhattan", "Zone": "Upper East Side North"},
            {"LocationID": 132, "Borough": "Queens", "Zone": "JFK Airport"},
        ])
        options = zone_dropdown_options(df)
        assert options == [
            ("JFK Airport, Queens", 132),
            ("Upper East Side North, Manhattan", 236),
        ]
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `docker compose exec pyspark bash -c "cd /home/jovyan && python -m pytest tests/test_predict_helpers.py -v"`
Expected: `ModuleNotFoundError: No module named 'work.predict_helpers'`

- [ ] **Step 3: Implement `work/predict_helpers.py`**

```python
# work/predict_helpers.py
"""Pure helper functions bridging the Streamlit form to the trained PipelineModel's
expected raw input columns. Kept free of Spark/Streamlit imports so they're fast
to unit test."""

DAY_NAME_TO_SPARK_DOW = {
    "Sunday": 1,
    "Monday": 2,
    "Tuesday": 3,
    "Wednesday": 4,
    "Thursday": 5,
    "Friday": 6,
    "Saturday": 7,
}


def payment_flags(method: str) -> dict:
    if method == "Credit Card":
        return {"pay_credit_card": 1, "pay_cash": 0}
    if method == "Cash":
        return {"pay_credit_card": 0, "pay_cash": 1}
    raise ValueError(f"Unknown payment method: {method}")


def build_input_row(
    pu_location_id: int,
    do_location_id: int,
    trip_distance: float,
    pickup_hour: int,
    day_name: str,
    passenger_count: int,
    payment_method: str,
) -> dict:
    row = {
        "PULocationID": pu_location_id,
        "DOLocationID": do_location_id,
        "trip_distance": trip_distance,
        "pickup_hour": pickup_hour,
        "day_of_week": DAY_NAME_TO_SPARK_DOW[day_name],
        "passenger_count": passenger_count,
    }
    row.update(payment_flags(payment_method))
    return row


def load_best_model_name(metrics: dict) -> str:
    return min(metrics, key=lambda name: metrics[name]["rmse"])


def zone_dropdown_options(zone_df) -> list:
    options = [
        (f"{row.Zone}, {row.Borough}", int(row.LocationID))
        for row in zone_df.itertuples()
    ]
    return sorted(options, key=lambda pair: pair[0])
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `docker compose exec pyspark bash -c "cd /home/jovyan && python -m pytest tests/test_predict_helpers.py -v"`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add work/predict_helpers.py tests/test_predict_helpers.py
git commit -m "feat: pure helper functions for prediction UI form mapping"
```

---

### Task 5: Streamlit prediction app

**Files:**
- Create: `ui/predict_app.py`

**Interfaces:**
- Consumes: `get_spark()` (`work/spark_helper.py`), `OUTPUT_PATH` (`work/constants.py`), `build_input_row`, `load_best_model_name`, `zone_dropdown_options`, `DAY_NAME_TO_SPARK_DOW` (`work/predict_helpers.py`, Task 4), `data/output/models/{name}/` + `metrics.json` (Task 2), `data/taxi_zone_lookup.csv` (Task 3)

This file is UI wiring, not unit-tested directly — Task 6 covers running and manually verifying it end-to-end.

- [ ] **Step 1: Create `ui/predict_app.py`**

```python
# ui/predict_app.py
import json

import pandas as pd
import streamlit as st
from pyspark.ml import PipelineModel
from pyspark.sql import Row

from work.spark_helper import get_spark
from work.constants import OUTPUT_PATH
from work.predict_helpers import (
    DAY_NAME_TO_SPARK_DOW,
    build_input_row,
    load_best_model_name,
    zone_dropdown_options,
)

st.set_page_config(page_title="Taxi Fare Predictor", page_icon="🚕")
st.title("🚕 NYC Yellow Taxi — Predicted Total Amount")


@st.cache_resource
def load_model_and_metrics():
    metrics_path = OUTPUT_PATH + "models/metrics.json"
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
    except FileNotFoundError:
        return None, None, None

    best_name = load_best_model_name(metrics)
    spark = get_spark("predict_ui")
    model = PipelineModel.load(OUTPUT_PATH + f"models/{best_name}/")
    best_info = {"name": best_name, **metrics[best_name]}
    return spark, model, best_info


@st.cache_resource
def load_zone_options():
    zones = pd.read_csv("data/taxi_zone_lookup.csv")
    return zone_dropdown_options(zones)


spark, model, best_info = load_model_and_metrics()

if model is None:
    st.error(
        "No trained models found at `data/output/models/`. "
        "Run `make run-ml` first, then reload this page."
    )
    st.stop()

zone_options = load_zone_options()
zone_labels = [label for label, _ in zone_options]
zone_id_by_label = dict(zone_options)

with st.form("predict_form"):
    pu_label = st.selectbox("Pickup zone", zone_labels)
    do_label = st.selectbox("Dropoff zone", zone_labels)
    trip_distance = st.number_input(
        "Trip distance (miles)", min_value=0.1, value=2.5, step=0.1
    )
    pickup_hour = st.slider("Pickup hour", 0, 23, 10)
    day_name = st.selectbox("Day of week", list(DAY_NAME_TO_SPARK_DOW.keys()))
    passenger_count = st.number_input(
        "Passenger count", min_value=1, max_value=6, value=1, step=1
    )
    payment_method = st.radio("Payment method", ["Credit Card", "Cash"])
    submitted = st.form_submit_button("Predict")

if submitted:
    row = build_input_row(
        pu_location_id=zone_id_by_label[pu_label],
        do_location_id=zone_id_by_label[do_label],
        trip_distance=trip_distance,
        pickup_hour=pickup_hour,
        day_name=day_name,
        passenger_count=int(passenger_count),
        payment_method=payment_method,
    )
    input_df = spark.createDataFrame([Row(**row)])
    prediction = model.transform(input_df).first()["prediction"]

    st.metric("Predicted total amount", f"${prediction:.2f}")
    st.caption(
        f"Served by {best_info['name'].upper()} — "
        f"RMSE ${best_info['rmse']:.2f}, R² {best_info['r2']:.4f}"
    )
```

- [ ] **Step 2: Commit**

```bash
git add ui/predict_app.py
git commit -m "feat: Streamlit prediction UI"
```

(Verification that this actually runs happens in Task 6, once `streamlit` is installed in the container.)

---

### Task 6: Wire up Docker, Makefile, dependencies, and README

**Files:**
- Modify: `pyproject.toml`
- Modify: `docker-compose.yml`
- Modify: `Makefile`
- Modify: `README.md`

**Interfaces:**
- Consumes: `ui/predict_app.py` (Task 5)
- Produces: `make run-ui` — the command a user runs to start the app; port `8501` reachable at `http://localhost:8501`

- [ ] **Step 1: Add `streamlit` and `pandas` to `pyproject.toml`**

```toml
[project]
name = "cs675-final"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pyspark==3.5.0",
    "pytest==8.2.0",
    "streamlit==1.37.0",
    "pandas==2.2.2",
]
```

- [ ] **Step 2: Expose port 8501 and install streamlit in `docker-compose.yml`**

Modify the `pyspark` service:

```yaml
services:
  pyspark:
    image: jupyter/pyspark-notebook:spark-3.5.0
    volumes:
      - ./work:/home/jovyan/work
      - ./tests:/home/jovyan/tests
      - ./data:/home/jovyan/data
      - ./ui:/home/jovyan/ui
      - ./spark-events:/tmp/spark-events
    ports:
      - "4040:4040"
      - "18080:18080"
      - "8501:8501"
    environment:
      - SPARK_MASTER=local[*]
      - JUPYTER_ENABLE_LAB=yes
      - PYTHONPATH=/home/jovyan:/usr/local/spark/python:/usr/local/spark/python/lib/py4j-0.10.9.7-src.zip
    command: bash -c "pip install -q pytest==8.2.0 streamlit==1.37.0 && start.sh jupyter lab --no-browser"
```

(Only the `volumes`, `ports`, and `command` lines change — add the `./ui` volume mount, add the `8501:8501` port, add `streamlit==1.37.0` to the `pip install` in `command`. `pandas` ships already in the `jupyter/pyspark-notebook` base image, so no install step is needed for it.)

- [ ] **Step 3: Add `run-ui` target to `Makefile`**

```makefile
run-ui:
	docker compose exec pyspark bash -c "cd /home/jovyan && streamlit run ui/predict_app.py --server.port 8501 --server.address 0.0.0.0"
```

Update `.PHONY` to include `run-ui` (already added alongside `download-zones` in Task 3 — verify both are present).

- [ ] **Step 4: Restart the container so the new volume mount and port take effect**

```bash
docker compose up -d --force-recreate pyspark
```
Expected: `docker compose ps` shows `pyspark` as `Up`.

- [ ] **Step 5: Run the UI and verify manually**

```bash
make run-ui
```
Then open `http://localhost:8501` in a browser:
- Confirm the form renders with pickup/dropoff zone dropdowns, distance, hour, day, passenger count, and payment method fields.
- Fill in values and click "Predict".
- Confirm a `$X.XX` prediction renders along with a caption naming the serving model and its RMSE/R².
- Stop the app with `Ctrl+C` once verified.

- [ ] **Step 6: Add a "Run the prediction UI" section to `README.md`**

Add after the existing "Run Locally" section:

```markdown
## Run the Prediction UI

Prerequisites: models must already be trained (`make run-ml`) and the zone lookup downloaded (`make download-zones`).

```bash
make download-zones
make run-ui
```

Open http://localhost:8501, fill in trip details, and click "Predict" to see the estimated `total_amount` from the best-performing trained model.
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml docker-compose.yml Makefile README.md
git commit -m "feat: wire up Streamlit prediction UI in Docker + Makefile"
```

---

## Self-Review Against Spec

| Spec requirement | Covered by |
|---|---|
| Fix: full pipeline (indexers + assembler + regressor) saved as one `PipelineModel` | Task 1 (`build_pipeline`), Task 2 (retrain + save) |
| `metrics.json` with RMSE/R² per model | Task 2 |
| Zone lookup CSV for dropdowns | Task 3 |
| Payment method as single mutually-exclusive choice | Task 4 (`payment_flags`), Task 5 (`st.radio`) |
| Day-of-week UI mapped to Spark's 1–7 convention | Task 4 (`DAY_NAME_TO_SPARK_DOW`), Task 5 |
| Best-model selection by lowest RMSE | Task 4 (`load_best_model_name`), Task 5 |
| Streamlit form with all specified fields | Task 5 |
| Cached Spark session + model load (`@st.cache_resource`) | Task 5 |
| Missing-model error handling | Task 5 (`st.error` + `st.stop()`) |
| Pure helper functions unit-tested without Spark | Task 4 |
| `build_pipeline()` unit tests, including unseen zone handling | Task 1 |
| Local-only deployment, Docker/Makefile wiring | Task 6 |
| README run instructions | Task 6 |
| `build_features()` left unchanged | Task 1 (explicit constraint, verified by existing tests still passing) |
