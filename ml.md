# ML Revenue Prediction — Implementation Plan

**Goal:** Train a LinearRegression baseline and a GBTRegressor to predict `total_amount` per trip from preprocessed Yellow Taxi data, compare RMSE and R² between the two models.

**Architecture:** Reads from the already-preprocessed `taxi_clean/` Parquet produced by `work/02_preprocess.py`. A `build_features()` function runs StringIndexer on zone IDs and assembles all features into a single vector. Both models are trained locally on the Jan 2022 slice and saved to disk. The same scripts run on cloud against the full 260M-row dataset with `s3://` paths.

**Tech Stack:** PySpark 3.5 MLlib (StringIndexer, VectorAssembler, LinearRegression, GBTRegressor, RegressionEvaluator), pytest, Docker (local), AWS EMR Serverless (cloud)

---

## File Map

| File | Responsibility |
|------|----------------|
| `work/ml_features.py` | `build_features(df)` — StringIndexer on zone IDs + VectorAssembler → `features` column |
| `work/04_ml.py` | Load clean data, train LR + GBT, evaluate, save models, print comparison table |
| `tests/test_ml.py` | Unit tests: feature vector shape, no nulls, target column present, model trains on tiny data |
| `cloud/04_ml_cloud.py` | Same as `04_ml.py` with S3 input/output paths, no local Spark config |

---

## Task 1: Feature Assembly (TDD)

**Files:**
- Create: `work/ml_features.py`
- Create: `tests/test_ml.py`

- [ ] **Step 1: Write failing tests in `tests/test_ml.py`**

```python
# tests/test_ml.py
import pytest
import datetime
from pyspark.sql import Row


def make_clean_rows(spark, n=3):
    """Create n rows of preprocessed taxi data with all columns needed for ML."""
    rows = [
        Row(
            pickup_date=datetime.date(2022, 1, 15),
            pickup_hour=10,
            day_of_week=7,
            time_of_day="morning",
            PULocationID=161,
            DOLocationID=236,
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
        )
        for _ in range(n)
    ]
    return spark.createDataFrame(rows)


class TestBuildFeatures:
    def test_features_column_exists(self, spark):
        from work.ml_features import build_features
        result = build_features(make_clean_rows(spark))
        assert "features" in result.columns

    def test_features_vector_size_is_8(self, spark):
        from work.ml_features import build_features
        from pyspark.ml.linalg import Vector
        result = build_features(make_clean_rows(spark))
        vec = result.first()["features"]
        assert vec.size == 8

    def test_no_null_features(self, spark):
        from work.ml_features import build_features
        from pyspark.sql.functions import col
        result = build_features(make_clean_rows(spark))
        null_count = result.filter(col("features").isNull()).count()
        assert null_count == 0

    def test_target_column_present(self, spark):
        from work.ml_features import build_features
        result = build_features(make_clean_rows(spark))
        assert "total_amount" in result.columns

    def test_pu_idx_and_do_idx_columns_created(self, spark):
        from work.ml_features import build_features
        result = build_features(make_clean_rows(spark))
        assert "pu_idx" in result.columns
        assert "do_idx" in result.columns
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
make test
```
Expected: `ImportError: cannot import name 'build_features' from 'work.ml_features'`

- [ ] **Step 3: Create `work/ml_features.py`**

```python
# work/ml_features.py
from pyspark.sql import DataFrame
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline


# The 8 features fed to both models.
# Order matters — must stay consistent between local and cloud runs.
FEATURE_COLS = [
    "trip_distance",
    "pickup_hour",
    "day_of_week",
    "passenger_count",
    "pu_idx",       # StringIndexer output for PULocationID
    "do_idx",       # StringIndexer output for DOLocationID
    "pay_credit_card",
    "pay_cash",
]


def build_features(df: DataFrame) -> DataFrame:
    """
    Transforms preprocessed taxi DataFrame into ML-ready form:
      1. StringIndexer: PULocationID → pu_idx, DOLocationID → do_idx
      2. VectorAssembler: FEATURE_COLS → features vector

    Returns the DataFrame with added columns: pu_idx, do_idx, features.
    The target column (total_amount) is left untouched.
    """
    pu_indexer = StringIndexer(
        inputCol="PULocationID",
        outputCol="pu_idx",
        handleInvalid="keep",   # unseen zones at inference time get index 0
    )
    do_indexer = StringIndexer(
        inputCol="DOLocationID",
        outputCol="do_idx",
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="features",
        handleInvalid="skip",   # drop rows where any feature is null
    )
    pipeline = Pipeline(stages=[pu_indexer, do_indexer, assembler])
    return pipeline.fit(df).transform(df)
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
make test
```
Expected: `5 passed` (new ml tests) plus all prior tests still green.

- [ ] **Step 5: Commit**

```bash
git add work/ml_features.py tests/test_ml.py
git commit -m "feat: ML feature assembly with unit tests (StringIndexer + VectorAssembler)"
```

---

## Task 2: Train, Evaluate, and Save Models Locally

**Files:**
- Create: `work/04_ml.py`
- Modify: `Makefile` (add `run-ml` target)

- [ ] **Step 1: Add `run-ml` to `Makefile`**

Open `Makefile` and add this target after `run-analytics`:

```makefile
run-ml:
	docker compose exec pyspark bash -c "cd /home/jovyan && python work/04_ml.py"
```

- [ ] **Step 2: Create `work/04_ml.py`**

```python
# work/04_ml.py
from work.spark_helper import get_spark
from work.constants import OUTPUT_PATH
from work.ml_features import build_features

from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = get_spark("04_ml")

# ── Load preprocessed data ──────────────────────────────────────────────────────
print("Loading clean taxi data...")
df = spark.read.parquet(OUTPUT_PATH + "taxi_clean/")
print(f"  Rows: {df.count():,}")

# ── Feature assembly ────────────────────────────────────────────────────────────
print("Building feature vectors...")
featured = build_features(df)
print(f"  Rows after feature assembly: {featured.count():,}")

# ── Train / test split ──────────────────────────────────────────────────────────
train_df, test_df = featured.randomSplit([0.8, 0.2], seed=42)
print(f"  Train: {train_df.count():,} | Test: {test_df.count():,}")

evaluator_rmse = RegressionEvaluator(
    labelCol="total_amount", predictionCol="prediction", metricName="rmse"
)
evaluator_r2 = RegressionEvaluator(
    labelCol="total_amount", predictionCol="prediction", metricName="r2"
)

# ── Baseline: Linear Regression ─────────────────────────────────────────────────
print("\nTraining Linear Regression (baseline)...")
lr = LinearRegression(
    featuresCol="features",
    labelCol="total_amount",
    maxIter=100,
    regParam=0.1,
    elasticNetParam=0.0,
)
lr_model = lr.fit(train_df)
lr_preds = lr_model.transform(test_df)
lr_rmse = evaluator_rmse.evaluate(lr_preds)
lr_r2   = evaluator_r2.evaluate(lr_preds)
print(f"  LR  — RMSE: ${lr_rmse:.2f} | R²: {lr_r2:.4f}")

# ── Main model: Gradient Boosted Trees ──────────────────────────────────────────
print("\nTraining Gradient Boosted Trees...")
gbt = GBTRegressor(
    featuresCol="features",
    labelCol="total_amount",
    maxIter=50,
    maxDepth=5,
    seed=42,
)
gbt_model = gbt.fit(train_df)
gbt_preds = gbt_model.transform(test_df)
gbt_rmse = evaluator_rmse.evaluate(gbt_preds)
gbt_r2   = evaluator_r2.evaluate(gbt_preds)
print(f"  GBT — RMSE: ${gbt_rmse:.2f} | R²: {gbt_r2:.4f}")

# ── Comparison table ─────────────────────────────────────────────────────────────
print("\n── Model Comparison ─────────────────────────────────────────────────────")
print(f"{'Model':<30} {'RMSE':>10} {'R²':>10}")
print("-" * 52)
print(f"{'Linear Regression (baseline)':<30} ${lr_rmse:>9.2f} {lr_r2:>10.4f}")
print(f"{'Gradient Boosted Trees':<30} ${gbt_rmse:>9.2f} {gbt_r2:>10.4f}")
improvement = (lr_rmse - gbt_rmse) / lr_rmse * 100
print(f"\nGBT reduced RMSE by {improvement:.1f}% over baseline.")

# ── Save models ──────────────────────────────────────────────────────────────────
lr_model.write().overwrite().save(OUTPUT_PATH + "models/lr/")
gbt_model.write().overwrite().save(OUTPUT_PATH + "models/gbt/")
print(f"\nModels saved to {OUTPUT_PATH}models/")

spark.stop()
```

- [ ] **Step 3: Run ML training locally**

```bash
make run-ml
```
Expected output (approximate — your numbers will vary by slice):
```
── Model Comparison ─────────────────────────────────────────────────────
Model                          RMSE         R²
----------------------------------------------------
Linear Regression (baseline)   $  4.82     0.9123
Gradient Boosted Trees         $  2.41     0.9654

GBT reduced RMSE by 50.0% over baseline.

Models saved to /home/jovyan/data/output/models/
```
Record the actual RMSE and R² values — you'll need them for the README and demo.

- [ ] **Step 4: Commit**

```bash
git add work/04_ml.py Makefile
git commit -m "feat: train LR baseline + GBT model to predict total_amount, save models"
```

---

## Task 3: Cloud ML Job

**Files:**
- Create: `cloud/04_ml_cloud.py`

- [ ] **Step 1: Create `cloud/04_ml_cloud.py`**

```python
# cloud/04_ml_cloud.py
# Same logic as work/04_ml.py — only difference is S3 paths and no local Spark config.
import os, sys
sys.path.insert(0, "/home/hadoop/")

from pyspark.sql import SparkSession
from ml_features import build_features
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("cs675-ml-cloud").getOrCreate()

BUCKET = os.environ["CS675_BUCKET"]
IN     = f"s3://{BUCKET}/output/taxi_clean/"
OUT    = f"s3://{BUCKET}/output/"

print("Loading clean taxi data from S3...")
df = spark.read.parquet(IN)
print(f"  Rows: {df.count():,}")

featured = build_features(df)
train_df, test_df = featured.randomSplit([0.8, 0.2], seed=42)
print(f"  Train: {train_df.count():,} | Test: {test_df.count():,}")

evaluator_rmse = RegressionEvaluator(
    labelCol="total_amount", predictionCol="prediction", metricName="rmse"
)
evaluator_r2 = RegressionEvaluator(
    labelCol="total_amount", predictionCol="prediction", metricName="r2"
)

print("Training Linear Regression...")
lr_model = LinearRegression(
    featuresCol="features", labelCol="total_amount",
    maxIter=100, regParam=0.1, elasticNetParam=0.0,
).fit(train_df)
lr_preds = lr_model.transform(test_df)
lr_rmse = evaluator_rmse.evaluate(lr_preds)
lr_r2   = evaluator_r2.evaluate(lr_preds)
print(f"  LR  — RMSE: ${lr_rmse:.2f} | R²: {lr_r2:.4f}")

print("Training Gradient Boosted Trees...")
gbt_model = GBTRegressor(
    featuresCol="features", labelCol="total_amount",
    maxIter=50, maxDepth=5, seed=42,
).fit(train_df)
gbt_preds = gbt_model.transform(test_df)
gbt_rmse = evaluator_rmse.evaluate(gbt_preds)
gbt_r2   = evaluator_r2.evaluate(gbt_preds)
print(f"  GBT — RMSE: ${gbt_rmse:.2f} | R²: {gbt_r2:.4f}")

print(f"\n── Model Comparison ──────────────────────────────────────────────────")
print(f"{'Linear Regression':<30} RMSE=${lr_rmse:.2f}  R²={lr_r2:.4f}")
print(f"{'Gradient Boosted Trees':<30} RMSE=${gbt_rmse:.2f}  R²={gbt_r2:.4f}")

lr_model.write().overwrite().save(OUT + "models/lr/")
gbt_model.write().overwrite().save(OUT + "models/gbt/")
print(f"Models saved to {OUT}models/")

spark.stop()
```

- [ ] **Step 2: Submit to EMR Serverless**

```bash
export CS675_BUCKET=$(cd infrastructure && terraform output -raw bucket_name)
export EMR_ROLE_ARN="arn:aws:iam::ACCOUNT_ID:role/EMRServerlessExecutionRole"

# Upload the cloud script and ml_features helper
aws s3 cp cloud/04_ml_cloud.py      "s3://${CS675_BUCKET}/scripts/04_ml_cloud.py"  --profile ds
aws s3 cp work/ml_features.py       "s3://${CS675_BUCKET}/scripts/ml_features.py"  --profile ds

./cloud/emr_job_runner.sh 04_ml_cloud
```
Expected: `Done: 04_ml_cloud`. Models saved at `s3://bucket/output/models/lr/` and `s3://bucket/output/models/gbt/`.

- [ ] **Step 3: Commit**

```bash
git add cloud/04_ml_cloud.py
git commit -m "feat: cloud ML training job for LR + GBT on full 260M-row dataset"
```

---

## Self-Review Against Spec

| Spec requirement | Covered by |
|---|---|
| Target: `total_amount` | Task 2 — `labelCol="total_amount"` in both models |
| Features: trip_distance, pickup_hour, day_of_week, passenger_count | Task 1 — `FEATURE_COLS` in `ml_features.py` |
| Features: PULocationID, DOLocationID | Task 1 — StringIndexer → `pu_idx`, `do_idx` |
| Features: pay_credit_card, pay_cash | Task 1 — included in `FEATURE_COLS` |
| Baseline: LinearRegression | Task 2 — `lr = LinearRegression(...)` |
| Main model: GBTRegressor | Task 2 — `gbt = GBTRegressor(...)` |
| Evaluation: RMSE + R² | Task 2 — two `RegressionEvaluator` instances |
| Save models | Task 2 — `.write().overwrite().save(...)` |
| Cloud version | Task 3 — `cloud/04_ml_cloud.py` |
| Unit tests | Task 1 — 5 tests in `tests/test_ml.py` |
