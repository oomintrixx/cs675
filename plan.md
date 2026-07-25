# CS-675 Final Project: NYC Yellow Taxi Analytics at Cloud Scale — Implementation Plan

**Goal:** Build a PySpark analytics pipeline on NYC Yellow Taxi data — clean it locally with a monthly slice, then deploy to AWS at 100M+ row scale using S3 + Athena + EMR Serverless.

**Architecture:** Local development uses Docker PySpark on a single-month taxi slice (~3M rows). Cloud deployment uses Terraform to provision S3 + Athena Glue catalog + EMR Serverless, uploads 2019–2022 full Yellow Taxi data (~260M rows), and runs the same PySpark jobs pointing at `s3://` paths.

**Tech Stack:** PySpark 3.5, Docker (local), AWS S3 + Athena + EMR Serverless + Glue, Terraform 1.5+, Python 3.12, pytest

---

## Research Questions

1. How does trip demand vary by hour of day and day of week?
2. Which pickup zones generate the most revenue and trips?
3. Does tipping behavior differ across trip distance buckets (short / medium / long)?
4. How does fare-per-mile vary across distance buckets — do short trips cost more per mile?

---

## File Map

| File | Responsibility |
|------|----------------|
| `docker-compose.yml` | Local PySpark + Spark History Server |
| `Makefile` | Lifecycle: up/down/test/run local scripts |
| `pyproject.toml` | Python deps (pyspark, pytest) |
| `work/constants.py` | Data paths, auto-switch local vs cloud |
| `work/spark_helper.py` | `get_spark()` factory (local vs EMR) |
| `work/01_explore.py` | Profile raw taxi schema, row counts, null counts |
| `work/preprocess_steps.py` | Preprocessing functions (impute, outlier, normalize, encode, bin) |
| `work/02_preprocess.py` | Full preprocessing pipeline runner |
| `work/analytics.py` | 4 analytical query functions |
| `work/03_analytics.py` | Runs all 4 queries locally, writes results |
| `cloud/02_preprocess_cloud.py` | Same as 02 but reads/writes `s3://` paths |
| `cloud/03_analytics_cloud.py` | Same as 03 but reads/writes `s3://` paths |
| `cloud/emr_job_runner.sh` | Submits PySpark job to EMR Serverless, polls until done |
| `scripts/download_full_data.sh` | Downloads 2019–2022 Yellow Taxi parquet and uploads to S3 |
| `infrastructure/main.tf` | S3 bucket + Athena workgroup + Glue DB + EMR Serverless |
| `infrastructure/variables.tf` | `student_id`, `aws_region` |
| `infrastructure/outputs.tf` | Bucket name, Athena workgroup, EMR app ID |
| `infrastructure/glue_taxi.tf` | Glue table for Yellow Taxi Parquet partitioned by year/month |
| `tests/conftest.py` | Shared `spark` pytest fixture |
| `tests/test_preprocess.py` | Unit tests for all preprocessing transformations |
| `tests/test_analytics.py` | Unit tests for all 4 query functions |
| `README.md` | Run instructions, dataset description, results, preprocessing justification |

---

## Task 1: Docker Environment and Project Skeleton

**Files:**
- Create: `docker-compose.yml`
- Create: `Makefile`
- Create: `pyproject.toml`
- Create: `work/constants.py`
- Create: `work/spark_helper.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `docker-compose.yml`**

```yaml
# docker-compose.yml
version: "3.8"
services:
  pyspark:
    image: jupyter/pyspark-notebook:spark-3.5.0
    volumes:
      - ./work:/home/jovyan/work
      - ./tests:/home/jovyan/tests
      - ./data:/home/jovyan/data
    ports:
      - "4040:4040"
      - "18080:18080"
    environment:
      - SPARK_MASTER=local[*]
    command: start.sh jupyter lab --no-browser
  history:
    image: jupyter/pyspark-notebook:spark-3.5.0
    volumes:
      - ./spark-events:/tmp/spark-events
    ports:
      - "18081:18080"
    command: bash -c "/usr/local/spark/sbin/start-history-server.sh && tail -f /dev/null"
    environment:
      - SPARK_HISTORY_OPTS=-Dspark.history.fs.logDirectory=/tmp/spark-events
```

- [ ] **Step 2: Create `Makefile`**

```makefile
# Makefile
.PHONY: up down shell test run-explore run-preprocess run-analytics download-data

up:
	docker compose up -d

down:
	docker compose down

shell:
	docker compose exec pyspark bash

test:
	docker compose exec pyspark bash -c "cd /home/jovyan && python -m pytest tests/ -v"

run-explore:
	docker compose exec pyspark bash -c "cd /home/jovyan && python work/01_explore.py"

run-preprocess:
	docker compose exec pyspark bash -c "cd /home/jovyan && python work/02_preprocess.py"

run-analytics:
	docker compose exec pyspark bash -c "cd /home/jovyan && python work/03_analytics.py"

download-data:
	mkdir -p data/taxi
	curl -L "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-01.parquet" \
	     -o data/taxi/yellow_2022-01.parquet
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "cs675-final"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pyspark==3.5.0",
    "pytest==8.2.0",
]
```

- [ ] **Step 4: Create `work/constants.py`**

```python
# work/constants.py
import os

IS_CLOUD = os.environ.get("CS675_ENV") == "cloud"

LOCAL_TAXI_PATH   = "/home/jovyan/data/taxi/"
LOCAL_OUTPUT_PATH = "/home/jovyan/data/output/"

BUCKET            = os.environ.get("CS675_BUCKET", "ds-student-workspace")
CLOUD_TAXI_PATH   = f"s3://{BUCKET}/data/taxi/"
CLOUD_OUTPUT_PATH = f"s3://{BUCKET}/output/"

TAXI_PATH   = CLOUD_TAXI_PATH   if IS_CLOUD else LOCAL_TAXI_PATH
OUTPUT_PATH = CLOUD_OUTPUT_PATH if IS_CLOUD else LOCAL_OUTPUT_PATH
```

- [ ] **Step 5: Create `work/spark_helper.py`**

```python
# work/spark_helper.py
from pyspark.sql import SparkSession
import os

def get_spark(app_name: str = "cs675") -> SparkSession:
    """
    Local: sets master, event log, and shuffle partitions.
    Cloud (EMR Serverless): calls getOrCreate() only — EMR provides the session.
    """
    builder = SparkSession.builder.appName(app_name)
    if os.environ.get("CS675_ENV") != "cloud":
        builder = (
            builder
            .master("local[*]")
            .config("spark.eventLog.enabled", "true")
            .config("spark.eventLog.dir", "/tmp/spark-events")
            .config("spark.sql.shuffle.partitions", "8")
        )
    return builder.getOrCreate()
```

- [ ] **Step 6: Create `tests/conftest.py`**

```python
# tests/conftest.py
import pytest
from pyspark.sql import SparkSession

@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("cs675-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()
```

- [ ] **Step 7: Start Docker and verify**

```bash
make up
docker compose ps
```
Expected: `pyspark` service shows `Up`

- [ ] **Step 8: Download sample data**

```bash
make download-data
ls data/taxi/
```
Expected: `yellow_2022-01.parquet`

- [ ] **Step 9: Commit**

```bash
git add docker-compose.yml Makefile pyproject.toml work/constants.py work/spark_helper.py tests/conftest.py
git commit -m "feat: project skeleton with Docker, Spark helpers, constants"
```

---

## Task 2: Data Exploration

**Files:**
- Create: `work/01_explore.py`

- [ ] **Step 1: Write `work/01_explore.py`**

```python
# work/01_explore.py
from work.spark_helper import get_spark
from work.constants import TAXI_PATH
from pyspark.sql.functions import col, count, when, isnan

spark = get_spark("01_explore")

taxi = spark.read.parquet(TAXI_PATH)

print("Schema:")
taxi.printSchema()

print(f"\nRow count: {taxi.count():,}")

print("\nDescriptive stats:")
taxi.describe("trip_distance", "fare_amount", "tip_amount", "passenger_count").show()

print("\nNull counts:")
taxi.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c)
    for c in ["trip_distance", "fare_amount", "passenger_count",
              "tpep_pickup_datetime", "PULocationID"]
]).show()

print("\nSample rows:")
taxi.select(
    "tpep_pickup_datetime", "PULocationID", "DOLocationID",
    "trip_distance", "fare_amount", "tip_amount", "payment_type"
).show(10)

spark.stop()
```

- [ ] **Step 2: Run exploration**

```bash
make run-explore
```
Expected: schema printed, row count ~3M, null counts shown. Note actual null % for README.

- [ ] **Step 3: Commit**

```bash
git add work/01_explore.py
git commit -m "feat: data exploration script"
```

---

## Task 3: Preprocessing Pipeline (TDD)

**Files:**
- Create: `tests/test_preprocess.py`
- Create: `work/preprocess_steps.py`
- Create: `work/02_preprocess.py`

- [ ] **Step 1: Write failing tests in `tests/test_preprocess.py`**

```python
# tests/test_preprocess.py
import pytest
import datetime
from pyspark.sql import Row


def make_row(spark, **overrides):
    defaults = dict(
        tpep_pickup_datetime=datetime.datetime(2022, 1, 15, 10, 0, 0),
        tpep_dropoff_datetime=datetime.datetime(2022, 1, 15, 10, 30, 0),
        trip_distance=2.5,
        fare_amount=12.0,
        tip_amount=2.0,
        total_amount=15.0,
        passenger_count=1,
        payment_type=1,
        PULocationID=161,
        DOLocationID=236,
    )
    defaults.update(overrides)
    return spark.createDataFrame([Row(**defaults)])


class TestImputation:
    def test_null_passenger_count_filled_with_1(self, spark):
        from work.preprocess_steps import impute
        df = make_row(spark, passenger_count=None)
        assert impute(df).first()["passenger_count"] == 1

    def test_null_fare_amount_row_dropped(self, spark):
        from work.preprocess_steps import impute
        df = make_row(spark, fare_amount=None)
        assert impute(df).count() == 0


class TestOutlierRemoval:
    def test_negative_distance_dropped(self, spark):
        from work.preprocess_steps import remove_outliers
        df = make_row(spark, trip_distance=-1.0)
        assert remove_outliers(df).count() == 0

    def test_zero_distance_dropped(self, spark):
        from work.preprocess_steps import remove_outliers
        df = make_row(spark, trip_distance=0.0)
        assert remove_outliers(df).count() == 0

    def test_extreme_fare_capped_at_500(self, spark):
        from work.preprocess_steps import remove_outliers
        df = make_row(spark, fare_amount=9999.0)
        assert remove_outliers(df).first()["fare_amount"] == 500.0

    def test_valid_row_kept(self, spark):
        from work.preprocess_steps import remove_outliers
        df = make_row(spark)
        assert remove_outliers(df).count() == 1


class TestNormalization:
    def test_fare_normalized_to_0_and_1(self, spark):
        from work.preprocess_steps import normalize
        from functools import reduce
        df = reduce(lambda a, b: a.union(b), [
            make_row(spark, fare_amount=0.0),
            make_row(spark, fare_amount=100.0),
        ])
        norms = sorted(r["fare_norm"] for r in normalize(df).collect())
        assert norms[0] == pytest.approx(0.0)
        assert norms[1] == pytest.approx(1.0)


class TestEncoding:
    def test_credit_card_encoded(self, spark):
        from work.preprocess_steps import encode
        row = encode(make_row(spark, payment_type=1)).first()
        assert row["pay_credit_card"] == 1
        assert row["pay_cash"] == 0

    def test_cash_encoded(self, spark):
        from work.preprocess_steps import encode
        row = encode(make_row(spark, payment_type=2)).first()
        assert row["pay_credit_card"] == 0
        assert row["pay_cash"] == 1

    def test_unknown_payment_type_all_zero(self, spark):
        from work.preprocess_steps import encode
        row = encode(make_row(spark, payment_type=99)).first()
        assert row["pay_credit_card"] == 0
        assert row["pay_cash"] == 0


class TestBinning:
    def test_short_trip(self, spark):
        from work.preprocess_steps import bin_distance
        assert bin_distance(make_row(spark, trip_distance=0.5)).first()["distance_bucket"] == "short"

    def test_medium_trip(self, spark):
        from work.preprocess_steps import bin_distance
        assert bin_distance(make_row(spark, trip_distance=5.0)).first()["distance_bucket"] == "medium"

    def test_long_trip(self, spark):
        from work.preprocess_steps import bin_distance
        assert bin_distance(make_row(spark, trip_distance=15.0)).first()["distance_bucket"] == "long"

    def test_morning_time_of_day(self, spark):
        from work.preprocess_steps import bin_time_of_day
        import datetime
        df = make_row(spark, tpep_pickup_datetime=datetime.datetime(2022, 1, 15, 8, 0, 0))
        assert bin_time_of_day(df).first()["time_of_day"] == "morning"

    def test_night_time_of_day(self, spark):
        from work.preprocess_steps import bin_time_of_day
        import datetime
        df = make_row(spark, tpep_pickup_datetime=datetime.datetime(2022, 1, 15, 23, 0, 0))
        assert bin_time_of_day(df).first()["time_of_day"] == "night"
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
make test
```
Expected: `ImportError: cannot import name 'impute' from 'work.preprocess_steps'`

- [ ] **Step 3: Create `work/preprocess_steps.py`**

```python
# work/preprocess_steps.py
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, lit, hour, to_date,
    min as spark_min, max as spark_max, least,
)


def impute(df: DataFrame) -> DataFrame:
    """
    Fill passenger_count nulls with 1 (solo rider is the mode).
    Drop rows where fare_amount is null — it is the primary outcome variable.
    """
    return (
        df.fillna({"passenger_count": 1})
          .filter(col("fare_amount").isNotNull())
    )


def remove_outliers(df: DataFrame) -> DataFrame:
    """
    Drop trips with distance <= 0 (impossible) or negative fare.
    Cap fare at $500 — the highest legitimate fare (JFK flat rate ~$70; >$500 is a data error).
    Before: fare 0–$999,999. After: $0–$500.
    """
    return (
        df.filter(col("trip_distance") > 0)
          .filter(col("fare_amount") >= 0)
          .withColumn("fare_amount", least(col("fare_amount"), lit(500.0)))
    )


def normalize(df: DataFrame) -> DataFrame:
    """
    Min-max normalize fare_amount and trip_distance to [0, 1].
    Allows comparison across years and enables future ML use.
    """
    stats = df.agg(
        spark_min("fare_amount").alias("fare_min"),
        spark_max("fare_amount").alias("fare_max"),
        spark_min("trip_distance").alias("dist_min"),
        spark_max("trip_distance").alias("dist_max"),
    ).first()

    fare_range = float(stats["fare_max"] - stats["fare_min"]) or 1.0
    dist_range = float(stats["dist_max"] - stats["dist_min"]) or 1.0

    return (
        df.withColumn("fare_norm",
                      (col("fare_amount") - lit(float(stats["fare_min"]))) / lit(fare_range))
          .withColumn("dist_norm",
                      (col("trip_distance") - lit(float(stats["dist_min"]))) / lit(dist_range))
    )


def encode(df: DataFrame) -> DataFrame:
    """
    One-hot encode payment_type (1=credit card, 2=cash, 3=no charge).
    Unknown types produce all zeros. Converts categorical to numeric for analysis.
    """
    return (
        df.withColumn("pay_credit_card", when(col("payment_type") == 1, 1).otherwise(0))
          .withColumn("pay_cash",        when(col("payment_type") == 2, 1).otherwise(0))
          .withColumn("pay_no_charge",   when(col("payment_type") == 3, 1).otherwise(0))
    )


def bin_distance(df: DataFrame) -> DataFrame:
    """
    Bin trip_distance: short (<1mi), medium (1–10mi), long (>10mi).
    Thresholds align with TLC fare structure (base fare tiers).
    """
    return df.withColumn(
        "distance_bucket",
        when(col("trip_distance") < 1.0, "short")
        .when(col("trip_distance") <= 10.0, "medium")
        .otherwise("long")
    )


def bin_time_of_day(df: DataFrame) -> DataFrame:
    """
    Bin pickup hour into time-of-day segments:
      overnight (0–5), morning (6–11), afternoon (12–17), evening (18–21), night (22–23).
    Captures demand patterns that differ by commute vs leisure vs late-night.
    """
    h = hour(col("tpep_pickup_datetime"))
    return df.withColumn(
        "time_of_day",
        when(h < 6,  "overnight")
        .when(h < 12, "morning")
        .when(h < 18, "afternoon")
        .when(h < 22, "evening")
        .otherwise("night")
    )
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
make test
```
Expected: `14 passed`

- [ ] **Step 5: Write `work/02_preprocess.py`**

```python
# work/02_preprocess.py
from work.spark_helper import get_spark
from work.constants import TAXI_PATH, OUTPUT_PATH
from work.preprocess_steps import impute, remove_outliers, normalize, encode, bin_distance, bin_time_of_day
from pyspark.sql.functions import to_date, hour, dayofweek, col

spark = get_spark("02_preprocess")

print("Loading raw taxi data...")
raw = spark.read.parquet(TAXI_PATH)
print(f"  Raw rows: {raw.count():,}")

df = (
    raw
    .withColumn("pickup_date",  to_date(col("tpep_pickup_datetime")))
    .withColumn("pickup_hour",  hour(col("tpep_pickup_datetime")))
    .withColumn("day_of_week",  dayofweek(col("tpep_pickup_datetime")))
)
df = impute(df)
df = remove_outliers(df)
df = normalize(df)
df = encode(df)
df = bin_distance(df)
df = bin_time_of_day(df)

print(f"  After preprocessing: {df.count():,} rows")
df.select(
    "pickup_date", "pickup_hour", "day_of_week", "time_of_day",
    "PULocationID", "trip_distance", "distance_bucket",
    "fare_amount", "fare_norm", "pay_credit_card"
).show(5)

df.write.mode("overwrite").parquet(OUTPUT_PATH + "taxi_clean/")
print(f"Written to {OUTPUT_PATH}taxi_clean/")
spark.stop()
```

- [ ] **Step 6: Run preprocessing**

```bash
make run-preprocess
```
Expected: row counts before/after (expect 5–15% drop from outlier removal), parquet written.

- [ ] **Step 7: Commit**

```bash
git add work/preprocess_steps.py work/02_preprocess.py tests/test_preprocess.py
git commit -m "feat: preprocessing pipeline with unit tests (impute/outlier/normalize/encode/bin)"
```

---

## Task 4: Analytics Queries (TDD)

**Files:**
- Create: `tests/test_analytics.py`
- Create: `work/analytics.py`
- Create: `work/03_analytics.py`

- [ ] **Step 1: Write failing tests in `tests/test_analytics.py`**

```python
# tests/test_analytics.py
import datetime
import pytest
from pyspark.sql import Row


def make_clean_row(spark, **kw):
    defaults = dict(
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
        total_amount=15.0,
        passenger_count=1,
        payment_type=1,
        pay_credit_card=1,
        pay_cash=0,
        pay_no_charge=0,
        dist_norm=0.5,
    )
    defaults.update(kw)
    return spark.createDataFrame([Row(**defaults)])


class TestQuery1HourlyDemand:
    def test_output_has_required_columns(self, spark):
        from work.analytics import query_hourly_demand
        result = query_hourly_demand(make_clean_row(spark))
        assert "pickup_hour" in result.columns
        assert "day_of_week" in result.columns
        assert "trip_count" in result.columns

    def test_counts_correctly(self, spark):
        from work.analytics import query_hourly_demand
        result = query_hourly_demand(make_clean_row(spark))
        assert result.first()["trip_count"] == 1


class TestQuery2ZoneRevenue:
    def test_output_has_required_columns(self, spark):
        from work.analytics import query_zone_revenue
        result = query_zone_revenue(make_clean_row(spark))
        assert "PULocationID" in result.columns
        assert "total_revenue" in result.columns
        assert "trip_count" in result.columns

    def test_revenue_matches_fare(self, spark):
        from work.analytics import query_zone_revenue
        result = query_zone_revenue(make_clean_row(spark, fare_amount=20.0))
        assert result.first()["total_revenue"] == pytest.approx(20.0)


class TestQuery3TippingByDistance:
    def test_output_has_required_columns(self, spark):
        from work.analytics import query_tipping_by_distance
        result = query_tipping_by_distance(make_clean_row(spark))
        assert "distance_bucket" in result.columns
        assert "avg_tip_pct" in result.columns
        assert "trip_count" in result.columns

    def test_tip_pct_computed(self, spark):
        from work.analytics import query_tipping_by_distance
        # tip=2, fare=10 → tip_pct=20%
        result = query_tipping_by_distance(make_clean_row(spark, tip_amount=2.0, fare_amount=10.0))
        assert result.first()["avg_tip_pct"] == pytest.approx(20.0, abs=0.1)


class TestQuery4FarePerMile:
    def test_output_has_required_columns(self, spark):
        from work.analytics import query_fare_per_mile
        result = query_fare_per_mile(make_clean_row(spark))
        assert "distance_bucket" in result.columns
        assert "avg_fare_per_mile" in result.columns

    def test_fare_per_mile_computed(self, spark):
        from work.analytics import query_fare_per_mile
        # fare=10, distance=2 → fare_per_mile=5
        result = query_fare_per_mile(make_clean_row(spark, fare_amount=10.0, trip_distance=2.0))
        assert result.first()["avg_fare_per_mile"] == pytest.approx(5.0, abs=0.1)
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
make test
```
Expected: `ImportError: cannot import name 'query_hourly_demand' from 'work.analytics'`

- [ ] **Step 3: Create `work/analytics.py`**

```python
# work/analytics.py
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, avg, count, sum as spark_sum, round as spark_round


def query_hourly_demand(df: DataFrame) -> DataFrame:
    """
    Q1: Trip count by hour of day and day of week.
    Shows commute spikes (Mon–Fri 7–9am, 5–7pm) vs weekend leisure patterns.
    """
    return (
        df.groupBy("pickup_hour", "day_of_week")
          .agg(count("*").alias("trip_count"))
          .orderBy("day_of_week", "pickup_hour")
    )


def query_zone_revenue(df: DataFrame) -> DataFrame:
    """
    Q2: Total revenue and trip count per pickup zone (PULocationID).
    Identifies highest-value zones for drivers and fleet operators.
    """
    return (
        df.groupBy("PULocationID")
          .agg(
              spark_round(spark_sum("fare_amount"), 2).alias("total_revenue"),
              spark_round(avg("fare_amount"), 2).alias("avg_fare"),
              count("*").alias("trip_count"),
          )
          .orderBy("total_revenue", ascending=False)
    )


def query_tipping_by_distance(df: DataFrame) -> DataFrame:
    """
    Q3: Average tip percentage by distance bucket (short/medium/long).
    Tests whether longer trips earn proportionally more tips.
    tip_pct = (tip_amount / fare_amount) * 100
    """
    return (
        df.groupBy("distance_bucket")
          .agg(
              spark_round(avg(col("tip_amount") / col("fare_amount") * 100), 2).alias("avg_tip_pct"),
              spark_round(avg("tip_amount"), 2).alias("avg_tip_amount"),
              count("*").alias("trip_count"),
          )
          .orderBy("distance_bucket")
    )


def query_fare_per_mile(df: DataFrame) -> DataFrame:
    """
    Q4: Average fare per mile by distance bucket.
    Short trips should have higher fare/mile due to base fare dominating.
    """
    return (
        df.groupBy("distance_bucket")
          .agg(
              spark_round(avg(col("fare_amount") / col("trip_distance")), 2).alias("avg_fare_per_mile"),
              spark_round(avg("fare_amount"), 2).alias("avg_fare"),
              spark_round(avg("trip_distance"), 2).alias("avg_distance"),
              count("*").alias("trip_count"),
          )
          .orderBy("distance_bucket")
    )
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
make test
```
Expected: `18 passed`

- [ ] **Step 5: Write `work/03_analytics.py`**

```python
# work/03_analytics.py
from work.spark_helper import get_spark
from work.constants import OUTPUT_PATH
from work.analytics import (
    query_hourly_demand,
    query_zone_revenue,
    query_tipping_by_distance,
    query_fare_per_mile,
)

spark = get_spark("03_analytics")

print("Loading clean taxi data...")
df = spark.read.parquet(OUTPUT_PATH + "taxi_clean/")
print(f"  Rows: {df.count():,}")

print("\n── Q1: Trip Demand by Hour and Day of Week ───────────────────────────")
query_hourly_demand(df).show(48)

print("\n── Q2: Revenue by Pickup Zone (top 20) ───────────────────────────────")
query_zone_revenue(df).show(20)

print("\n── Q3: Tipping Behavior by Distance Bucket ───────────────────────────")
query_tipping_by_distance(df).show()

print("\n── Q4: Fare Per Mile by Distance Bucket ──────────────────────────────")
query_fare_per_mile(df).show()

# Persist results
query_hourly_demand(df).write.mode("overwrite").parquet(OUTPUT_PATH + "q1_hourly_demand/")
query_zone_revenue(df).write.mode("overwrite").parquet(OUTPUT_PATH + "q2_zone_revenue/")
query_tipping_by_distance(df).write.mode("overwrite").parquet(OUTPUT_PATH + "q3_tipping/")
query_fare_per_mile(df).write.mode("overwrite").parquet(OUTPUT_PATH + "q4_fare_per_mile/")

print("\nAll results written to", OUTPUT_PATH)
spark.stop()
```

- [ ] **Step 6: Run analytics**

```bash
make run-analytics
```
Expected: 4 result tables printed. Note findings for README.

- [ ] **Step 7: Commit**

```bash
git add work/analytics.py work/03_analytics.py tests/test_analytics.py
git commit -m "feat: 4 analytics queries with unit tests"
```

---

## Task 5: AWS Infrastructure with Terraform

**Files:**
- Create: `infrastructure/variables.tf`
- Create: `infrastructure/main.tf`
- Create: `infrastructure/glue_taxi.tf`
- Create: `infrastructure/outputs.tf`

> **Prerequisite:** AWS CLI installed, profile `ds` configured. Verify: `aws sts get-caller-identity --profile ds`

- [ ] **Step 1: Create `infrastructure/variables.tf`**

```hcl
# infrastructure/variables.tf
variable "student_id" {
  description = "Your student ID — used to name all AWS resources"
  type        = string
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}
```

- [ ] **Step 2: Create `infrastructure/main.tf`**

```hcl
# infrastructure/main.tf
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region  = var.aws_region
  profile = "ds"
}

locals {
  bucket_name      = "ds-${var.student_id}-workspace"
  athena_workgroup = "ds-${var.student_id}"
  glue_db          = "ds_${replace(var.student_id, "-", "_")}"
}

resource "aws_s3_bucket" "workspace" {
  bucket        = local.bucket_name
  force_destroy = true
  tags = { Owner = var.student_id, Project = "cs675" }
}

resource "aws_s3_bucket_versioning" "workspace" {
  bucket = aws_s3_bucket.workspace.id
  versioning_configuration { status = "Disabled" }
}

resource "aws_s3_object" "athena_results_prefix" {
  bucket  = aws_s3_bucket.workspace.id
  key     = "athena-results/.keep"
  content = ""
}

resource "aws_athena_workgroup" "main" {
  name = local.athena_workgroup
  configuration {
    result_configuration {
      output_location = "s3://${aws_s3_bucket.workspace.bucket}/athena-results/"
    }
    bytes_scanned_cutoff_per_query = 10737418240  # 10 GB scan cap
  }
  tags = { Owner = var.student_id }
}

resource "aws_glue_catalog_database" "main" {
  name = local.glue_db
}

resource "aws_emrserverless_application" "spark" {
  name          = "ds-${var.student_id}-spark"
  release_label = "emr-7.1.0"
  type          = "SPARK"

  maximum_capacity {
    cpu    = "8 vCPU"
    memory = "32 GB"
  }

  auto_stop_configuration {
    enabled              = true
    idle_timeout_minutes = 15
  }

  tags = { Owner = var.student_id }
}
```

- [ ] **Step 3: Create `infrastructure/glue_taxi.tf`**

```hcl
# infrastructure/glue_taxi.tf
resource "aws_glue_catalog_table" "yellow_taxi" {
  database_name = aws_glue_catalog_database.main.name
  name          = "yellow_taxi"
  table_type    = "EXTERNAL_TABLE"

  parameters = {
    "classification"            = "parquet"
    "parquet.compress"          = "SNAPPY"
    "projection.enabled"        = "true"
    "projection.year.type"      = "integer"
    "projection.year.range"     = "2019,2022"
    "projection.month.type"     = "integer"
    "projection.month.range"    = "1,12"
    "projection.month.digits"   = "2"
    "storage.location.template" = "s3://${aws_s3_bucket.workspace.bucket}/data/taxi/year=$${year}/month=$${month}/"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.workspace.bucket}/data/taxi/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"
    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }
    columns { name = "pickup_date";      type = "date"   }
    columns { name = "pickup_hour";      type = "int"    }
    columns { name = "day_of_week";      type = "int"    }
    columns { name = "time_of_day";      type = "string" }
    columns { name = "PULocationID";     type = "int"    }
    columns { name = "DOLocationID";     type = "int"    }
    columns { name = "trip_distance";    type = "double" }
    columns { name = "distance_bucket";  type = "string" }
    columns { name = "fare_amount";      type = "double" }
    columns { name = "fare_norm";        type = "double" }
    columns { name = "tip_amount";       type = "double" }
    columns { name = "pay_credit_card";  type = "int"    }
    columns { name = "pay_cash";         type = "int"    }
  }

  partition_keys { name = "year";  type = "int" }
  partition_keys { name = "month"; type = "int" }
}
```

- [ ] **Step 4: Create `infrastructure/outputs.tf`**

```hcl
# infrastructure/outputs.tf
output "bucket_name"      { value = aws_s3_bucket.workspace.bucket }
output "athena_workgroup" { value = aws_athena_workgroup.main.name }
output "glue_database"    { value = aws_glue_catalog_database.main.name }
output "emr_app_id"       { value = aws_emrserverless_application.spark.id }
```

- [ ] **Step 5: Add gitignore entries, create `terraform.tfvars` (do not commit)**

```bash
cat >> .gitignore << 'EOF'
infrastructure/terraform.tfvars
infrastructure/.terraform/
infrastructure/terraform.tfstate*
infrastructure/.terraform.lock.hcl
data/
spark-events/
EOF
```

Create `infrastructure/terraform.tfvars` manually (never commit):
```hcl
student_id = "your-student-id"
```

- [ ] **Step 6: Apply Terraform**

```bash
cd infrastructure
terraform init
terraform plan
terraform apply
```
Expected:
```
Apply complete! Resources: 7 added.
Outputs:
bucket_name      = "ds-your-id-workspace"
athena_workgroup = "ds-your-id"
glue_database    = "ds_your_id"
emr_app_id       = "00f..."
```

- [ ] **Step 7: Commit infrastructure**

```bash
git add infrastructure/*.tf .gitignore
git commit -m "feat: Terraform for S3 + Athena + Glue catalog + EMR Serverless"
```

---

## Task 6: Upload Full Dataset to S3

- [ ] **Step 1: Create `scripts/download_full_data.sh`**

```bash
#!/bin/bash
# scripts/download_full_data.sh
# Downloads Yellow Taxi Parquet (2019–2022, 48 files) and uploads to S3.
set -e

BUCKET="${CS675_BUCKET:?Set CS675_BUCKET env var}"
PROFILE="${AWS_PROFILE:-ds}"
mkdir -p /tmp/taxi_download

for YEAR in 2019 2020 2021 2022; do
  for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
    FILE="yellow_tripdata_${YEAR}-${MONTH}.parquet"
    URL="https://d37ci6vzurychx.cloudfront.net/trip-data/${FILE}"
    echo "Downloading $FILE..."
    curl -L "$URL" -o "/tmp/taxi_download/${FILE}"
    aws s3 cp "/tmp/taxi_download/${FILE}" \
      "s3://${BUCKET}/data/taxi/year=${YEAR}/month=${MONTH}/${FILE}" \
      --profile "${PROFILE}"
    rm "/tmp/taxi_download/${FILE}"
    echo "  Uploaded."
  done
done

echo "Done. 48 files uploaded."
```

- [ ] **Step 2: Run upload**

```bash
export CS675_BUCKET=$(cd infrastructure && terraform output -raw bucket_name)
chmod +x scripts/download_full_data.sh
./scripts/download_full_data.sh
```
Expected: 48 files in S3 under `data/taxi/year=*/month=*/`

- [ ] **Step 3: Verify**

```bash
aws s3 ls "s3://${CS675_BUCKET}/data/taxi/" --recursive --profile ds | wc -l
```
Expected: `48`

- [ ] **Step 4: Commit**

```bash
git add scripts/
git commit -m "feat: data upload script for full 2019-2022 Yellow Taxi to S3"
```

---

## Task 7: Cloud-Scale Execution on EMR Serverless

**Files:**
- Create: `cloud/02_preprocess_cloud.py`
- Create: `cloud/03_analytics_cloud.py`
- Create: `cloud/emr_job_runner.sh`

- [ ] **Step 1: Create `cloud/02_preprocess_cloud.py`**

```python
# cloud/02_preprocess_cloud.py
# Same logic as work/02_preprocess.py — S3 paths, no local Spark config.
import os, sys
sys.path.insert(0, "/home/hadoop/")

from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, hour, dayofweek, col
from preprocess_steps import impute, remove_outliers, normalize, encode, bin_distance, bin_time_of_day

spark = SparkSession.builder.appName("cs675-preprocess-cloud").getOrCreate()

BUCKET  = os.environ["CS675_BUCKET"]
TAXI_IN = f"s3://{BUCKET}/data/taxi/"
OUT     = f"s3://{BUCKET}/output/taxi_clean/"

print("Loading taxi from S3...")
raw = spark.read.parquet(TAXI_IN)
print(f"  Raw rows: {raw.count():,}")

df = (
    raw
    .withColumn("pickup_date", to_date(col("tpep_pickup_datetime")))
    .withColumn("pickup_hour", hour(col("tpep_pickup_datetime")))
    .withColumn("day_of_week", dayofweek(col("tpep_pickup_datetime")))
)
df = impute(df)
df = remove_outliers(df)
df = normalize(df)
df = encode(df)
df = bin_distance(df)
df = bin_time_of_day(df)

print(f"  Clean rows: {df.count():,}")
df.write.mode("overwrite").partitionBy("year", "month").parquet(OUT)
print("Cloud preprocessing complete.")
spark.stop()
```

- [ ] **Step 2: Create `cloud/03_analytics_cloud.py`**

```python
# cloud/03_analytics_cloud.py
import os, sys
sys.path.insert(0, "/home/hadoop/")

from pyspark.sql import SparkSession
from analytics import (
    query_hourly_demand, query_zone_revenue,
    query_tipping_by_distance, query_fare_per_mile,
)

spark = SparkSession.builder.appName("cs675-analytics-cloud").getOrCreate()

BUCKET = os.environ["CS675_BUCKET"]
IN     = f"s3://{BUCKET}/output/taxi_clean/"
OUT    = f"s3://{BUCKET}/output/results/"

print("Reading clean taxi data from S3...")
df = spark.read.parquet(IN)
print(f"  Rows: {df.count():,}")

query_hourly_demand(df).write.mode("overwrite").parquet(OUT + "q1/")
query_zone_revenue(df).write.mode("overwrite").parquet(OUT + "q2/")
query_tipping_by_distance(df).write.mode("overwrite").parquet(OUT + "q3/")
query_fare_per_mile(df).write.mode("overwrite").parquet(OUT + "q4/")

print("Cloud analytics complete. Results at:", OUT)
spark.stop()
```

- [ ] **Step 3: Create `cloud/emr_job_runner.sh`**

```bash
#!/bin/bash
# cloud/emr_job_runner.sh
set -e

BUCKET="${CS675_BUCKET:?}"
PROFILE="${AWS_PROFILE:-ds}"
APP_ID=$(cd infrastructure && terraform output -raw emr_app_id)
ROLE_ARN="${EMR_ROLE_ARN:?Set EMR_ROLE_ARN to the EMR execution role ARN}"
SCRIPT="${1:?Usage: $0 <script_name>  e.g. 02_preprocess_cloud}"

aws s3 cp "cloud/${SCRIPT}.py"        "s3://${BUCKET}/scripts/${SCRIPT}.py"        --profile "${PROFILE}"
aws s3 cp "work/preprocess_steps.py"  "s3://${BUCKET}/scripts/preprocess_steps.py" --profile "${PROFILE}"
aws s3 cp "work/analytics.py"         "s3://${BUCKET}/scripts/analytics.py"        --profile "${PROFILE}"

echo "Submitting ${SCRIPT} to EMR Serverless..."
RUN_ID=$(aws emr-serverless start-job-run \
  --application-id "${APP_ID}" \
  --execution-role-arn "${ROLE_ARN}" \
  --name "cs675-${SCRIPT}" \
  --job-driver "{
    \"sparkSubmit\": {
      \"entryPoint\": \"s3://${BUCKET}/scripts/${SCRIPT}.py\",
      \"sparkSubmitParameters\": \"--py-files s3://${BUCKET}/scripts/preprocess_steps.py,s3://${BUCKET}/scripts/analytics.py\"
    }
  }" \
  --configuration-overrides "{
    \"monitoringConfiguration\": {
      \"s3MonitoringConfiguration\": { \"logUri\": \"s3://${BUCKET}/emr-logs/\" }
    }
  }" \
  --environment "{\"CS675_BUCKET\": \"${BUCKET}\"}" \
  --profile "${PROFILE}" \
  --query "jobRunId" --output text)

echo "Job ID: ${RUN_ID}"
while true; do
  STATUS=$(aws emr-serverless get-job-run \
    --application-id "${APP_ID}" --job-run-id "${RUN_ID}" \
    --profile "${PROFILE}" --query "jobRun.state" --output text)
  echo "  Status: ${STATUS}"
  [[ "$STATUS" == "SUCCESS" || "$STATUS" == "FAILED" || "$STATUS" == "CANCELLED" ]] && break
  sleep 30
done

[[ "$STATUS" != "SUCCESS" ]] && { echo "FAILED. Logs: s3://${BUCKET}/emr-logs/"; exit 1; }
echo "Done: ${SCRIPT}"
```

- [ ] **Step 4: Run preprocessing at cloud scale**

```bash
export CS675_BUCKET=$(cd infrastructure && terraform output -raw bucket_name)
export EMR_ROLE_ARN="arn:aws:iam::ACCOUNT_ID:role/EMRServerlessExecutionRole"
chmod +x cloud/emr_job_runner.sh
./cloud/emr_job_runner.sh 02_preprocess_cloud
```
Expected: `Done: 02_preprocess_cloud`. Clean parquet written to `s3://bucket/output/taxi_clean/`.

- [ ] **Step 5: Run analytics at cloud scale**

```bash
./cloud/emr_job_runner.sh 03_analytics_cloud
```
Expected: `Done: 03_analytics_cloud`. Results at `s3://bucket/output/results/q1/` through `q4/`.

- [ ] **Step 6: Download and inspect results**

```bash
mkdir -p /tmp/results
aws s3 cp "s3://${CS675_BUCKET}/output/results/" /tmp/results/ --recursive --profile ds
docker compose exec pyspark bash -c "python -c \"
from pyspark.sql import SparkSession
spark = SparkSession.builder.master('local[*]').getOrCreate()
for q in ['q1','q2','q3','q4']:
    print(f'=== {q} ===')
    spark.read.parquet(f'/home/jovyan/data/results/{q}/').show()
\""
```
Record the actual numbers for the README and demo Q&A.

- [ ] **Step 7: Commit cloud scripts**

```bash
git add cloud/ scripts/
git commit -m "feat: cloud PySpark scripts + EMR Serverless job runner"
```

---

## Task 8: README, Results, and Cleanup

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# CS-675 Final Project: NYC Yellow Taxi Analytics at Cloud Scale

## Overview

Big-data analytics pipeline on NYC TLC Yellow Taxi data (2019–2022, ~260M trips).
Runs locally on a one-month slice for fast iteration, then deploys to AWS for full-scale results.

**Dataset:** NYC TLC Yellow Taxi 2019–2022 — public Parquet files from TLC open data portal.

## Research Questions & Results

| # | Question | Finding |
|---|----------|---------|
| Q1 | How does demand vary by hour and day? | [fill after cloud run] |
| Q2 | Which pickup zones generate the most revenue? | [fill after cloud run] |
| Q3 | Do longer trips earn higher tip percentages? | [fill after cloud run] |
| Q4 | Do short trips cost more per mile? | [fill after cloud run] |

## Tools Used

- **PySpark 3.5** — local preprocessing and analytics
- **AWS S3** — cloud storage for 260M-row dataset
- **AWS Athena** — interactive SQL over S3 Parquet
- **AWS EMR Serverless** — cloud-scale PySpark job execution
- **Terraform** — reproducible infrastructure provisioning

## Run Locally

Prerequisites: Docker Desktop

```bash
make up
make download-data   # downloads Jan 2022 sample (~3M rows)
make run-preprocess
make run-analytics
```

Spark UI: http://localhost:4040 | History: http://localhost:18081

Tests: `make test`

## Run on AWS

Prerequisites: AWS CLI + Terraform, profile `ds` configured.

```bash
export CS675_BUCKET=ds-<your-id>-workspace
export EMR_ROLE_ARN=arn:aws:iam::<account>:role/EMRServerlessExecutionRole

cd infrastructure && terraform init && terraform apply
cd ..
./scripts/download_full_data.sh          # ~30–60 min, uploads 48 files
./cloud/emr_job_runner.sh 02_preprocess_cloud
./cloud/emr_job_runner.sh 03_analytics_cloud

# Tear down to avoid charges
cd infrastructure && terraform destroy
```

## Preprocessing Justification

| Step | Before | After | Rationale |
|------|--------|-------|-----------|
| Impute `passenger_count` | X% null | 0% null | Fill with 1 — solo rider is the mode |
| Drop null `fare_amount` | X% null | dropped | Primary outcome; can't impute meaningfully |
| Remove distance ≤ 0 | X rows | removed | Physical impossibility — sensor error |
| Cap fare at $500 | max ~$999k | max $500 | Above $500 is a data entry error |
| Min-max normalize fare | 0–500 | 0–1 | Enables year-over-year comparison |
| One-hot encode `payment_type` | 1–6 int | binary flags | Converts categorical to numeric |
| Bin distance | 0–100 mi | short/medium/long | Aligns with TLC fare tier breakpoints |
| Bin time of day | 0–23 hour | overnight/morning/afternoon/evening/night | Captures demand pattern shifts |
```

- [ ] **Step 2: Commit README**

```bash
git add README.md
git commit -m "docs: README with results, run instructions, preprocessing justification"
```

- [ ] **Step 3: Tear down AWS resources**

```bash
cd infrastructure && terraform destroy
```
Expected: `Destroy complete! Resources: 7 destroyed.`

- [ ] **Step 4: Final test run**

```bash
make test
```
Expected: all tests pass.

---

## Spec Coverage Checklist

| Requirement | Task |
|---|---|
| Preprocessing: imputation | Task 3 — `impute()` |
| Preprocessing: outlier treatment | Task 3 — `remove_outliers()` |
| Preprocessing: normalization | Task 3 — `normalize()` |
| Preprocessing: encoding | Task 3 — `encode()` |
| Preprocessing: binning | Task 3 — `bin_distance()`, `bin_time_of_day()` |
| Before/after justification | Task 8 — README table |
| Multiple analytical queries | Task 4 — Q1–Q4 in `analytics.py` |
| Cloud infrastructure (reproducible) | Task 5 — Terraform |
| 100M+ rows at cloud scale | Tasks 6–7 — ~260M rows (2019–2022) |
| GitHub repo + README + run instructions | Task 8 |
| Evaluate and discuss results | Task 8 — README results table (fill after Task 7) |
| Two course tools | PySpark + AWS (Athena + EMR Serverless) |
| Tear down cloud resources | Task 8 Step 3 |

**Watch-outs:**
- Fill in actual result numbers in README after Task 7 Step 6 — the demo Q&A will probe these.
- `EMR_ROLE_ARN` is provisioned by the instructor (or your own AWS account), not by `main.tf`.
