# CS-675 Final Project: Big Data Analytics at Cloud Scale — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PySpark analytics pipeline that joins NYC Yellow Taxi trips with NOAA weather data, runs locally first on a monthly slice, then deploys to AWS at 100M+ row scale using S3 + Athena + EMR Serverless.

**Architecture:** Local development uses the course's Docker PySpark environment on a single-month taxi slice (≈3M rows) joined with NOAA GSOD weather for NYC stations. Cloud deployment uses Terraform to provision S3 + Athena Glue catalog + EMR Serverless, uploads 2019–2022 full Yellow Taxi data (≈260M rows) + full NOAA GSOD, and runs the same PySpark jobs pointing at `s3://` paths.

**Tech Stack:** PySpark 3.x, Docker (local), AWS S3 + Athena + EMR Serverless + Glue, Terraform 1.5+, Python 3.12, pytest

---

## Research Questions

1. Does precipitation increase average trip duration and fare?
2. Which pickup zones see the biggest demand drops during rain/snow?
3. How does temperature affect hourly ride volume throughout the day?
4. Do snowstorms correlate with longer trips but shorter distances (traffic)?

---

## File Map

| File | Responsibility |
|------|----------------|
| `docker-compose.yml` | Local PySpark + Spark History Server |
| `Makefile` | Lifecycle: up/down/test/run local scripts |
| `pyproject.toml` | Python deps (pyspark, pytest, pandas) |
| `work/constants.py` | Data paths, schema definitions, NYC weather station IDs |
| `work/spark_helper.py` | `get_spark()` factory (auto-detects local vs cloud) |
| `work/01_explore.py` | Profile raw taxi + weather schemas and row counts |
| `work/02_preprocess.py` | Full preprocessing pipeline: impute → outlier → normalize → encode → bin |
| `work/03_analytics.py` | Cross-source join queries (4 analytical queries) |
| `cloud/02_preprocess_cloud.py` | Same as 02 but reads/writes `s3://` paths, drops local config |
| `cloud/03_analytics_cloud.py` | Same as 03 but reads/writes `s3://` paths |
| `infrastructure/main.tf` | S3 bucket + Athena workgroup + Glue DB + EMR Serverless |
| `infrastructure/variables.tf` | `student_id`, `aws_region` |
| `infrastructure/outputs.tf` | Bucket name, Athena workgroup, EMR app ID |
| `infrastructure/glue_taxi.tf` | Glue table for Yellow Taxi Parquet partitioned by year/month |
| `infrastructure/glue_weather.tf` | Glue table for NOAA GSOD CSV |
| `tests/conftest.py` | Shared `spark` pytest fixture (local SparkSession) |
| `tests/test_preprocess.py` | Unit tests for preprocessing transformations |
| `tests/test_analytics.py` | Unit tests for join logic and query correctness |
| `README.md` | Project overview, dataset description, run instructions, results |

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
      - JUPYTER_ENABLE_LAB=yes
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
	mkdir -p data/taxi data/weather
	# Yellow taxi Jan 2022 (~3M rows, ~50MB)
	curl -L "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-01.parquet" \
	     -o data/taxi/yellow_2022-01.parquet
	# NOAA GSOD 2022 for NY stations (small CSV)
	curl -L "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/2022/94728.csv" \
	     -o data/weather/gsod_2022_central_park.csv
	curl -L "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/2022/94789.csv" \
	     -o data/weather/gsod_2022_jfk.csv
	curl -L "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/2022/14732.csv" \
	     -o data/weather/gsod_2022_lga.csv
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
    "pandas==2.2.0",
]
```

- [ ] **Step 4: Create `work/constants.py`**

```python
# work/constants.py
import os

IS_CLOUD = os.environ.get("CS675_ENV") == "cloud"

# Local paths
LOCAL_TAXI_PATH = "/home/jovyan/data/taxi/"
LOCAL_WEATHER_PATH = "/home/jovyan/data/weather/"
LOCAL_OUTPUT_PATH = "/home/jovyan/data/output/"

# Cloud paths (set CS675_BUCKET env var)
BUCKET = os.environ.get("CS675_BUCKET", "ds-student-workspace")
CLOUD_TAXI_PATH = f"s3://{BUCKET}/data/taxi/"
CLOUD_WEATHER_PATH = f"s3://{BUCKET}/data/weather/"
CLOUD_OUTPUT_PATH = f"s3://{BUCKET}/output/"

# Active paths
TAXI_PATH = CLOUD_TAXI_PATH if IS_CLOUD else LOCAL_TAXI_PATH
WEATHER_PATH = CLOUD_WEATHER_PATH if IS_CLOUD else LOCAL_WEATHER_PATH
OUTPUT_PATH = CLOUD_OUTPUT_PATH if IS_CLOUD else LOCAL_OUTPUT_PATH

# NYC NOAA GSOD station IDs for Central Park, JFK, LGA
NYC_WEATHER_STATIONS = ["94728", "94789", "14732"]

# Taxi schema columns we keep after preprocessing
TAXI_KEEP_COLS = [
    "pickup_date", "pickup_hour", "PULocationID", "DOLocationID",
    "trip_distance", "fare_amount", "tip_amount", "total_amount",
    "passenger_count", "payment_type",
]

WEATHER_KEEP_COLS = [
    "DATE", "STATION", "TEMP", "DEWP", "WDSP", "PRCP", "SNDP", "FRSHTT",
]
```

- [ ] **Step 5: Create `work/spark_helper.py`**

```python
# work/spark_helper.py
from pyspark.sql import SparkSession
import os

def get_spark(app_name: str = "cs675") -> SparkSession:
    """
    Returns a SparkSession. In cloud (EMR Serverless), uses getOrCreate() only.
    Locally, adds event log and UI config.
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
# Wait 15 seconds for container to start
docker compose ps
```
Expected: `pyspark` service shows `Up`

- [ ] **Step 8: Download sample data**

```bash
make download-data
ls data/taxi/ data/weather/
```
Expected: `yellow_2022-01.parquet`, `gsod_2022_central_park.csv`, `gsod_2022_jfk.csv`, `gsod_2022_lga.csv`

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
from work.constants import TAXI_PATH, WEATHER_PATH
import os

spark = get_spark("01_explore")

# ── Taxi ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TAXI DATA")
print("=" * 60)

taxi = spark.read.parquet(TAXI_PATH)
taxi.printSchema()
print(f"Row count: {taxi.count():,}")
taxi.describe("trip_distance", "fare_amount", "tip_amount", "passenger_count").show()

# Null counts
from pyspark.sql.functions import col, count, when, isnan
null_counts = taxi.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c)
    for c in ["trip_distance", "fare_amount", "passenger_count", "tpep_pickup_datetime"]
])
print("Null counts:")
null_counts.show()

# ── Weather ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("WEATHER DATA")
print("=" * 60)

weather = spark.read.option("header", True).option("inferSchema", True).csv(WEATHER_PATH)
weather.printSchema()
print(f"Row count: {weather.count():,}")
weather.describe("TEMP", "PRCP", "WDSP").show()

null_w = weather.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in ["DATE", "TEMP", "PRCP", "WDSP"]
])
print("Weather null counts:")
null_w.show()

spark.stop()
```

- [ ] **Step 2: Run exploration**

```bash
make run-explore
```
Expected: schema printed, row counts shown, null counts table visible. Record observations in `README.md`.

- [ ] **Step 3: Commit**

```bash
git add work/01_explore.py
git commit -m "feat: data exploration script for taxi + weather"
```

---

## Task 3: Preprocessing Pipeline (with TDD)

**Files:**
- Create: `tests/test_preprocess.py`
- Create: `work/02_preprocess.py`

- [ ] **Step 1: Write failing tests in `tests/test_preprocess.py`**

```python
# tests/test_preprocess.py
import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, TimestampType
import datetime

def make_taxi_row(spark, **overrides):
    """Helper: create a one-row taxi DataFrame with defaults."""
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


def make_weather_row(spark, **overrides):
    defaults = dict(
        DATE="2022-01-15",
        STATION="94728",
        TEMP=32.0,
        DEWP=25.0,
        WDSP=10.0,
        PRCP=0.0,
        SNDP=0.0,
        FRSHTT="000000",
    )
    defaults.update(overrides)
    return spark.createDataFrame([Row(**defaults)])


class TestImputation:
    def test_null_passenger_count_filled_with_1(self, spark):
        from work.preprocess_steps import impute_taxi
        df = make_taxi_row(spark, passenger_count=None)
        result = impute_taxi(df)
        assert result.first()["passenger_count"] == 1

    def test_null_fare_amount_rows_dropped(self, spark):
        from work.preprocess_steps import impute_taxi
        df = make_taxi_row(spark, fare_amount=None)
        result = impute_taxi(df)
        assert result.count() == 0  # critical field: drop rather than impute


class TestOutlierTreatment:
    def test_negative_trip_distance_dropped(self, spark):
        from work.preprocess_steps import remove_taxi_outliers
        df = make_taxi_row(spark, trip_distance=-1.0)
        result = remove_taxi_outliers(df)
        assert result.count() == 0

    def test_extreme_fare_capped_at_500(self, spark):
        from work.preprocess_steps import remove_taxi_outliers
        df = make_taxi_row(spark, fare_amount=9999.0)
        result = remove_taxi_outliers(df)
        assert result.first()["fare_amount"] == 500.0

    def test_zero_distance_trip_dropped(self, spark):
        from work.preprocess_steps import remove_taxi_outliers
        df = make_taxi_row(spark, trip_distance=0.0)
        result = remove_taxi_outliers(df)
        assert result.count() == 0


class TestNormalization:
    def test_fare_amount_min_max_normalized(self, spark):
        from work.preprocess_steps import normalize_taxi
        # two rows: fare 0 and fare 100 -> normalized to 0.0 and 1.0
        rows = [
            make_taxi_row(spark, fare_amount=0.0),
            make_taxi_row(spark, fare_amount=100.0),
        ]
        from functools import reduce
        df = reduce(lambda a, b: a.union(b), rows)
        result = normalize_taxi(df)
        norms = sorted([r["fare_amount_norm"] for r in result.collect()])
        assert norms[0] == pytest.approx(0.0)
        assert norms[1] == pytest.approx(1.0)


class TestEncoding:
    def test_payment_type_one_hot(self, spark):
        from work.preprocess_steps import encode_taxi
        df = make_taxi_row(spark, payment_type=1)
        result = encode_taxi(df)
        row = result.first()
        assert row["payment_credit_card"] == 1
        assert row["payment_cash"] == 0

    def test_unknown_payment_type_zeroed(self, spark):
        from work.preprocess_steps import encode_taxi
        df = make_taxi_row(spark, payment_type=99)
        result = encode_taxi(df)
        row = result.first()
        assert row["payment_credit_card"] == 0
        assert row["payment_cash"] == 0


class TestBinning:
    def test_trip_distance_binned_short(self, spark):
        from work.preprocess_steps import bin_taxi
        df = make_taxi_row(spark, trip_distance=0.8)
        result = bin_taxi(df)
        assert result.first()["distance_bucket"] == "short"  # <1 mile

    def test_trip_distance_binned_long(self, spark):
        from work.preprocess_steps import bin_taxi
        df = make_taxi_row(spark, trip_distance=15.0)
        result = bin_taxi(df)
        assert result.first()["distance_bucket"] == "long"  # >10 miles

    def test_temp_binned_freezing(self, spark):
        from work.preprocess_steps import bin_weather
        df = make_weather_row(spark, TEMP=28.0)
        result = bin_weather(df)
        assert result.first()["temp_bucket"] == "freezing"  # <32°F


class TestWeatherImputation:
    def test_missing_prcp_filled_with_zero(self, spark):
        from work.preprocess_steps import impute_weather
        df = make_weather_row(spark, PRCP=99.99)  # NOAA uses 99.99 as missing
        result = impute_weather(df)
        assert result.first()["PRCP"] == 0.0
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
make test
```
Expected: `ImportError: cannot import name 'impute_taxi' from 'work.preprocess_steps'` (module not yet created)

- [ ] **Step 3: Create `work/preprocess_steps.py` with minimal implementations**

```python
# work/preprocess_steps.py
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, lit, min as spark_min, max as spark_max, to_date,
    hour, greatest, least
)

# ── Taxi Imputation ─────────────────────────────────────────────────────────────
def impute_taxi(df: DataFrame) -> DataFrame:
    """Fill passenger_count nulls with 1; drop rows with null fare_amount."""
    return (
        df.fillna({"passenger_count": 1})
          .filter(col("fare_amount").isNotNull())
    )

# ── Taxi Outlier Removal ────────────────────────────────────────────────────────
def remove_taxi_outliers(df: DataFrame) -> DataFrame:
    """
    Drop trips with distance ≤ 0 or fare < 0.
    Cap fare at $500 (legitimate max for airport trips).
    Before: fare ranges 0–99999; After: 0.01–500.
    """
    return (
        df.filter(col("trip_distance") > 0)
          .filter(col("fare_amount") >= 0)
          .withColumn("fare_amount", least(col("fare_amount"), lit(500.0)))
    )

# ── Taxi Normalization ──────────────────────────────────────────────────────────
def normalize_taxi(df: DataFrame) -> DataFrame:
    """
    Min-max normalize fare_amount and trip_distance.
    Applied per-batch; in production use a fitted scaler or use z-score with known mean/std.
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
        df.withColumn("fare_amount_norm",
                      (col("fare_amount") - lit(float(stats["fare_min"]))) / lit(fare_range))
          .withColumn("trip_distance_norm",
                      (col("trip_distance") - lit(float(stats["dist_min"]))) / lit(dist_range))
    )

# ── Taxi Encoding ───────────────────────────────────────────────────────────────
def encode_taxi(df: DataFrame) -> DataFrame:
    """
    One-hot encode payment_type:
      1=credit card, 2=cash, 3=no charge, 4=dispute, 5=unknown, 6=voided
    """
    return (
        df.withColumn("payment_credit_card", when(col("payment_type") == 1, 1).otherwise(0))
          .withColumn("payment_cash",        when(col("payment_type") == 2, 1).otherwise(0))
          .withColumn("payment_no_charge",   when(col("payment_type") == 3, 1).otherwise(0))
    )

# ── Taxi Binning ────────────────────────────────────────────────────────────────
def bin_taxi(df: DataFrame) -> DataFrame:
    """
    Bin trip_distance: short (<1mi), medium (1-10mi), long (>10mi).
    Rationale: matches fare tiers and neighborhood vs outer-borough patterns.
    """
    return df.withColumn(
        "distance_bucket",
        when(col("trip_distance") < 1.0, "short")
        .when(col("trip_distance") <= 10.0, "medium")
        .otherwise("long")
    )

# ── Weather Imputation ──────────────────────────────────────────────────────────
def impute_weather(df: DataFrame) -> DataFrame:
    """
    NOAA GSOD uses 99.99 as sentinel for missing precipitation and 9999.9 for missing wind.
    Replace with 0.0 (no rain, no wind data → conservative assumption).
    """
    return (
        df.withColumn("PRCP", when(col("PRCP") >= 99.0, lit(0.0)).otherwise(col("PRCP")))
          .withColumn("WDSP", when(col("WDSP") >= 999.0, lit(0.0)).otherwise(col("WDSP")))
          .withColumn("SNDP", when(col("SNDP") >= 999.0, lit(0.0)).otherwise(col("SNDP")))
    )

# ── Weather Binning ─────────────────────────────────────────────────────────────
def bin_weather(df: DataFrame) -> DataFrame:
    """
    Bin TEMP (°F): freezing (<32), cold (32-50), mild (50-70), warm (>70).
    Bin PRCP: dry (0), light (0-0.1in), moderate (0.1-0.5in), heavy (>0.5in).
    """
    return (
        df.withColumn(
            "temp_bucket",
            when(col("TEMP") < 32.0, "freezing")
            .when(col("TEMP") < 50.0, "cold")
            .when(col("TEMP") < 70.0, "mild")
            .otherwise("warm")
        )
        .withColumn(
            "prcp_bucket",
            when(col("PRCP") == 0.0, "dry")
            .when(col("PRCP") <= 0.1, "light_rain")
            .when(col("PRCP") <= 0.5, "moderate_rain")
            .otherwise("heavy_rain")
        )
    )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
make test
```
Expected: `13 passed` (all preprocess tests green)

- [ ] **Step 5: Write `work/02_preprocess.py` (full pipeline)**

```python
# work/02_preprocess.py
from work.spark_helper import get_spark
from work.constants import TAXI_PATH, WEATHER_PATH, OUTPUT_PATH
from work.preprocess_steps import (
    impute_taxi, remove_taxi_outliers, normalize_taxi, encode_taxi, bin_taxi,
    impute_weather, bin_weather,
)
from pyspark.sql.functions import to_date, hour, col

spark = get_spark("02_preprocess")

# ── Taxi Pipeline ───────────────────────────────────────────────────────────────
print("Loading taxi data...")
taxi_raw = spark.read.parquet(TAXI_PATH)
print(f"  Raw rows: {taxi_raw.count():,}")

taxi = (
    taxi_raw
    .withColumn("pickup_date", to_date(col("tpep_pickup_datetime")))
    .withColumn("pickup_hour", hour(col("tpep_pickup_datetime")))
)
taxi = impute_taxi(taxi)
taxi = remove_taxi_outliers(taxi)
taxi = normalize_taxi(taxi)
taxi = encode_taxi(taxi)
taxi = bin_taxi(taxi)

print(f"  After preprocessing: {taxi.count():,} rows")
print("  Sample:")
taxi.select(
    "pickup_date", "pickup_hour", "PULocationID", "trip_distance",
    "fare_amount", "fare_amount_norm", "distance_bucket", "payment_credit_card"
).show(5)

taxi.write.mode("overwrite").parquet(OUTPUT_PATH + "taxi_clean/")
print(f"  Written to {OUTPUT_PATH}taxi_clean/")

# ── Weather Pipeline ─────────────────────────────────────────────────────────────
print("\nLoading weather data...")
weather_raw = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(WEATHER_PATH)
)
print(f"  Raw rows: {weather_raw.count():,}")

weather = impute_weather(weather_raw)
weather = bin_weather(weather)

print(f"  After preprocessing: {weather.count():,} rows")
weather.select("DATE", "STATION", "TEMP", "temp_bucket", "PRCP", "prcp_bucket").show(5)

weather.write.mode("overwrite").parquet(OUTPUT_PATH + "weather_clean/")
print(f"  Written to {OUTPUT_PATH}weather_clean/")

spark.stop()
```

- [ ] **Step 6: Run preprocessing**

```bash
make run-preprocess
```
Expected: row counts printed before and after (expect 5-15% row drop from outlier removal), both parquet outputs created.

- [ ] **Step 7: Commit**

```bash
git add work/preprocess_steps.py work/02_preprocess.py tests/test_preprocess.py
git commit -m "feat: full preprocessing pipeline with unit tests (impute/outlier/normalize/encode/bin)"
```

---

## Task 4: Cross-Source Join Analytics (with TDD)

**Files:**
- Create: `tests/test_analytics.py`
- Create: `work/03_analytics.py`

- [ ] **Step 1: Write failing tests in `tests/test_analytics.py`**

```python
# tests/test_analytics.py
import datetime
import pytest
from pyspark.sql import Row


def make_clean_taxi(spark, **kw):
    defaults = dict(
        pickup_date=datetime.date(2022, 1, 15),
        pickup_hour=10,
        PULocationID=161,
        DOLocationID=236,
        trip_distance=2.5,
        fare_amount=12.0,
        fare_amount_norm=0.5,
        tip_amount=2.0,
        total_amount=15.0,
        passenger_count=1,
        payment_type=1,
        payment_credit_card=1,
        payment_cash=0,
        payment_no_charge=0,
        distance_bucket="medium",
        trip_distance_norm=0.5,
    )
    defaults.update(kw)
    return spark.createDataFrame([Row(**defaults)])


def make_clean_weather(spark, **kw):
    defaults = dict(
        DATE="2022-01-15",
        STATION="94728",
        TEMP=32.0,
        DEWP=25.0,
        WDSP=5.0,
        PRCP=0.0,
        SNDP=0.0,
        FRSHTT="000000",
        temp_bucket="freezing",
        prcp_bucket="dry",
    )
    defaults.update(kw)
    return spark.createDataFrame([Row(**defaults)])


class TestJoin:
    def test_taxi_weather_join_on_date(self, spark):
        from work.analytics import join_taxi_weather
        taxi = make_clean_taxi(spark)
        weather = make_clean_weather(spark)
        result = join_taxi_weather(taxi, weather)
        assert result.count() == 1
        row = result.first()
        assert row["pickup_date"].strftime("%Y-%m-%d") == "2022-01-15"
        assert row["TEMP"] == 32.0

    def test_no_match_returns_empty(self, spark):
        from work.analytics import join_taxi_weather
        taxi = make_clean_taxi(spark, pickup_date=datetime.date(2022, 1, 15))
        weather = make_clean_weather(spark, DATE="2022-01-20")
        result = join_taxi_weather(taxi, weather)
        assert result.count() == 0


class TestQuery1PrecipitationEffect:
    def test_groups_by_prcp_bucket(self, spark):
        from work.analytics import query_precipitation_effect
        taxi = make_clean_taxi(spark)
        weather = make_clean_weather(spark, PRCP=0.3, prcp_bucket="moderate_rain")
        joined = __import__("work.analytics", fromlist=["join_taxi_weather"]).join_taxi_weather(taxi, weather)
        result = query_precipitation_effect(joined)
        row = result.first()
        assert "prcp_bucket" in result.columns
        assert "avg_fare" in result.columns
        assert "trip_count" in result.columns

    def test_dry_day_has_trips(self, spark):
        from work.analytics import query_precipitation_effect, join_taxi_weather
        taxi = make_clean_taxi(spark)
        weather = make_clean_weather(spark, PRCP=0.0, prcp_bucket="dry")
        joined = join_taxi_weather(taxi, weather)
        result = query_precipitation_effect(joined)
        assert result.filter("prcp_bucket = 'dry'").first()["trip_count"] == 1


class TestQuery2HourlyDemandByTemp:
    def test_returns_hour_and_temp_bucket(self, spark):
        from work.analytics import query_hourly_demand_by_temp, join_taxi_weather
        taxi = make_clean_taxi(spark, pickup_hour=10)
        weather = make_clean_weather(spark)
        joined = join_taxi_weather(taxi, weather)
        result = query_hourly_demand_by_temp(joined)
        assert "pickup_hour" in result.columns
        assert "temp_bucket" in result.columns
        assert "trip_count" in result.columns
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
make test
```
Expected: `ImportError: cannot import name 'join_taxi_weather' from 'work.analytics'`

- [ ] **Step 3: Create `work/analytics.py`**

```python
# work/analytics.py
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, avg, count, round as spark_round, to_date, stddev
)

def join_taxi_weather(taxi: DataFrame, weather: DataFrame) -> DataFrame:
    """
    Inner join taxi with weather on pickup_date = DATE (Central Park station only).
    We use Central Park as the representative NYC weather station.
    Central Park STATION id = '94728'.
    """
    nyc_weather = weather.filter(col("STATION") == "94728")
    nyc_weather = nyc_weather.withColumn("join_date", to_date(col("DATE")))
    return taxi.join(nyc_weather, taxi["pickup_date"] == nyc_weather["join_date"], "inner")


def query_precipitation_effect(joined: DataFrame) -> DataFrame:
    """
    Q1: How does precipitation affect average fare and trip volume?
    Groups by prcp_bucket, returns avg_fare, avg_tip_pct, trip_count.
    """
    return (
        joined
        .groupBy("prcp_bucket")
        .agg(
            spark_round(avg("fare_amount"), 2).alias("avg_fare"),
            spark_round(avg("trip_distance"), 2).alias("avg_distance"),
            spark_round(avg("tip_amount") / avg("fare_amount") * 100, 1).alias("avg_tip_pct"),
            count("*").alias("trip_count"),
        )
        .orderBy("prcp_bucket")
    )


def query_hourly_demand_by_temp(joined: DataFrame) -> DataFrame:
    """
    Q2: How does temperature bucket affect hourly trip volume?
    Returns pickup_hour x temp_bucket heatmap data.
    """
    return (
        joined
        .groupBy("pickup_hour", "temp_bucket")
        .agg(count("*").alias("trip_count"))
        .orderBy("pickup_hour", "temp_bucket")
    )


def query_zone_weather_sensitivity(joined: DataFrame) -> DataFrame:
    """
    Q3: Which pickup zones (PULocationID) see biggest demand change during rain?
    Compares rain (prcp_bucket != 'dry') vs dry days per zone.
    """
    from pyspark.sql.functions import when, lit

    rain_flag = when(col("prcp_bucket") != "dry", "rainy").otherwise("dry")

    return (
        joined
        .withColumn("weather_type", rain_flag)
        .groupBy("PULocationID", "weather_type")
        .agg(count("*").alias("trip_count"))
        .orderBy("PULocationID", "weather_type")
    )


def query_snowstorm_impact(joined: DataFrame) -> DataFrame:
    """
    Q4: Do snowstorms (SNDP > 1 inch) increase trip duration vs distance ratio?
    A higher fare/distance ratio suggests slower traffic (more time = more metered fare).
    """
    snow_flag = when(col("SNDP") > 1.0, "snow").otherwise("no_snow")

    return (
        joined
        .withColumn("snow_condition", snow_flag)
        .groupBy("snow_condition")
        .agg(
            spark_round(avg("fare_amount"), 2).alias("avg_fare"),
            spark_round(avg("trip_distance"), 2).alias("avg_distance"),
            spark_round(avg("fare_amount") / avg("trip_distance"), 2).alias("fare_per_mile"),
            count("*").alias("trip_count"),
        )
    )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
make test
```
Expected: `all tests passed` (≈20 tests total)

- [ ] **Step 5: Write `work/03_analytics.py` (full run)**

```python
# work/03_analytics.py
from work.spark_helper import get_spark
from work.constants import OUTPUT_PATH
from work.analytics import (
    join_taxi_weather,
    query_precipitation_effect,
    query_hourly_demand_by_temp,
    query_zone_weather_sensitivity,
    query_snowstorm_impact,
)

spark = get_spark("03_analytics")

print("Loading preprocessed data...")
taxi = spark.read.parquet(OUTPUT_PATH + "taxi_clean/")
weather = spark.read.parquet(OUTPUT_PATH + "weather_clean/")
print(f"  Taxi: {taxi.count():,} rows | Weather: {weather.count():,} rows")

joined = join_taxi_weather(taxi, weather)
print(f"  After join: {joined.count():,} rows")

print("\n── Q1: Precipitation Effect on Fares ─────────────────────────────────")
query_precipitation_effect(joined).show()

print("\n── Q2: Hourly Demand by Temperature ──────────────────────────────────")
query_hourly_demand_by_temp(joined).show(48)

print("\n── Q3: Zone Sensitivity to Rain ──────────────────────────────────────")
query_zone_weather_sensitivity(joined).orderBy("trip_count", ascending=False).show(20)

print("\n── Q4: Snowstorm Impact on Fare/Mile ─────────────────────────────────")
query_snowstorm_impact(joined).show()

# Save results as Parquet for inclusion in repo
joined.write.mode("overwrite").parquet(OUTPUT_PATH + "joined/")
query_precipitation_effect(joined).write.mode("overwrite").parquet(OUTPUT_PATH + "q1_prcp_effect/")
query_hourly_demand_by_temp(joined).write.mode("overwrite").parquet(OUTPUT_PATH + "q2_hourly_temp/")
query_zone_weather_sensitivity(joined).write.mode("overwrite").parquet(OUTPUT_PATH + "q3_zone_rain/")
query_snowstorm_impact(joined).write.mode("overwrite").parquet(OUTPUT_PATH + "q4_snowstorm/")

print("\nAll results written to", OUTPUT_PATH)
spark.stop()
```

- [ ] **Step 6: Run analytics**

```bash
make run-analytics
```
Expected: 4 result tables printed. Record interesting findings (e.g., "heavy rain days show 8% higher fare/mile").

- [ ] **Step 7: Commit**

```bash
git add work/analytics.py work/03_analytics.py tests/test_analytics.py
git commit -m "feat: 4 cross-source join queries taxi x weather with unit tests"
```

---

## Task 5: AWS Infrastructure with Terraform

**Files:**
- Create: `infrastructure/main.tf`
- Create: `infrastructure/variables.tf`
- Create: `infrastructure/outputs.tf`
- Create: `infrastructure/glue_taxi.tf`
- Create: `infrastructure/glue_weather.tf`

> **Prerequisite:** AWS CLI configured with profile `ds`. Run `aws sts get-caller-identity --profile ds` to verify.

- [ ] **Step 1: Create `infrastructure/variables.tf`**

```hcl
# infrastructure/variables.tf
variable "student_id" {
  description = "Your student ID (used to name all resources)"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}
```

- [ ] **Step 2: Create `infrastructure/main.tf`**

```hcl
# infrastructure/main.tf
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
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

# ── S3 ──────────────────────────────────────────────────────────────────────────
resource "aws_s3_bucket" "workspace" {
  bucket        = local.bucket_name
  force_destroy = true
  tags = { Owner = var.student_id, Project = "cs675" }
}

resource "aws_s3_bucket_versioning" "workspace" {
  bucket = aws_s3_bucket.workspace.id
  versioning_configuration { status = "Disabled" }
}

# Athena query results prefix
resource "aws_s3_object" "athena_results_prefix" {
  bucket  = aws_s3_bucket.workspace.id
  key     = "athena-results/.keep"
  content = ""
}

# ── Athena ───────────────────────────────────────────────────────────────────────
resource "aws_athena_workgroup" "main" {
  name = local.athena_workgroup
  configuration {
    result_configuration {
      output_location = "s3://${aws_s3_bucket.workspace.bucket}/athena-results/"
    }
    bytes_scanned_cutoff_per_query = 10737418240  # 10 GB scan cap per query
  }
  tags = { Owner = var.student_id }
}

# ── Glue Database ────────────────────────────────────────────────────────────────
resource "aws_glue_catalog_database" "main" {
  name = local.glue_db
}

# ── EMR Serverless ───────────────────────────────────────────────────────────────
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
# Glue table for Yellow Taxi Parquet partitioned by year and month

resource "aws_glue_catalog_table" "yellow_taxi" {
  database_name = aws_glue_catalog_database.main.name
  name          = "yellow_taxi"

  table_type = "EXTERNAL_TABLE"

  parameters = {
    "classification"         = "parquet"
    "parquet.compress"       = "SNAPPY"
    "projection.enabled"     = "true"
    "projection.year.type"   = "integer"
    "projection.year.range"  = "2019,2022"
    "projection.month.type"  = "integer"
    "projection.month.range" = "1,12"
    "projection.month.digits" = "2"
    "storage.location.template" = "s3://${aws_s3_bucket.workspace.bucket}/data/taxi/year=$${year}/month=$${month}/"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.workspace.bucket}/data/taxi/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"

    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }

    columns {
      name = "pickup_date"
      type = "date"
    }
    columns {
      name = "pickup_hour"
      type = "int"
    }
    columns {
      name = "PULocationID"
      type = "int"
    }
    columns {
      name = "DOLocationID"
      type = "int"
    }
    columns {
      name = "trip_distance"
      type = "double"
    }
    columns {
      name = "fare_amount"
      type = "double"
    }
    columns {
      name = "fare_amount_norm"
      type = "double"
    }
    columns {
      name = "distance_bucket"
      type = "string"
    }
    columns {
      name = "payment_credit_card"
      type = "int"
    }
  }

  partition_keys {
    name = "year"
    type = "int"
  }
  partition_keys {
    name = "month"
    type = "int"
  }
}
```

- [ ] **Step 4: Create `infrastructure/glue_weather.tf`**

```hcl
# infrastructure/glue_weather.tf
resource "aws_glue_catalog_table" "noaa_weather" {
  database_name = aws_glue_catalog_database.main.name
  name          = "noaa_weather"

  table_type = "EXTERNAL_TABLE"

  parameters = {
    "classification"  = "parquet"
    "parquet.compress" = "SNAPPY"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.workspace.bucket}/data/weather/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"

    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }

    columns {
      name = "DATE"
      type = "string"
    }
    columns {
      name = "STATION"
      type = "string"
    }
    columns {
      name = "TEMP"
      type = "double"
    }
    columns {
      name = "PRCP"
      type = "double"
    }
    columns {
      name = "SNDP"
      type = "double"
    }
    columns {
      name = "WDSP"
      type = "double"
    }
    columns {
      name = "temp_bucket"
      type = "string"
    }
    columns {
      name = "prcp_bucket"
      type = "string"
    }
  }
}
```

- [ ] **Step 5: Create `infrastructure/outputs.tf`**

```hcl
# infrastructure/outputs.tf
output "bucket_name" {
  value = aws_s3_bucket.workspace.bucket
}

output "athena_workgroup" {
  value = aws_athena_workgroup.main.name
}

output "glue_database" {
  value = aws_glue_catalog_database.main.name
}

output "emr_app_id" {
  value = aws_emrserverless_application.spark.id
}
```

- [ ] **Step 6: Create `infrastructure/terraform.tfvars` (gitignored)**

```bash
# DO NOT COMMIT THIS FILE — it's in .gitignore
cat >> .gitignore << 'EOF'
infrastructure/terraform.tfvars
infrastructure/.terraform/
infrastructure/terraform.tfstate*
infrastructure/.terraform.lock.hcl
data/
spark-events/
EOF
```

```hcl
# infrastructure/terraform.tfvars  (create manually, do not commit)
student_id = "your-student-id"
```

- [ ] **Step 7: Initialize and apply Terraform**

```bash
cd infrastructure
terraform init
terraform plan
terraform apply
```
Expected output:
```
Apply complete! Resources: 8 added.
Outputs:
bucket_name     = "ds-your-id-workspace"
athena_workgroup = "ds-your-id"
glue_database   = "ds_your_id"
emr_app_id      = "00f..."
```

- [ ] **Step 8: Commit infrastructure code**

```bash
git add infrastructure/*.tf .gitignore
git commit -m "feat: Terraform for S3 + Athena + Glue catalog + EMR Serverless"
```

---

## Task 6: Upload Full Data to S3

> **Goal:** Upload 2019–2022 Yellow Taxi (~260M rows) + NOAA GSOD weather for NYC stations (2019–2022) to S3.

- [ ] **Step 1: Create `scripts/download_full_data.sh`**

```bash
#!/bin/bash
# scripts/download_full_data.sh
# Downloads 2019–2022 Yellow Taxi parquet files from TLC
set -e

BUCKET="${CS675_BUCKET:?Set CS675_BUCKET env var}"
PROFILE="${AWS_PROFILE:-ds}"

mkdir -p /tmp/taxi_download

YEARS="2019 2020 2021 2022"
MONTHS="01 02 03 04 05 06 07 08 09 10 11 12"

for YEAR in $YEARS; do
  for MONTH in $MONTHS; do
    FILE="yellow_tripdata_${YEAR}-${MONTH}.parquet"
    URL="https://d37ci6vzurychx.cloudfront.net/trip-data/${FILE}"
    echo "Downloading $FILE..."
    curl -L "$URL" -o "/tmp/taxi_download/${FILE}"
    aws s3 cp "/tmp/taxi_download/${FILE}" \
      "s3://${BUCKET}/data/taxi/year=${YEAR}/month=${MONTH}/${FILE}" \
      --profile "${PROFILE}"
    rm "/tmp/taxi_download/${FILE}"
    echo "  Uploaded $FILE"
  done
done

echo "Taxi upload complete."
```

- [ ] **Step 2: Create `scripts/download_weather.sh`**

```bash
#!/bin/bash
# scripts/download_weather.sh
set -e

BUCKET="${CS675_BUCKET:?Set CS675_BUCKET env var}"
PROFILE="${AWS_PROFILE:-ds}"

# NYC stations: Central Park=94728, JFK=94789, LGA=14732
STATIONS="94728 94789 14732"
YEARS="2019 2020 2021 2022"

mkdir -p /tmp/weather_download

for YEAR in $YEARS; do
  for STATION in $STATIONS; do
    FILE="gsod_${YEAR}_${STATION}.csv"
    URL="https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/${YEAR}/${STATION}.csv"
    echo "Downloading $FILE..."
    curl -L "$URL" -o "/tmp/weather_download/${FILE}" || echo "  WARN: $FILE not available"
    if [ -f "/tmp/weather_download/${FILE}" ]; then
      aws s3 cp "/tmp/weather_download/${FILE}" \
        "s3://${BUCKET}/data/weather/raw/${FILE}" \
        --profile "${PROFILE}"
      rm "/tmp/weather_download/${FILE}"
    fi
  done
done

echo "Weather upload complete."
```

- [ ] **Step 3: Run uploads**

```bash
export CS675_BUCKET=$(cd infrastructure && terraform output -raw bucket_name)
chmod +x scripts/download_full_data.sh scripts/download_weather.sh
./scripts/download_full_data.sh
./scripts/download_weather.sh
```
Expected: S3 console shows `data/taxi/year=2019/month=01/` through `year=2022/month=12/` and `data/weather/raw/`.

- [ ] **Step 4: Verify upload**

```bash
aws s3 ls "s3://${CS675_BUCKET}/data/taxi/" --recursive --profile ds | wc -l
```
Expected: 48 (12 months × 4 years)

- [ ] **Step 5: Commit scripts**

```bash
git add scripts/
git commit -m "feat: data upload scripts for full TLC + NOAA to S3"
```

---

## Task 7: Run Preprocessing + Analytics at Cloud Scale

**Files:**
- Create: `cloud/02_preprocess_cloud.py`
- Create: `cloud/03_analytics_cloud.py`
- Create: `cloud/emr_job_runner.sh`

- [ ] **Step 1: Create `cloud/02_preprocess_cloud.py`**

```python
# cloud/02_preprocess_cloud.py
# Identical logic to work/02_preprocess.py but reads from S3 and uses EMR SparkSession.
import os
import sys
sys.path.insert(0, "/home/hadoop/")  # EMR Serverless working dir

from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, hour, col

# On EMR Serverless, SparkSession is provided by the runtime
spark = SparkSession.builder.appName("cs675-preprocess-cloud").getOrCreate()

BUCKET = os.environ["CS675_BUCKET"]
TAXI_IN  = f"s3://{BUCKET}/data/taxi/"
WEATHER_IN = f"s3://{BUCKET}/data/weather/raw/"
TAXI_OUT   = f"s3://{BUCKET}/output/taxi_clean/"
WEATHER_OUT = f"s3://{BUCKET}/output/weather_clean/"

# Import preprocess_steps — uploaded to S3 and provided via --py-files in EMR job
from preprocess_steps import (
    impute_taxi, remove_taxi_outliers, normalize_taxi, encode_taxi, bin_taxi,
    impute_weather, bin_weather,
)

print("Loading taxi from S3...")
taxi_raw = spark.read.parquet(TAXI_IN)
print(f"  Raw rows: {taxi_raw.count():,}")

taxi = taxi_raw.withColumn("pickup_date", to_date(col("tpep_pickup_datetime")))
taxi = taxi.withColumn("pickup_hour", hour(col("tpep_pickup_datetime")))
taxi = impute_taxi(taxi)
taxi = remove_taxi_outliers(taxi)
taxi = normalize_taxi(taxi)
taxi = encode_taxi(taxi)
taxi = bin_taxi(taxi)

print(f"  Clean rows: {taxi.count():,}")
taxi.write.mode("overwrite").partitionBy("pickup_date").parquet(TAXI_OUT)

print("Loading weather from S3...")
weather_raw = spark.read.option("header", True).option("inferSchema", True).csv(WEATHER_IN)
weather = impute_weather(weather_raw)
weather = bin_weather(weather_raw)
weather.write.mode("overwrite").parquet(WEATHER_OUT)

print("Cloud preprocessing complete.")
spark.stop()
```

- [ ] **Step 2: Create `cloud/03_analytics_cloud.py`**

```python
# cloud/03_analytics_cloud.py
import os
import sys
sys.path.insert(0, "/home/hadoop/")

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("cs675-analytics-cloud").getOrCreate()

BUCKET = os.environ["CS675_BUCKET"]
TAXI_IN    = f"s3://{BUCKET}/output/taxi_clean/"
WEATHER_IN = f"s3://{BUCKET}/output/weather_clean/"
OUT        = f"s3://{BUCKET}/output/results/"

from analytics import (
    join_taxi_weather,
    query_precipitation_effect,
    query_hourly_demand_by_temp,
    query_zone_weather_sensitivity,
    query_snowstorm_impact,
)

print("Reading preprocessed data from S3...")
taxi    = spark.read.parquet(TAXI_IN)
weather = spark.read.parquet(WEATHER_IN)
print(f"  Taxi: {taxi.count():,} | Weather: {weather.count():,}")

joined = join_taxi_weather(taxi, weather)
print(f"  Joined: {joined.count():,}")

query_precipitation_effect(joined).write.mode("overwrite").parquet(OUT + "q1/")
query_hourly_demand_by_temp(joined).write.mode("overwrite").parquet(OUT + "q2/")
query_zone_weather_sensitivity(joined).write.mode("overwrite").parquet(OUT + "q3/")
query_snowstorm_impact(joined).write.mode("overwrite").parquet(OUT + "q4/")

print("Cloud analytics complete. Results at:", OUT)
spark.stop()
```

- [ ] **Step 3: Create `cloud/emr_job_runner.sh`**

```bash
#!/bin/bash
# cloud/emr_job_runner.sh — submits PySpark job to EMR Serverless
set -e

BUCKET="${CS675_BUCKET:?}"
PROFILE="${AWS_PROFILE:-ds}"
APP_ID=$(cd infrastructure && terraform output -raw emr_app_id)
ROLE_ARN="${EMR_ROLE_ARN:?Set EMR_ROLE_ARN env var to the EMR execution role ARN}"

SCRIPT="${1:?Usage: $0 <script_name>}"  # e.g. 02_preprocess_cloud

# Upload the script and helper modules to S3
aws s3 cp "cloud/${SCRIPT}.py" "s3://${BUCKET}/scripts/${SCRIPT}.py" --profile "${PROFILE}"
aws s3 cp "work/preprocess_steps.py" "s3://${BUCKET}/scripts/preprocess_steps.py" --profile "${PROFILE}"
aws s3 cp "work/analytics.py" "s3://${BUCKET}/scripts/analytics.py" --profile "${PROFILE}"

echo "Submitting ${SCRIPT} to EMR Serverless app ${APP_ID}..."
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
      \"s3MonitoringConfiguration\": {
        \"logUri\": \"s3://${BUCKET}/emr-logs/\"
      }
    }
  }" \
  --environment "{\"CS675_BUCKET\": \"${BUCKET}\"}" \
  --profile "${PROFILE}" \
  --query "jobRunId" --output text)

echo "Job submitted: ${RUN_ID}"
echo "Waiting for job to complete..."

while true; do
  STATUS=$(aws emr-serverless get-job-run \
    --application-id "${APP_ID}" \
    --job-run-id "${RUN_ID}" \
    --profile "${PROFILE}" \
    --query "jobRun.state" --output text)
  echo "  Status: ${STATUS}"
  if [[ "$STATUS" == "SUCCESS" || "$STATUS" == "FAILED" || "$STATUS" == "CANCELLED" ]]; then
    break
  fi
  sleep 30
done

if [[ "$STATUS" != "SUCCESS" ]]; then
  echo "Job FAILED. Check logs at s3://${BUCKET}/emr-logs/"
  exit 1
fi
echo "Job complete: ${SCRIPT}"
```

- [ ] **Step 4: Run preprocessing job at cloud scale**

```bash
export CS675_BUCKET=$(cd infrastructure && terraform output -raw bucket_name)
export EMR_ROLE_ARN="arn:aws:iam::ACCOUNT_ID:role/EMRServerlessExecutionRole"
chmod +x cloud/emr_job_runner.sh
./cloud/emr_job_runner.sh 02_preprocess_cloud
```
Expected: Job STATUS = SUCCESS. Logs in `s3://bucket/emr-logs/`.

- [ ] **Step 5: Run analytics job at cloud scale**

```bash
./cloud/emr_job_runner.sh 03_analytics_cloud
```
Expected: STATUS = SUCCESS. Results in `s3://bucket/output/results/q1/` through `q4/`.

- [ ] **Step 6: Download and inspect results**

```bash
aws s3 cp "s3://${CS675_BUCKET}/output/results/q1/" /tmp/q1/ --recursive --profile ds
docker compose exec pyspark bash -c "python -c \"
from pyspark.sql import SparkSession
spark = SparkSession.builder.master('local[*]').getOrCreate()
df = spark.read.parquet('/tmp/q1/')
df.show()
\""
```

- [ ] **Step 7: Commit cloud scripts**

```bash
git add cloud/ scripts/
git commit -m "feat: cloud PySpark scripts + EMR Serverless job runner"
```

---

## Task 8: README, Results Discussion, and Cleanup

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# CS-675 Final Project: NYC Taxi + Weather Analytics at Cloud Scale

## Overview

This project analyzes how weather conditions affect NYC Yellow Taxi ridership patterns using two datasets:

- **NYC TLC Yellow Taxi** (2019–2022): ~260M trips from TLC open data
- **NOAA GSOD** (2019–2022): Daily weather summaries for 3 NYC stations (Central Park, JFK, LGA)

**Join key:** `pickup_date` (from taxi) ↔ `DATE` (from weather, Central Park station)

## Research Questions

1. Does precipitation increase average fare and reduce trip volume?
2. Which pickup zones are most sensitive to rain demand shifts?
3. How does temperature shape hourly ridership throughout the day?
4. Do snowstorms increase fare-per-mile (a proxy for slower traffic)?

## Tools Used

- **PySpark** (Spark 3.5): local preprocessing and analytics
- **AWS S3**: cloud storage for 260M+ row datasets
- **AWS Athena** (SQL): interactive queries over S3 Parquet
- **AWS EMR Serverless**: cloud-scale PySpark job execution
- **Terraform**: reproducible infrastructure provisioning

## Run Locally

```bash
# Prerequisites: Docker Desktop
make up
make download-data
make run-preprocess
make run-analytics
```

Spark UI: http://localhost:4040  
History Server: http://localhost:18081

## Run on AWS

```bash
# Prerequisites: AWS CLI + Terraform + configured profile "ds"
export CS675_BUCKET=ds-<your-id>-workspace
export EMR_ROLE_ARN=arn:aws:iam::<account>:role/EMRServerlessExecutionRole

cd infrastructure && terraform init && terraform apply
cd ..

# Upload data (takes ~30–60 min)
./scripts/download_full_data.sh
./scripts/download_weather.sh

# Run jobs
./cloud/emr_job_runner.sh 02_preprocess_cloud
./cloud/emr_job_runner.sh 03_analytics_cloud

# Tear down (important — avoids charges)
cd infrastructure && terraform destroy
```

## Key Results

[Fill in after cloud run — include actual numbers from Q1–Q4 outputs]

Example format:
- Q1: Heavy rain days → avg fare $X (+Y% vs dry), trip count down Z%
- Q4: Snow days → fare/mile $X vs $Y on clear days

## Preprocessing Justification

| Step | Before | After | Rationale |
|------|--------|-------|-----------|
| Impute `passenger_count` | X% null | 0% null | Default to 1 (solo rider most common) |
| Drop null `fare_amount` | X% null | dropped | Fare is primary outcome; can't impute |
| Remove distance ≤ 0 | X rows | removed | Physical impossibility |
| Cap fare at $500 | max $999k | max $500 | Likely data entry errors above $500 |
| Min-max normalize fare | 0–500 range | 0–1 | Enables ML and comparison across years |
| One-hot encode payment_type | 1–6 | binary flags | Categorical → numeric for analysis |
| Bin distance | 0–100 mi | short/medium/long | Meaningful fare tier groupings |
| Bin temperature (°F) | -10–104°F | freezing/cold/mild/warm | Demand behavior changes at these thresholds |
| Replace NOAA 99.99 PRCP | sentinel | 0.0 | NOAA convention: 99.99 = missing, assume no rain |
```

- [ ] **Step 2: Commit README**

```bash
git add README.md
git commit -m "docs: complete README with run instructions, results, preprocessing justification"
```

- [ ] **Step 3: Tear down cloud resources**

```bash
cd infrastructure && terraform destroy
```
Expected: `Destroy complete! Resources: 8 destroyed.`

- [ ] **Step 4: Final verification**

```bash
make test
```
Expected: all tests pass on local machine.

---

## Self-Review Against Spec

| Requirement | Covered by |
|---|---|
| Preprocessing: imputation | Task 3 — `impute_taxi`, `impute_weather` |
| Preprocessing: outlier treatment | Task 3 — `remove_taxi_outliers` |
| Preprocessing: normalization | Task 3 — `normalize_taxi` |
| Preprocessing: encoding | Task 3 — `encode_taxi` |
| Preprocessing: binning | Task 3 — `bin_taxi`, `bin_weather` |
| Before/after justification | Task 8 — README table |
| Cross-source join queries (4) | Task 4 — Q1–Q4 in `analytics.py` |
| Cloud infrastructure | Task 5 — Terraform (S3 + Athena + EMR) |
| 100M+ rows at cloud scale | Task 6–7 — 260M Yellow Taxi 2019–2022 |
| Two joinable datasets | Taxi + NOAA Weather |
| GitHub repo + README | Task 8 |
| Evaluate and discuss results | Task 8 README "Key Results" section |
| Two course tools | PySpark + AWS (Athena/EMR Serverless) |
| Tear down resources | Task 8 Step 3 |

**Gaps / watch-outs:**
- "Key Results" section in README is a placeholder — must fill in actual numbers after cloud run (Task 7 Step 6)
- EMR execution role ARN must be obtained from instructor or your AWS account; this is not provisioned in `main.tf` (instructor provisions it)
- NOAA GSOD files for 2019–2022 require checking availability for each year/station
