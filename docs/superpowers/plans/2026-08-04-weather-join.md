# Weather Join (Q5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 5th analytics query that joins NYC Yellow Taxi trips to NOAA daily weather (clear/rain/snow) by pickup date, wired through the local pipeline, tested, shown in the Streamlit UI, and documented — satisfying the assignment's cross-source join requirement.

**Architecture:** A new `work/weather_helpers.py` loads and categorizes the NOAA CSV into a tiny Spark DataFrame; a new `query_demand_by_weather()` in `work/analytics.py` broadcast-joins it against the existing cleaned taxi DataFrame and groups by weather condition. `work/03_analytics.py` wires it into the existing Q1–Q4 run/print/write flow. `cloud/03_analytics_cloud.py` gets the same wiring for parity but is not executed this round.

**Tech Stack:** PySpark 3.5 (existing), pandas + Altair (existing, UI), NOAA GHCN-Daily public CSV (new external data source), pytest (existing).

## Global Constraints

- No changes to `work/02_preprocess.py`, `data/output/taxi_clean/`, or `work/ml_features.py` — the join happens only at analytics-query time (Approach A from the spec), not during preprocessing.
- No re-provisioning of AWS infra, no `terraform apply`, no EMR Serverless job execution this round — cloud resources stay torn down. The cloud code path (Task 6) is written but not run.
- No changes to `ui/predict_app.py`'s prediction form.
- No new Python dependencies — pyspark, pandas, and altair are already in `pyproject.toml` / the Docker image.
- Weather source: NOAA GHCN-Daily, station `USW00094728` (NYC Central Park), full history at `https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00094728.csv`, filtered locally to `2019-01-01`..`2022-12-31`.
- Spec: `docs/superpowers/specs/2026-08-04-weather-join-design.md`.

---

### Task 1: Weather data download script

**Files:**
- Create: `scripts/download_weather.sh`
- Modify: `Makefile`

**Interfaces:**
- Produces: `data/weather_central_park.csv` with header `STATION,DATE,PRCP,SNOW` (DATE as `YYYY-MM-DD` string, PRCP/SNOW as plain integers, nulls already replaced with `0`), filtered to 2019-01-01..2022-12-31.

- [ ] **Step 1: Create the download script**

```bash
#!/bin/bash
# scripts/download_weather.sh
# Downloads NOAA GHCN-Daily weather for NYC Central Park (station USW00094728),
# filtered to 2019-2022 (matches the taxi dataset's year range).
set -e

mkdir -p data
curl -L "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00094728.csv" \
  | python3 -c "
import csv, sys
w = csv.writer(sys.stdout)
w.writerow(['STATION', 'DATE', 'PRCP', 'SNOW'])
for row in csv.DictReader(sys.stdin):
    if '2019-01-01' <= row['DATE'] <= '2022-12-31':
        w.writerow([row['STATION'], row['DATE'], row['PRCP'].strip() or '0', row['SNOW'].strip() or '0'])
" > data/weather_central_park.csv

echo "Done. Saved to data/weather_central_park.csv"
```

Make it executable:

```bash
chmod +x scripts/download_weather.sh
```

- [ ] **Step 2: Add a Makefile target**

In `Makefile`, add `download-weather` to the `.PHONY` line and add the target next to `download-zones`:

```makefile
download-weather:
	./scripts/download_weather.sh
```

- [ ] **Step 3: Run it and verify the output**

```bash
make download-weather
wc -l data/weather_central_park.csv
head -3 data/weather_central_park.csv
```

Expected: header line + ~1,461 data rows (4 years × ~365.25 days), header exactly `STATION,DATE,PRCP,SNOW`, first data row starts `USW00094728,2019-01-01,`.

- [ ] **Step 4: Commit**

```bash
git add scripts/download_weather.sh Makefile
git commit -m "feat: add NOAA weather download script for Q5 join"
```

---

### Task 2: `work/weather_helpers.py` — load and categorize weather (TDD)

**Files:**
- Create: `work/weather_helpers.py`
- Test: `tests/test_weather.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `load_weather(spark: SparkSession, path: str) -> DataFrame` — columns `STATION` (string), `DATE` (date), `PRCP` (int, null→0), `SNOW` (int, null→0).
  - `categorize_weather(df: DataFrame) -> DataFrame` — same columns plus `weather_condition` (string: `"snow"` | `"rain"` | `"clear"`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_weather.py`:

```python
# tests/test_weather.py
import datetime
from pyspark.sql import Row


def make_weather_row(**kw):
    defaults = dict(STATION="USW00094728", DATE=datetime.date(2022, 1, 15), PRCP=0, SNOW=0)
    defaults.update(kw)
    return Row(**defaults)


class TestCategorizeWeather:
    def test_clear_when_no_precip(self, spark):
        from work.weather_helpers import categorize_weather
        df = spark.createDataFrame([make_weather_row(PRCP=0, SNOW=0)])
        result = categorize_weather(df)
        assert result.first()["weather_condition"] == "clear"

    def test_rain_when_prcp_only(self, spark):
        from work.weather_helpers import categorize_weather
        df = spark.createDataFrame([make_weather_row(PRCP=50, SNOW=0)])
        result = categorize_weather(df)
        assert result.first()["weather_condition"] == "rain"

    def test_snow_when_snow_only(self, spark):
        from work.weather_helpers import categorize_weather
        df = spark.createDataFrame([make_weather_row(PRCP=0, SNOW=20)])
        result = categorize_weather(df)
        assert result.first()["weather_condition"] == "snow"

    def test_snow_takes_priority_over_rain(self, spark):
        from work.weather_helpers import categorize_weather
        df = spark.createDataFrame([make_weather_row(PRCP=94, SNOW=20)])
        result = categorize_weather(df)
        assert result.first()["weather_condition"] == "snow"


class TestLoadWeather:
    def test_loads_and_fills_nulls(self, spark, tmp_path):
        from work.weather_helpers import load_weather
        csv_path = tmp_path / "weather.csv"
        csv_path.write_text(
            "STATION,DATE,PRCP,SNOW\n"
            "USW00094728,2022-01-15,50,0\n"
            "USW00094728,2022-01-16,,\n"
        )
        result = load_weather(spark, str(csv_path))
        rows = {row["DATE"]: row for row in result.collect()}
        assert rows[datetime.date(2022, 1, 15)]["PRCP"] == 50
        assert rows[datetime.date(2022, 1, 16)]["PRCP"] == 0
        assert rows[datetime.date(2022, 1, 16)]["SNOW"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose exec pyspark bash -c "cd /home/jovyan && python -m pytest tests/test_weather.py -v"
```

Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'work.weather_helpers'`.

- [ ] **Step 3: Implement `work/weather_helpers.py`**

```python
# work/weather_helpers.py
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import StructType, StructField, StringType, DateType, IntegerType

WEATHER_SCHEMA = StructType([
    StructField("STATION", StringType()),
    StructField("DATE", DateType()),
    StructField("PRCP", IntegerType()),
    StructField("SNOW", IntegerType()),
])


def load_weather(spark: SparkSession, path: str) -> DataFrame:
    """
    Reads the filtered NOAA GHCN-Daily weather CSV (STATION, DATE, PRCP, SNOW)
    and fills any null PRCP/SNOW with 0 (no precipitation recorded that day).
    """
    return (
        spark.read.csv(path, header=True, schema=WEATHER_SCHEMA)
        .fillna({"PRCP": 0, "SNOW": 0})
    )


def categorize_weather(df: DataFrame) -> DataFrame:
    """
    Adds weather_condition: 'snow' if SNOW > 0, else 'rain' if PRCP > 0,
    else 'clear'. Snow takes priority on days with both.
    """
    return df.withColumn(
        "weather_condition",
        when(col("SNOW") > 0, "snow")
        .when(col("PRCP") > 0, "rain")
        .otherwise("clear")
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose exec pyspark bash -c "cd /home/jovyan && python -m pytest tests/test_weather.py -v"
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add work/weather_helpers.py tests/test_weather.py
git commit -m "feat: add weather_helpers.load_weather/categorize_weather"
```

---

### Task 3: `query_demand_by_weather()` in `work/analytics.py` (TDD)

**Files:**
- Modify: `work/analytics.py`
- Test: `tests/test_weather.py` (append)

**Interfaces:**
- Consumes: `categorize_weather()` from Task 2 (`work/weather_helpers.py`), the existing `pickup_date` column already produced by `work/preprocess_steps.py`/`work/02_preprocess.py`.
- Produces: `query_demand_by_weather(taxi_df: DataFrame, weather_df: DataFrame) -> DataFrame` — columns `weather_condition, trip_count, avg_fare, avg_distance`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_weather.py`:

```python
def make_taxi_row(pickup_date, fare_amount=10.0, trip_distance=2.0):
    return Row(pickup_date=pickup_date, fare_amount=fare_amount, trip_distance=trip_distance)


class TestQueryDemandByWeather:
    def test_joins_and_groups_by_condition(self, spark):
        from work.analytics import query_demand_by_weather
        from work.weather_helpers import categorize_weather

        taxi_df = spark.createDataFrame([
            make_taxi_row(datetime.date(2022, 1, 1), fare_amount=10.0, trip_distance=2.0),
            make_taxi_row(datetime.date(2022, 1, 1), fare_amount=20.0, trip_distance=4.0),
            make_taxi_row(datetime.date(2022, 1, 2), fare_amount=15.0, trip_distance=3.0),
            make_taxi_row(datetime.date(2022, 1, 3), fare_amount=30.0, trip_distance=6.0),
        ])
        weather_df = categorize_weather(spark.createDataFrame([
            make_weather_row(DATE=datetime.date(2022, 1, 1), PRCP=0, SNOW=0),   # clear
            make_weather_row(DATE=datetime.date(2022, 1, 2), PRCP=50, SNOW=0),  # rain
            make_weather_row(DATE=datetime.date(2022, 1, 3), PRCP=0, SNOW=30),  # snow
        ]))

        result = {
            row["weather_condition"]: row
            for row in query_demand_by_weather(taxi_df, weather_df).collect()
        }

        assert result["clear"]["trip_count"] == 2
        assert result["clear"]["avg_fare"] == pytest.approx(15.0)
        assert result["rain"]["trip_count"] == 1
        assert result["rain"]["avg_fare"] == pytest.approx(15.0)
        assert result["snow"]["trip_count"] == 1
        assert result["snow"]["avg_distance"] == pytest.approx(6.0)
```

Add `import pytest` at the top of `tests/test_weather.py` alongside the existing `import datetime` and `from pyspark.sql import Row`.

- [ ] **Step 2: Run test to verify it fails**

```bash
docker compose exec pyspark bash -c "cd /home/jovyan && python -m pytest tests/test_weather.py::TestQueryDemandByWeather -v"
```

Expected: FAIL — `ImportError: cannot import name 'query_demand_by_weather' from 'work.analytics'`.

- [ ] **Step 3: Implement `query_demand_by_weather()`**

In `work/analytics.py`, change the import line at the top from:

```python
from pyspark.sql.functions import col, avg, count, sum as spark_sum, round as spark_round
```

to:

```python
from pyspark.sql.functions import col, avg, count, sum as spark_sum, round as spark_round, broadcast
```

Then append this function to the end of the file:

```python
def query_demand_by_weather(taxi_df: DataFrame, weather_df: DataFrame) -> DataFrame:
    """
    Q5: Trip demand and avg fare/distance by daily weather condition
    (clear/rain/snow). Cross-source join: taxi trips (fact table, millions+
    rows) to NOAA daily weather (dimension table, ~1,461 rows) on
    pickup_date == DATE. weather_df is broadcast rather than shuffled since
    it's tiny relative to the taxi side.
    """
    return (
        taxi_df.join(broadcast(weather_df), taxi_df["pickup_date"] == weather_df["DATE"])
               .groupBy("weather_condition")
               .agg(
                   count("*").alias("trip_count"),
                   spark_round(avg("fare_amount"), 2).alias("avg_fare"),
                   spark_round(avg("trip_distance"), 2).alias("avg_distance"),
               )
               .orderBy("weather_condition")
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose exec pyspark bash -c "cd /home/jovyan && python -m pytest tests/test_weather.py -v"
```

Expected: 6 passed.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```bash
make test
```

Expected: all existing tests (previously 42) plus the 6 new ones pass.

- [ ] **Step 6: Commit**

```bash
git add work/analytics.py tests/test_weather.py
git commit -m "feat: add query_demand_by_weather (Q5 cross-source join)"
```

---

### Task 4: Wire Q5 into the local pipeline and run it

**Files:**
- Modify: `work/constants.py`
- Modify: `work/03_analytics.py`

**Interfaces:**
- Consumes: `WEATHER_PATH` (new constant), `load_weather`/`categorize_weather` (Task 2), `query_demand_by_weather` (Task 3).
- Produces: `results/q5_weather_demand.csv` with columns `weather_condition, trip_count, avg_fare, avg_distance` (checked into git, same as `results/q1_hourly_demand.csv` etc.).

- [ ] **Step 1: Add `WEATHER_PATH` to `work/constants.py`**

Current end of file:

```python
TAXI_PATH   = CLOUD_TAXI_PATH   if IS_CLOUD else LOCAL_TAXI_PATH
OUTPUT_PATH = CLOUD_OUTPUT_PATH if IS_CLOUD else LOCAL_OUTPUT_PATH
```

Change to:

```python
LOCAL_WEATHER_PATH = "/home/jovyan/data/weather_central_park.csv"
CLOUD_WEATHER_PATH = f"s3://{BUCKET}/data/weather_central_park.csv"

TAXI_PATH    = CLOUD_TAXI_PATH    if IS_CLOUD else LOCAL_TAXI_PATH
OUTPUT_PATH  = CLOUD_OUTPUT_PATH  if IS_CLOUD else LOCAL_OUTPUT_PATH
WEATHER_PATH = CLOUD_WEATHER_PATH if IS_CLOUD else LOCAL_WEATHER_PATH
```

- [ ] **Step 2: Wire it into `work/03_analytics.py`**

Change the import block at the top from:

```python
from work.spark_helper import get_spark
from work.constants import OUTPUT_PATH
from work.analytics import (
    query_hourly_demand,
    query_zone_revenue,
    query_tipping_by_distance,
    query_fare_per_mile,
)
```

to:

```python
from work.spark_helper import get_spark
from work.constants import OUTPUT_PATH, WEATHER_PATH
from work.analytics import (
    query_hourly_demand,
    query_zone_revenue,
    query_tipping_by_distance,
    query_fare_per_mile,
    query_demand_by_weather,
)
from work.weather_helpers import load_weather, categorize_weather
```

After the existing Q4 block (`query_fare_per_mile(df).show()`), add:

```python
print("\n── Q5: Trip Demand by Weather Condition ────────────────────────────────")
weather_df = categorize_weather(load_weather(spark, WEATHER_PATH))
query_demand_by_weather(df, weather_df).show()
```

And in the "Persist results" block, after the existing 4 writes, add:

```python
query_demand_by_weather(df, weather_df).write.mode("overwrite").parquet(OUTPUT_PATH + "q5_weather_demand/")
```

- [ ] **Step 3: Run the pipeline locally**

Requires `make up`, `make download-data`, `make download-zones`, `make download-weather`, and `make run-preprocess` to have already been run (existing setup steps from the README, plus the new `download-weather`).

```bash
make download-weather
make run-analytics
```

Expected: console output shows a "Q5: Trip Demand by Weather Condition" section with 3 rows (`clear`, `rain`, `snow`), non-zero `trip_count` for all three (per the earlier spec-time check: Jan 2022 has 20 clear / 6 rain / 5 snow days). Confirm no errors and that the run still prints Q1–Q4 correctly.

- [ ] **Step 4: Convert the parquet output to CSV**

```bash
docker compose exec pyspark bash -c "cd /home/jovyan && python -c \"
import pandas as pd
pd.read_parquet('data/output/q5_weather_demand/').to_csv('results/q5_weather_demand.csv', index=False)
\""
cat results/q5_weather_demand.csv
```

Expected: `results/q5_weather_demand.csv` exists on the host (mounted volume) with header `weather_condition,trip_count,avg_fare,avg_distance` and 3 data rows. Note the actual numbers printed — Task 7 needs them for the UI finding blurb and Task 8 needs them for `analytics_results.md`.

- [ ] **Step 5: Commit**

```bash
git add work/constants.py work/03_analytics.py results/q5_weather_demand.csv
git commit -m "feat: wire Q5 weather join into local analytics pipeline, add results/q5_weather_demand.csv"
```

---

### Task 5: Cloud path parity (code only, not executed)

**Files:**
- Modify: `cloud/03_analytics_cloud.py`
- Modify: `cloud/emr_job_runner.sh`

**Interfaces:**
- Consumes: `load_weather`/`categorize_weather` (Task 2), `query_demand_by_weather` (Task 3) — imported the same flat way the cloud script already imports `analytics.py` (EMR uploads `work/*.py` files flatly via `--py-files`, so cloud scripts import them as top-level modules, not as `work.foo`).
- Produces: no runtime output this round — this task only makes the cloud script buildable/runnable later.

- [ ] **Step 1: Update `cloud/03_analytics_cloud.py`**

Current file:

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

Replace it with:

```python
# cloud/03_analytics_cloud.py
import os, sys
sys.path.insert(0, "/home/hadoop/")

from pyspark.sql import SparkSession
from analytics import (
    query_hourly_demand, query_zone_revenue,
    query_tipping_by_distance, query_fare_per_mile, query_demand_by_weather,
)
from weather_helpers import load_weather, categorize_weather

spark = SparkSession.builder.appName("cs675-analytics-cloud").getOrCreate()

BUCKET  = os.environ["CS675_BUCKET"]
IN      = f"s3://{BUCKET}/output/taxi_clean/"
OUT     = f"s3://{BUCKET}/output/results/"
WEATHER = f"s3://{BUCKET}/data/weather_central_park.csv"

print("Reading clean taxi data from S3...")
df = spark.read.parquet(IN)
print(f"  Rows: {df.count():,}")

query_hourly_demand(df).write.mode("overwrite").parquet(OUT + "q1/")
query_zone_revenue(df).write.mode("overwrite").parquet(OUT + "q2/")
query_tipping_by_distance(df).write.mode("overwrite").parquet(OUT + "q3/")
query_fare_per_mile(df).write.mode("overwrite").parquet(OUT + "q4/")

weather_df = categorize_weather(load_weather(spark, WEATHER))
query_demand_by_weather(df, weather_df).write.mode("overwrite").parquet(OUT + "q5/")

print("Cloud analytics complete. Results at:", OUT)
spark.stop()
```

- [ ] **Step 2: Update `cloud/emr_job_runner.sh` to upload `weather_helpers.py`**

Current upload block:

```bash
aws s3 cp "cloud/${SCRIPT}.py"        "s3://${BUCKET}/scripts/${SCRIPT}.py"        --profile "${PROFILE}"
aws s3 cp "work/preprocess_steps.py"  "s3://${BUCKET}/scripts/preprocess_steps.py" --profile "${PROFILE}"
aws s3 cp "work/analytics.py"         "s3://${BUCKET}/scripts/analytics.py"        --profile "${PROFILE}"
aws s3 cp "work/ml_features.py"       "s3://${BUCKET}/scripts/ml_features.py"      --profile "${PROFILE}"
```

Change to:

```bash
aws s3 cp "cloud/${SCRIPT}.py"        "s3://${BUCKET}/scripts/${SCRIPT}.py"        --profile "${PROFILE}"
aws s3 cp "work/preprocess_steps.py"  "s3://${BUCKET}/scripts/preprocess_steps.py" --profile "${PROFILE}"
aws s3 cp "work/analytics.py"         "s3://${BUCKET}/scripts/analytics.py"        --profile "${PROFILE}"
aws s3 cp "work/ml_features.py"       "s3://${BUCKET}/scripts/ml_features.py"      --profile "${PROFILE}"
aws s3 cp "work/weather_helpers.py"   "s3://${BUCKET}/scripts/weather_helpers.py"  --profile "${PROFILE}"
```

And in the `--py-files` value inside `sparkSubmitParameters`, change:

```
--py-files s3://${BUCKET}/scripts/preprocess_steps.py,s3://${BUCKET}/scripts/analytics.py,s3://${BUCKET}/scripts/ml_features.py
```

to:

```
--py-files s3://${BUCKET}/scripts/preprocess_steps.py,s3://${BUCKET}/scripts/analytics.py,s3://${BUCKET}/scripts/ml_features.py,s3://${BUCKET}/scripts/weather_helpers.py
```

- [ ] **Step 3: Verify by reading, not running**

There is no cloud infra up this round (torn down in the prior session), so this cannot be executed end-to-end. Verify correctness by inspection:
- `cloud/03_analytics_cloud.py` imports match the flat (non-`work.`) module names EMR will see, consistent with the existing `from analytics import ...` line already in the file.
- `emr_job_runner.sh`'s upload list and `--py-files` list both include `weather_helpers.py`, matching the pattern of the other 3 shared modules.

- [ ] **Step 4: Commit**

```bash
git add cloud/03_analytics_cloud.py cloud/emr_job_runner.sh
git commit -m "feat: add Q5 weather join to cloud analytics path (not executed this round)"
```

---

### Task 6: Streamlit UI — Q5 section

**Files:**
- Modify: `ui/analytics_section.py`

**Interfaces:**
- Consumes: `results/q5_weather_demand.csv` (Task 4), the actual numbers observed when that file was generated.

- [ ] **Step 1: Read the actual Q5 numbers**

```bash
cat results/q5_weather_demand.csv
```

Use these real values (not placeholders) in the finding sentence written in Step 2.

- [ ] **Step 2: Update `_load_results()`**

Change:

```python
@st.cache_data
def _load_results():
    hourly = pd.read_csv("results/q1_hourly_demand.csv")
    zones = pd.read_csv("results/q2_zone_revenue.csv")
    tipping = pd.read_csv("results/q3_tipping_by_distance.csv")
    fare_per_mile = pd.read_csv("results/q4_fare_per_mile.csv")
    zone_lookup = pd.read_csv("data/taxi_zone_lookup.csv")
    return hourly, zones, tipping, fare_per_mile, zone_lookup
```

to:

```python
@st.cache_data
def _load_results():
    hourly = pd.read_csv("results/q1_hourly_demand.csv")
    zones = pd.read_csv("results/q2_zone_revenue.csv")
    tipping = pd.read_csv("results/q3_tipping_by_distance.csv")
    fare_per_mile = pd.read_csv("results/q4_fare_per_mile.csv")
    weather = pd.read_csv("results/q5_weather_demand.csv")
    zone_lookup = pd.read_csv("data/taxi_zone_lookup.csv")
    return hourly, zones, tipping, fare_per_mile, weather, zone_lookup
```

And update the unpacking call:

```python
    try:
        hourly, zones, tipping, fare_per_mile, zone_lookup = _load_results()
```

to:

```python
    try:
        hourly, zones, tipping, fare_per_mile, weather, zone_lookup = _load_results()
```

- [ ] **Step 3: Add the Q5 subsection**

After the existing Q4 block (ends with `st.dataframe(fare_ordered, use_container_width=True)`), append. Replace `<clear count>`, `<rain count>`, `<snow count>`, `<clear avg_fare>` below with the real values read in Step 1 (they must be filled in with actual numbers, not left as placeholders):

```python
    st.subheader("Q5 — Demand by weather condition")
    st.write(
        "Joined against NOAA daily weather for NYC (Central Park station): "
        "<clear count> trips on clear days vs. <rain count> on rainy days and "
        "<snow count> on snowy days in the Jan 2022 local sample, with average "
        "fare of $<clear avg_fare> on clear days."
    )
    weather_bar = (
        alt.Chart(weather)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X("weather_condition:N", sort=["clear", "rain", "snow"], title="Weather condition"),
            y=alt.Y("trip_count:Q", title="Trips"),
            tooltip=["weather_condition", "trip_count", "avg_fare", "avg_distance"],
        )
        .properties(height=260)
    )
    st.altair_chart(weather_bar, use_container_width=True)
    st.dataframe(weather, use_container_width=True)
```

- [ ] **Step 4: Update the missing-file warning path**

The existing `except FileNotFoundError` block's message already says "Analytics results not found under `results/`" generically — confirm it still applies now that `_load_results()` reads a 5th results file (no wording change needed since it doesn't enumerate filenames).

- [ ] **Step 5: Manually verify in the browser**

```bash
make run-ui
```

Open http://localhost:8501, scroll past the prediction form and Q1–Q4, confirm the new "Q5 — Demand by weather condition" section renders a 3-bar chart (clear/rain/snow order) and a 3-row table with no errors.

- [ ] **Step 6: Commit**

```bash
git add ui/analytics_section.py
git commit -m "feat: show Q5 weather-join results in Streamlit UI"
```

---

### Task 7: Documentation

**Files:**
- Modify: `README.md`
- Modify: `analytics_results.md`

**Interfaces:**
- Consumes: the real numbers in `results/q5_weather_demand.csv` (Task 4).

- [ ] **Step 1: Update README's research-questions list**

In the "Overview" section, change the 4-item numbered list to add a 5th item:

```markdown
5. Does trip demand or average fare change on rainy or snowy days? (joined against NOAA daily weather — the project's cross-source join)
```

- [ ] **Step 2: Add a "Cross-Source Join: Weather (Q5)" section to README**

Add after the existing "Preprocessing: Before → After" section (before "## Data Source"):

```markdown
## Cross-Source Join: Weather (Q5)

`work/weather_helpers.py` loads NOAA GHCN-Daily daily weather for NYC (Central Park station `USW00094728`) and buckets each day into `clear` / `rain` / `snow`. `query_demand_by_weather()` in `work/analytics.py` joins this against the cleaned taxi trips on `pickup_date`, broadcasting the weather side (a few hundred to ~1,461 rows) against the taxi fact table (millions of rows) to avoid a shuffle — a standard big-data join pattern for a large-fact/small-dimension join.

Run locally with `make download-weather` before `make run-analytics`. Results: [`results/q5_weather_demand.csv`](results/q5_weather_demand.csv), discussed in [`analytics_results.md`](analytics_results.md).
```

- [ ] **Step 3: Add the weather source to README's "Data Source" section**

Add a third bullet after the existing zone lookup line:

```markdown
- Daily weather (used by `scripts/download_weather.sh`): NOAA GHCN-Daily, NYC Central Park station (`USW00094728`) — `https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00094728.csv`
```

And update the sentence right below the bullet list from:

```markdown
`make download-data` / `make download-zones` (local) and `scripts/download_full_data.sh` (cloud) fetch these directly — no manual download needed.
```

to:

```markdown
`make download-data` / `make download-zones` / `make download-weather` (local) and `scripts/download_full_data.sh` (cloud) fetch these directly — no manual download needed.
```

- [ ] **Step 4: Add `download-weather` to the Local Setup steps**

In the "Local Setup" section's step 2 code block, change:

```bash
make download-data    # one month (2022-01) of Yellow Taxi trip data, for local dev
make download-zones   # taxi zone lookup table (LocationID -> Borough/Zone)
```

to:

```bash
make download-data    # one month (2022-01) of Yellow Taxi trip data, for local dev
make download-zones   # taxi zone lookup table (LocationID -> Borough/Zone)
make download-weather # NOAA daily weather for NYC (Central Park station), for Q5
```

- [ ] **Step 5: Add a Q5 section to `analytics_results.md`**

Using the real values from `results/q5_weather_demand.csv` (read in Task 6 Step 1), append a new section following the existing Q1–Q4 style, explicitly labeled as a local-sample result (unlike Q1–Q4, which are cloud-scale):

```markdown
## Q5 — Demand by Weather Condition (local sample)

`results/q5_weather_demand.csv` — computed on the **local Jan 2022 sample** (not the full cloud-scale run), joined against NOAA daily weather for NYC Central Park.

| Weather condition | Trip count | Avg fare | Avg distance |
|---|---|---|---|
| clear | <value> | $<value> | <value> mi |
| rain | <value> | $<value> | <value> mi |
| snow | <value> | $<value> | <value> mi |

<One or two sentences of interpretation, written from the actual numbers — e.g. whether demand drops on snow days, whether average fare/distance shifts.>
```

Replace every `<value>` and the interpretation sentence with the real numbers and an honest read of them — do not leave placeholders in the committed file.

- [ ] **Step 6: Commit**

```bash
git add README.md analytics_results.md
git commit -m "docs: document Q5 weather join in README and analytics_results.md"
```

---

## Self-Review Notes

- **Spec coverage:** download script (Task 1) ✓, `weather_helpers.py` (Task 2) ✓, `query_demand_by_weather` (Task 3) ✓, local wiring + results CSV (Task 4) ✓, cloud parity not executed (Task 5) ✓, UI section (Task 6) ✓, README + `analytics_results.md` (Task 7) ✓. Spec's "explicitly out of scope" items (infra reprovisioning, `02_preprocess.py`/`taxi_clean/`/ML changes, `predict_app.py` changes) are not touched by any task.
- **Placeholder scan:** the only bracketed placeholders (`<value>`, `<clear count>`, etc.) appear inside explicit instructions to replace them with real numbers observed during Task 4/6, not as unresolved plan content.
- **Type/name consistency:** `load_weather(spark, path)`, `categorize_weather(df)`, `query_demand_by_weather(taxi_df, weather_df)` — same names and signatures used consistently across Tasks 2–6. Column names (`weather_condition`, `trip_count`, `avg_fare`, `avg_distance`) match between Task 3's implementation, Task 4's CSV, and Task 6's UI code.
