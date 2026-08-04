# CS-675 Final Project — Scope Review
**NYC Yellow Taxi Analytics at Cloud Scale**

---

## Dataset

- **NYC TLC Yellow Taxi** (2019–2022)
- ~180 million trips · 19 columns · Parquet format
- Public data from TLC open data portal
- Local dev: Jan 2022 slice (~2.5M rows, 37 MB)

### Raw Columns (19 total)

| Column | Type | Description |
|--------|------|-------------|
| `tpep_pickup_datetime` | timestamp | Trip start time |
| `tpep_dropoff_datetime` | timestamp | Trip end time |
| `passenger_count` | float | Number of passengers (2.9% null) |
| `trip_distance` | float | Distance in miles |
| `PULocationID` | int | Pickup zone ID (1–265) |
| `DOLocationID` | int | Dropoff zone ID (1–265) |
| `payment_type` | int | 1=credit card, 2=cash, 3=no charge, 4=dispute |
| `fare_amount` | float | Metered fare |
| `tip_amount` | float | Tip (only populated for credit card payments) |
| `total_amount` | float | Fare + tip + tolls + surcharges |
| `congestion_surcharge` | float | NYC congestion pricing fee (2.9% null) |
| `airport_fee` | float | JFK/LGA airport fee (2.9% null) |
| `VendorID`, `RatecodeID`, `extra`, `mta_tax`, `tolls_amount`, `improvement_surcharge`, `store_and_fwd_flag` | various | Vendor/tax metadata |

### Key Data Facts (Jan 2022 sample)

| Stat | Value |
|------|-------|
| Total trips | 2,463,931 |
| Median trip distance | 1.74 miles |
| Trips under 3 miles | 72.6% |
| Avg fare | $12.95 |
| Avg total amount | $19.17 |
| Credit card payments | 76.1% |
| Cash payments | 20.1% |
| Peak demand hours | 5pm – 7pm |
| Rows with distance = 0 | 29,373 (1.2%) — dropped |
| Rows with negative fare | 12,733 (0.5%) — dropped |

---

## Research Questions

1. How does trip demand vary by hour of day and day of week?
2. Which pickup zones generate the most revenue?
3. Do longer trips earn higher tip percentages?
4. Do short trips cost more per mile than long trips?

---

## Preprocessing Pipeline (Step 00)

| Step | What | Why |
|------|------|-----|
| Imputation | Fill null `passenger_count` → 1; drop null `fare_amount` | Solo rider is the mode; fare can't be imputed |
| Outlier removal | Drop distance ≤ 0; cap fare at $500 | Physical impossibility; >$500 is data entry error |
| Normalization | Min-max scale fare and distance to [0, 1] | Enables year-over-year comparison |
| Encoding | One-hot encode `payment_type` (credit/cash/no-charge) | Converts categorical to numeric |
| Binning | Distance → short/medium/long; hour → time-of-day segment | Groups trips into meaningful behavioral buckets |

---

## Analytics Queries (Step 00)

### Q1 — Trip Demand by Hour and Day of Week

**Columns used:** `tpep_pickup_datetime` → derived `pickup_hour`, `day_of_week`

```sql
SELECT
    HOUR(tpep_pickup_datetime)      AS pickup_hour,
    DAY_OF_WEEK(tpep_pickup_datetime) AS day_of_week,
    COUNT(*)                        AS trip_count
FROM yellow_taxi
GROUP BY 1, 2
ORDER BY 1, 2;
```

---

### Q2 — Revenue by Pickup Zone

**Columns used:** `PULocationID`, `fare_amount`

```sql
SELECT
    PULocationID,
    ROUND(SUM(fare_amount), 2)  AS total_revenue,
    ROUND(AVG(fare_amount), 2)  AS avg_fare,
    COUNT(*)                    AS trip_count
FROM yellow_taxi
GROUP BY PULocationID
ORDER BY total_revenue DESC
LIMIT 20;
```

---

### Q3 — Tipping Behavior by Distance Bucket

**Columns used:** `trip_distance` → derived `distance_bucket`, `tip_amount`, `fare_amount`

```sql
SELECT
    CASE
        WHEN trip_distance < 1  THEN 'short'
        WHEN trip_distance <= 10 THEN 'medium'
        ELSE 'long'
    END                                           AS distance_bucket,
    ROUND(AVG(tip_amount / fare_amount * 100), 2) AS avg_tip_pct,
    ROUND(AVG(tip_amount), 2)                     AS avg_tip_amount,
    COUNT(*)                                      AS trip_count
FROM yellow_taxi
WHERE fare_amount > 0
GROUP BY 1
ORDER BY 1;
```

---

### Q4 — Fare Per Mile by Distance Bucket

**Columns used:** `trip_distance` → derived `distance_bucket`, `fare_amount`

```sql
SELECT
    CASE
        WHEN trip_distance < 1  THEN 'short'
        WHEN trip_distance <= 10 THEN 'medium'
        ELSE 'long'
    END                                              AS distance_bucket,
    ROUND(AVG(fare_amount / trip_distance), 2)       AS avg_fare_per_mile,
    ROUND(AVG(fare_amount), 2)                       AS avg_fare,
    ROUND(AVG(trip_distance), 2)                     AS avg_distance,
    COUNT(*)                                         AS trip_count
FROM yellow_taxi
WHERE trip_distance > 0
GROUP BY 1
ORDER BY 1;
```

---

## Cloud Setup (Step 01)

| Component | Tool | Purpose |
|-----------|------|---------|
| Storage | AWS S3 | Stores all 48 Parquet files (2019–2022) |
| SQL queries | AWS Athena + Glue catalog | Interactive analysis over S3 |
| Batch compute | AWS EMR Serverless | Runs full PySpark pipeline at 180M-row scale |
| Infrastructure | Terraform | Reproducible provisioning + `terraform destroy` for cleanup |

**Local → Cloud:** Same PySpark scripts, just swap local paths for `s3://` paths.

---

## ML Extension (Step 02 — +10 bonus points)

**Goal:** Predict `total_amount` (total revenue per trip) using PySpark MLlib.

| | Model | Purpose |
|-|-------|---------|
| Baseline | Linear Regression | Simple interpretable benchmark |
| Main | Gradient Boosted Trees | Captures non-linear zone + time patterns |

**Features:** trip distance, pickup hour, day of week, passenger count, pickup zone, dropoff zone, payment type

**Evaluation:** RMSE (dollars) + R² on 20% held-out test set. Compare baseline vs GBT.

---

## Timeline

| Date | Milestone |
|------|-----------|
| Jul 25 (today) | Scope review ✓ |
| Jul 26 | Preprocessing pipeline + unit tests done |
| Jul 27 | 4 analytics queries + unit tests done |
| Jul 28 | ML model (LR + GBT) locally trained and evaluated |
| Jul 28 | Terraform infra applied, data uploaded to S3 |
| Jul 29–30 | EMR Serverless cloud run (analytics + ML), results collected |
| Jul 31 | README + results discussion finalized |
| Aug 1 | **Final demo** |

---

## Tools Used (satisfies ≥ 2 course tools)

- **PySpark** — local preprocessing and analytics
- **AWS Athena** — SQL over S3
- **AWS EMR Serverless** — cloud-scale PySpark execution
- **Terraform** — reproducible infrastructure

---

## Repo

`https://github.com/oomintrixx/cs675`
