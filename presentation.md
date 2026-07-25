# CS-675 Final Project — Scope Review
**NYC Yellow Taxi Analytics at Cloud Scale**

---

## Dataset

- **NYC TLC Yellow Taxi** (2019–2022)
- ~260 million trips
- Public Parquet files from TLC open data portal
- Local dev: Jan 2022 slice (~3M rows) for fast iteration

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

| # | Query | Insight |
|---|-------|---------|
| Q1 | Trip count by hour × day of week | Identifies commute peaks vs weekend leisure patterns |
| Q2 | Revenue and trip count by pickup zone | Shows highest-value zones for drivers |
| Q3 | Average tip % by distance bucket | Tests whether longer trips earn proportionally more |
| Q4 | Fare per mile by distance bucket | Quantifies base-fare effect on short trips |

---

## Cloud Setup (Step 01)

| Component | Tool | Purpose |
|-----------|------|---------|
| Storage | AWS S3 | Stores all 48 Parquet files (2019–2022) |
| SQL queries | AWS Athena + Glue catalog | Interactive analysis over S3 |
| Batch compute | AWS EMR Serverless | Runs full PySpark pipeline at 260M-row scale |
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
