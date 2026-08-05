# Final Project Requirements — Coverage Checklist

Maps every item in the [CS-675 final project spec](https://github.com/tkr22777/teaching-cs-675/blob/main/projects/final-project.md) to what's actually in this repo, with file references. Written for grading — each row says exactly where to look.

## Required work (100 pts)

| # | Requirement | Points | Status | Evidence |
|---|---|---|---|---|
| 1 | Preprocessing pipeline (imputation, outlier treatment, normalization, encoding, binning, with before/after justification) | 10 | ✅ | [`work/preprocess_steps.py`](work/preprocess_steps.py) — 6 transforms. Before/after numbers (measured, not estimated) in README's [Preprocessing: Before → After](README.md#preprocessing-before--after) section. |
| 2 | Cross-source join analytics (queries joining across data sources) | 20 | ✅ | Q5: `query_demand_by_weather()` in [`work/analytics.py`](work/analytics.py) — broadcast-joins 177M+ Yellow Taxi trips against 4 years of NOAA GHCN-Daily weather on `pickup_date`. Run at full cloud scale; results in [`results/q5_weather_demand.csv`](results/q5_weather_demand.csv), discussed in [`analytics_results.md`](analytics_results.md#q5--demand-by-weather-condition). README section: [Cross-Source Join: Weather (Q5)](README.md#cross-source-join-weather-q5). |
| 3 | Cloud infrastructure (storage + query/compute engine, reproducible from repo) | 20 | ✅ | [`infrastructure/*.tf`](infrastructure/) — S3, Athena workgroup, Glue catalog, EMR Serverless app, IAM, all via `terraform apply`/`terraform destroy`. Actually provisioned and torn down twice during development (not just written, verified working). |
| 4 | Big data at real scale (≥100M rows, ≥2 joinable datasets) | 20 | ✅ | 177,171,999 cleaned trips (Yellow Taxi 2019–2022) joined against NOAA daily weather, computed on EMR Serverless — see headline numbers in [`analytics_results.md`](analytics_results.md#headline-numbers). |
| 5 | Code quality, repo organization, README, run instructions | 10 | ✅ | [`README.md`](README.md) — setup, cloud deploy, data sources, project structure, env vars. 50 pytest unit tests (`tests/`). Modular `work/` (shared logic) / `cloud/` (S3 entry points) / `ui/` (Streamlit) / `infrastructure/` (Terraform) split. |
| 6 | Results evaluation & discussion + final presentation with Q&A | 20 | ✅ | [`analytics_results.md`](analytics_results.md) — narrative discussion for all 5 queries. [`CS675_Presentation.pptx`](CS675_Presentation.pptx) — 13-slide deck (background, architecture, deployment, Q1–Q5 analytics, ML extension, wrap-up, Q&A), presented Aug 1, 2026. |

**Required subtotal: 100/100**

## Optional extensions (10 pts each, Step 02)

| Extension | Points | Status | Evidence |
|---|---|---|---|
| Machine learning model | 10 | ✅ | [`work/04_ml.py`](work/04_ml.py) / [`work/ml_features.py`](work/ml_features.py) — Linear Regression, Random Forest, and Gradient Boosted Trees compared on `total_amount` prediction (Random Forest wins, RMSE $4.44 vs. $13.22 baseline). Trained models committed at [`data/output/models/`](data/output/models/). Write-up in [`ml.md`](ml.md). |
| User interface / dashboard | 10 | ✅ | Streamlit app — [`ui/predict_app.py`](ui/predict_app.py) (live fare prediction) + [`ui/analytics_section.py`](ui/analytics_section.py) (Q1–Q5 tables/charts on the same page). Screenshot in [README](README.md#run-the-prediction-ui). |
| Query-performance tuning | 10 | ⚠️ partial | Glue table partitioned by `year`/`month` ([`infrastructure/glue_taxi.tf`](infrastructure/glue_taxi.tf)); Q5's join explicitly broadcasts the small weather side against the 177M-row taxi fact table to avoid a shuffle (`work/analytics.py`). Not the project's primary focus — not claiming full credit here, listed for transparency. |
| Terabyte-scale data | 10 | ❌ not attempted | Dataset is ~180M rows / a few GB, not TB-scale. |
| Real-time ingestion | 10 | ❌ not attempted | Batch pipeline only (EMR Serverless jobs on static S3 data). |

**Optional subtotal claimed: 20/50** (ML + UI; query-tuning listed as partial evidence, not claimed as full credit)

## Guidelines checklist

| Guideline | Status | Evidence |
|---|---|---|
| Public GitHub repo, committed incrementally | ✅ | [github.com/oomintrixx/cs675](https://github.com/oomintrixx/cs675) — full commit history, incremental throughout. |
| July 25 scope & plan | ✅ | [`plan.md`](plan.md), [`presentation.md`](presentation.md) (scope-review script). |
| Aug 1 slide deck + live demo | ✅ | [`CS675_Presentation.pptx`](CS675_Presentation.pptx). |
| Individual work | ✅ | Solo project. |
| ≥2 course tools used | ✅ | PySpark, AWS Athena, AWS EMR Serverless, Terraform (4 tools, exceeds the minimum). |
| Tear down cloud resources when concluded | ✅ | `terraform destroy` run and verified (empty state, no S3 bucket) after each cloud-scale run — most recently after the Q5 full-scale run. |

## Notes for the grader

- Q1–Q4 and Q5 were **both** run at full cloud scale (177M+ trips) on 2026-08-04 — not a local-only approximation. `analytics_results.md` also documents an earlier local-sample-only version of Q5 for transparency, and explains why the two runs disagreed (a 31-day local sample isn't a reliable signal; the full 4-year join reverses the local sample's fare/distance conclusion).
- All numbers in this repo are measured from actual runs, not estimated — including the dataset size itself: an earlier planning estimate of "~260M rows" was corrected to the measured ~180M after reading the TLC source Parquet files' own row-count metadata directly (see README's Preprocessing section).
- Cloud infrastructure currently does **not** exist (torn down per the guidelines) — `terraform apply` reproduces it from [`infrastructure/`](infrastructure/) in about a minute.
