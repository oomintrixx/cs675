# Weather Join (Q5) — Design Spec

**Date:** 2026-08-04
**Feature:** Join NOAA daily weather data into the taxi analytics pipeline and add a 5th research question — does demand/fare change on rainy or snowy days? — satisfying the assignment's "cross-source join analytics" requirement (Step 00, 20 points) and giving the project a second joinable dataset.

---

## Goal

Add one new analytics query, `query_demand_by_weather()`, that joins Yellow Taxi trips to NOAA GHCN-Daily weather records by pickup date, grouped into `clear` / `rain` / `snow`, and shows trip count / avg fare / avg distance per bucket. Ship it through the same local pipeline (`work/`), the same cloud entry point (`cloud/`, code-complete but not executed this round), and the Streamlit UI, following the existing Q1–Q4 pattern exactly.

---

## Data source

**NOAA GHCN-Daily**, station `USW00094728` (NY City Central Park, the standard NYC reference station).

- Full station history (1869–present, ~57.5k rows, 17.8MB): `https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00094728.csv`
- Verified reachable (HTTP 200) and columns confirmed: `STATION, DATE, LATITUDE, LONGITUDE, ELEVATION, NAME, PRCP, PRCP_ATTRIBUTES, SNOW, SNOW_ATTRIBUTES, ...` (plus many mostly-empty optional fields).
- `PRCP` = daily precipitation in tenths of mm; `SNOW` = daily snowfall in mm. Both are `0` (not null) on dry days in this dataset; nulls do occur on some historical rows and are treated as `0`.
- Verified sample: Jan 2022 (the local dev slice's date range) has 20 clear / 6 rain / 5 snow days out of 31 — including the Jan 29, 2022 NYC blizzard — so the local demo has a meaningful class balance, not a degenerate single-bucket result.

### `scripts/download_weather.sh` (new)

Mirrors `scripts/download_zone_lookup.sh`:

```bash
#!/bin/bash
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

Output: `data/weather_central_park.csv`, filtered to 2019–2022 (matches the taxi dataset's year range), four columns only. A few hundred KB, ~1,461 rows. Not committed — same treatment as `data/taxi_zone_lookup.csv` and the raw taxi Parquet, fetched on demand, already covered by the README's "Data Source" section (add one line for this new source + script).

`Makefile`: add a `download-weather` target calling this script, alongside the existing `download-data` / `download-zones`.

---

## New files / changes

### `work/weather_helpers.py` (new)

```python
def load_weather(spark, path: str) -> DataFrame:
    """Reads the filtered weather CSV, casts DATE to DateType and PRCP/SNOW
    to int (nulls -> 0)."""

def categorize_weather(df: DataFrame) -> DataFrame:
    """Adds weather_condition: 'snow' if SNOW > 0, else 'rain' if PRCP > 0,
    else 'clear'. Snow takes priority on days with both (matches how a
    rider/driver would describe the day)."""
```

Same `when/otherwise` style as `preprocess_steps.bin_distance()`.

### `work/analytics.py` (add one function)

```python
def query_demand_by_weather(taxi_df: DataFrame, weather_df: DataFrame) -> DataFrame:
    """
    Q5: Trip demand and avg fare/distance by daily weather condition
    (clear/rain/snow). Cross-source join: taxi trips (fact) to NOAA daily
    weather (dimension) on pickup_date == DATE.
    """
    return (
        taxi_df.join(broadcast(weather_df), taxi_df.pickup_date == weather_df.DATE)
               .groupBy("weather_condition")
               .agg(
                   count("*").alias("trip_count"),
                   spark_round(avg("fare_amount"), 2).alias("avg_fare"),
                   spark_round(avg("trip_distance"), 2).alias("avg_distance"),
               )
               .orderBy("weather_condition")
    )
```

`broadcast()` on the weather side is a deliberate big-data technique: the weather table is tiny (≤1,461 rows) relative to the taxi fact table (millions/hundreds of millions of rows), so broadcasting it avoids a shuffle join — worth calling out in the README next to this query, matching the assignment's interest in "big-data techniques."

Existing `work/preprocess_steps.py`, `work/02_preprocess.py`, and `taxi_clean/` are untouched — the join happens only at query time (Approach A, chosen over joining during preprocessing or a standalone join script, to keep this isolated from the already-working, already-tested pipeline).

### `work/03_analytics.py` (modified)

After the existing Q1–Q4 block: load `data/weather_central_park.csv` via `load_weather()` + `categorize_weather()`, call `query_demand_by_weather(df, weather_df)`, `.show()`, and write to `results/q5_weather_demand/` (parquet, same as Q1–Q4) → checked in as `results/q5_weather_demand.csv` (same as the others).

### `cloud/03_analytics_cloud.py` (modified, not executed this round)

Same addition, reading the weather CSV from `s3://{BUCKET}/data/weather_central_park.csv`. This keeps local/cloud parity (the project's existing "same PySpark logic, cloud is a path swap" principle) but per the scope decision below, **this path is written but not run** — no `terraform apply`, no re-upload, no EMR job this round.

### `ui/analytics_section.py` (modified)

One new subsection after Q4, following the existing pattern (`@st.cache_data` loader addition for `results/q5_weather_demand.csv`, one Altair bar chart: x=`weather_condition` [clear/rain/snow order], y=`trip_count`, with `avg_fare` shown alongside), plus a one-sentence finding blurb once real numbers are in. Same missing-file warning behavior as the existing Q1–Q4 loader — if `results/q5_weather_demand.csv` is absent, warn and skip rather than crash.

### `tests/test_weather.py` (new)

Same style as `tests/test_analytics.py` (uses the shared `spark` fixture from `conftest.py`):

- `load_weather()`: tiny synthetic CSV → correct dtypes, null PRCP/SNOW → 0
- `categorize_weather()`: rows covering all 3 buckets + the snow-and-rain-same-day case → correct priority (snow wins)
- `query_demand_by_weather()`: a handful of synthetic taxi rows across 3 dates joined to 3 synthetic weather rows (one per condition) → group-by counts/averages match hand-computed expected values

---

## Documentation updates

- **README.md**: add Q5 to the research-questions list in Overview; add a short subsection (near "Preprocessing: Before → After") describing the weather source, the join, and the broadcast-join rationale; add the weather CSV source URL to the "Data Source" section; add `make download-weather` to the setup steps.
- **`analytics_results.md`**: add a Q5 section once the local run produces real numbers (mirrors the Q1–Q4 write-up style: headline table + 1-2 sentence interpretation). This will be the **local Jan 2022 sample** result, explicitly labeled as such (not a cloud/full-scale number) — consistent with how local vs. cloud results are already distinguished elsewhere in this file.
- `plan.md`, `presentation.md`, and `CS675_Presentation.pptx` are **not** touched by this feature — those are existing, already-corrected deliverables; updating them for Q5 is out of scope unless requested separately.

---

## Explicitly out of scope this round

- Re-provisioning AWS infra (`terraform apply`), re-uploading the 48-month dataset, or running an EMR Serverless job for Q5 at full cloud scale. The cloud code path is written for parity but stays unexecuted; resources remain torn down from the earlier `terraform destroy`. Running Q5 at cloud scale is a clear follow-up if/when full-scale results are needed again.
- Changes to `work/02_preprocess.py`, `taxi_clean/`, or the ML feature pipeline (`work/ml_features.py`) — weather is not added as an ML feature in this round.
- Changes to `ui/predict_app.py`'s prediction form.

---

## Testing / verification plan

1. `tests/test_weather.py` — new unit tests, run via `make test`.
2. `make run-explore` / existing tests unaffected — verify full `make test` suite (42 existing + new) still passes.
3. `make download-weather` then `make run-analytics` locally — confirm Q5 prints a 3-row (clear/rain/snow) table with non-degenerate counts (per the Jan 2022 sample already checked: 20/6/5-day split) and writes `results/q5_weather_demand.csv`.
4. `make run-ui` — confirm the new Q5 chart renders under the existing Q4 section with no errors and no missing-file warning.
