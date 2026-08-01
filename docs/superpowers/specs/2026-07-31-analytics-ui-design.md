# Analytics Results in UI — Design Spec

**Date:** 2026-07-31
**Feature:** Display the 4 analytics query results on the Streamlit prediction page

---

## Goal

Show the 4 analytics queries (hourly demand, zone revenue, tipping by distance, fare per mile) as tables + charts on the same Streamlit page as the `total_amount` prediction form, so the app demonstrates both the ML model and the analytics findings in one place.

---

## Data source

`results/*.csv` — the checked-in CSV outputs from the full-scale cloud run (2019–2022, ~177M trips), documented narratively in `analytics_results.md`:

| File | Columns |
|---|---|
| `results/q1_hourly_demand.csv` | `pickup_hour, day_of_week, trip_count` (168 rows) |
| `results/q2_zone_revenue.csv` | `PULocationID, total_revenue, avg_fare, trip_count` (264 rows, already sorted by `total_revenue` desc) |
| `results/q3_tipping_by_distance.csv` | `distance_bucket, avg_tip_pct, avg_tip_amount, trip_count` (3 rows, alphabetical: long, medium, short) |
| `results/q4_fare_per_mile.csv` | `distance_bucket, avg_fare_per_mile, avg_fare, avg_distance, trip_count` (3 rows, alphabetical: long, medium, short) |

Read with plain pandas — no Spark session needed for this static, already-aggregated data.

`results/` is not currently mounted into the `pyspark` Docker service. Add `./results:/home/jovyan/results` to the `pyspark` service's `volumes` in `docker-compose.yml`.

---

## New files

### `work/analytics_helpers.py`

Pure functions, no Spark/Streamlit/Altair imports — unit-testable with plain pandas:

```python
def order_distance_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Reindexes a distance_bucket-keyed DataFrame into short/medium/long order
    (the CSVs come out alphabetical: long, medium, short)."""

def top_zones_by_revenue(zone_revenue_df: pd.DataFrame, zone_lookup_df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Joins PULocationID -> Zone/Borough from the TLC zone lookup, returns the
    top n rows by total_revenue with an added 'zone_label' column
    formatted '{Zone}, {Borough}'."""

def pivot_hourly_demand(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Pivots (pickup_hour, day_of_week, trip_count) into a 7x24 matrix:
    rows = day names (Sunday..Saturday, matching Spark's dayofweek() 1-7),
    columns = pickup_hour (0-23), values = trip_count. Feeds both the heatmap
    and the table."""
```

### `tests/test_analytics_helpers.py`

Unit tests for all three functions above (no `spark` fixture — pure pandas, same style as `tests/test_predict_helpers.py`).

### `ui/analytics_section.py`

```python
def render_analytics_section() -> None:
    """Reads results/*.csv + data/taxi_zone_lookup.csv, calls the
    analytics_helpers functions, and renders 4 subsections (Q1-Q4) each with
    a short finding blurb (adapted from analytics_results.md), a table, and
    an Altair chart. If any results/*.csv is missing, shows st.warning and
    skips the section instead of crashing the page."""
```

Data loading (`pd.read_csv` for the 4 result files + the zone lookup) is wrapped in `@st.cache_data` so it only happens once per Streamlit session, not on every form rerun.

### `ui/predict_app.py` (modified)

After the existing prediction form, add:
```python
st.divider()
render_analytics_section()
```
importing `render_analytics_section` from `ui/analytics_section.py`.

---

## Per-query presentation

| Query | Chart | Table |
|---|---|---|
| Q1 hourly demand | Altair heatmap: x=pickup_hour (0-23), y=day name (Sun-Sat), color=trip_count | The same 7x24 pivoted matrix from `pivot_hourly_demand()`, via `st.dataframe` |
| Q2 zone revenue | Altair horizontal bar chart, top 15 zones by `total_revenue`, labeled with zone name + borough | Full 264-row `st.dataframe` (user-sortable/scrollable), raw CSV plus a `zone_label` column |
| Q3 tipping by distance | Altair bar chart, x=distance_bucket (short/medium/long order), y=avg_tip_pct | 3-row table, same order |
| Q4 fare per mile | Altair bar chart, x=distance_bucket (short/medium/long order), y=avg_fare_per_mile | 3-row table, same order |

Each subsection includes one sentence of finding, adapted from the existing prose in `analytics_results.md` (e.g. Q2: "Airport zones generate the most revenue despite far fewer trips than busy Manhattan zones, since airport fares run 3-4x higher.").

Altair is bundled with Streamlit already — no new dependency. Chart code must follow the `dataviz` skill's guidance (invoked at implementation time, before writing chart specs).

---

## Error handling

`render_analytics_section()` wraps its CSV loading in a try/except: if any of the 4 result files or the zone lookup CSV is missing, it renders `st.warning("Analytics results not found under results/ or data/taxi_zone_lookup.csv — run `make download-zones` and ensure results/*.csv exist.")` and returns without raising, so the prediction form above it keeps working.

---

## Testing

- `tests/test_analytics_helpers.py`: unit tests for `order_distance_buckets`, `top_zones_by_revenue`, `pivot_hourly_demand` — pure pandas fixtures, no Spark.
- Manual verification: run `make run-ui`, confirm all 4 sections render below the prediction form with correct tables/charts and no missing-file warning (since `results/*.csv` are already committed).

---

## Constraints

- No changes to `work/analytics.py` (the Spark query definitions) — this feature only displays the already-computed, checked-in CSV outputs.
- No new Python dependencies (Altair ships with Streamlit).
- `docker-compose.yml` needs `./results:/home/jovyan/results` added to the `pyspark` service's volumes.
