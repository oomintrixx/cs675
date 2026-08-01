# Analytics Results in UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Display the 4 analytics query results (hourly demand, zone revenue, tipping by distance, fare per mile) as tables + charts on the Streamlit prediction page.

**Architecture:** Pure pandas helper functions in `work/analytics_helpers.py` reshape the already-committed `results/*.csv` (full-scale cloud run output) for display; `ui/analytics_section.py` reads those CSVs plus `data/taxi_zone_lookup.csv`, calls the helpers, and renders 4 subsections (table + Altair chart each) that `ui/predict_app.py` invokes below the existing prediction form.

**Tech Stack:** pandas, Altair (bundled with Streamlit — no new dependency), pytest.

## Global Constraints

- No changes to `work/analytics.py` (the Spark query definitions) — this feature only displays the already-computed, checked-in `results/*.csv` outputs.
- No new Python dependencies.
- Chart color: single accent `#3987e5` for all bar charts (one series each — magnitude by category, not multiple series, so no legend or categorical palette needed); Q1's heatmap uses a sequential blue ramp (Altair's built-in `scheme="blues"`) since it encodes a single magnitude (trip_count), matching the "sequential = one hue, light→dark" rule from the dataviz skill. Never dual-axis; every chart gets hover tooltips.
- `results/` must be mounted into the `pyspark` Docker service (`./results:/home/jovyan/results`) — it isn't currently.

---

### Task 1: Pure analytics display helpers (TDD)

**Files:**
- Create: `work/analytics_helpers.py`
- Test: `tests/test_analytics_helpers.py`

**Interfaces:**
- Produces:
  - `DISTANCE_BUCKET_ORDER: list[str]` = `["short", "medium", "long"]`
  - `DAY_OF_WEEK_NAMES: dict[int, str]` = `{1: "Sunday", ..., 7: "Saturday"}`
  - `order_distance_buckets(df: pd.DataFrame) -> pd.DataFrame` — reindexes a DataFrame with a `distance_bucket` column into short/medium/long order (source CSVs come out alphabetical: long, medium, short)
  - `add_day_name(hourly_df: pd.DataFrame) -> pd.DataFrame` — adds a `day_name` column (Sunday..Saturday) derived from `day_of_week` (1-7, Spark's `dayofweek()` convention)
  - `pivot_hourly_demand(hourly_df: pd.DataFrame) -> pd.DataFrame` — pivots `(pickup_hour, day_of_week, trip_count)` into a 7x24 matrix: index = day names in Sunday..Saturday order, columns = `pickup_hour` (0-23), values = `trip_count`
  - `top_zones_by_revenue(zone_revenue_df: pd.DataFrame, zone_lookup_df: pd.DataFrame, n: int = 15) -> pd.DataFrame` — joins `PULocationID` -> `Zone`/`Borough` from the TLC zone lookup, returns the top `n` rows by `total_revenue` (descending) with an added `zone_label` column formatted `"{Zone}, {Borough}"`
- Consumed by: Task 2's `ui/analytics_section.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_analytics_helpers.py
import pandas as pd


class TestOrderDistanceBuckets:
    def test_reorders_to_short_medium_long(self):
        from work.analytics_helpers import order_distance_buckets
        df = pd.DataFrame({
            "distance_bucket": ["long", "medium", "short"],
            "avg_tip_pct": [16.69, 18.86, 22.42],
        })
        result = order_distance_buckets(df)
        assert list(result["distance_bucket"]) == ["short", "medium", "long"]
        assert list(result["avg_tip_pct"]) == [22.42, 18.86, 16.69]


class TestAddDayName:
    def test_maps_spark_dow_to_names(self):
        from work.analytics_helpers import add_day_name
        df = pd.DataFrame({"pickup_hour": [0, 0], "day_of_week": [1, 7], "trip_count": [10, 20]})
        result = add_day_name(df)
        assert list(result["day_name"]) == ["Sunday", "Saturday"]


class TestPivotHourlyDemand:
    def test_pivots_into_7x24_matrix(self):
        from work.analytics_helpers import pivot_hourly_demand
        rows = [
            {"pickup_hour": h, "day_of_week": d, "trip_count": h + d}
            for d in range(1, 8) for h in range(24)
        ]
        df = pd.DataFrame(rows)
        result = pivot_hourly_demand(df)
        assert result.shape == (7, 24)
        assert list(result.index) == [
            "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"
        ]
        assert list(result.columns) == list(range(24))
        assert result.loc["Sunday", 0] == 1  # h=0, d=1 -> trip_count = 0 + 1


class TestTopZonesByRevenue:
    def test_adds_zone_label_and_sorts_top_n(self):
        from work.analytics_helpers import top_zones_by_revenue
        zone_revenue_df = pd.DataFrame({
            "PULocationID": [132, 161, 138],
            "total_revenue": [279797932.95, 82492983.28, 132983656.15],
            "avg_fare": [45.73, 11.78, 30.98],
            "trip_count": [6118817, 7003527, 4292378],
        })
        zone_lookup_df = pd.DataFrame({
            "LocationID": [132, 161, 138],
            "Borough": ["Queens", "Manhattan", "Queens"],
            "Zone": ["JFK Airport", "Midtown Center", "LaGuardia Airport"],
        })
        result = top_zones_by_revenue(zone_revenue_df, zone_lookup_df, n=2)
        assert len(result) == 2
        assert list(result["zone_label"]) == ["JFK Airport, Queens", "LaGuardia Airport, Queens"]
        assert list(result["PULocationID"]) == [132, 138]
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `docker compose exec pyspark bash -c "cd /home/jovyan && python -m pytest tests/test_analytics_helpers.py -v"`
Expected: `ModuleNotFoundError: No module named 'work.analytics_helpers'`

- [ ] **Step 3: Implement `work/analytics_helpers.py`**

```python
# work/analytics_helpers.py
import pandas as pd

DISTANCE_BUCKET_ORDER = ["short", "medium", "long"]

DAY_OF_WEEK_NAMES = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday",
}


def order_distance_buckets(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.set_index("distance_bucket").loc[DISTANCE_BUCKET_ORDER]
    return ordered.reset_index()


def add_day_name(hourly_df: pd.DataFrame) -> pd.DataFrame:
    df = hourly_df.copy()
    df["day_name"] = df["day_of_week"].map(DAY_OF_WEEK_NAMES)
    return df


def pivot_hourly_demand(hourly_df: pd.DataFrame) -> pd.DataFrame:
    with_day_name = add_day_name(hourly_df)
    pivoted = with_day_name.pivot(index="day_name", columns="pickup_hour", values="trip_count")
    return pivoted.reindex(list(DAY_OF_WEEK_NAMES.values()))


def top_zones_by_revenue(
    zone_revenue_df: pd.DataFrame, zone_lookup_df: pd.DataFrame, n: int = 15
) -> pd.DataFrame:
    merged = zone_revenue_df.merge(
        zone_lookup_df[["LocationID", "Zone", "Borough"]],
        left_on="PULocationID",
        right_on="LocationID",
        how="left",
    )
    merged["zone_label"] = merged["Zone"] + ", " + merged["Borough"]
    return (
        merged.sort_values("total_revenue", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `docker compose exec pyspark bash -c "cd /home/jovyan && python -m pytest tests/test_analytics_helpers.py -v"`
Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add work/analytics_helpers.py tests/test_analytics_helpers.py
git commit -m "feat: pure helpers for shaping analytics results for display"
```

---

### Task 2: Streamlit analytics section + wire into the prediction page

**Files:**
- Create: `ui/analytics_section.py`
- Modify: `ui/predict_app.py`
- Modify: `docker-compose.yml`

**Interfaces:**
- Consumes: `order_distance_buckets`, `add_day_name`, `pivot_hourly_demand`, `top_zones_by_revenue` (Task 1, `work/analytics_helpers.py`)
- Produces: `render_analytics_section() -> None` — called from `ui/predict_app.py`

This task is UI wiring, verified by manually running the app and checking the browser — not unit tests.

- [ ] **Step 1: Mount `results/` into the `pyspark` Docker service**

In `docker-compose.yml`, add a line to the `pyspark` service's `volumes` list (alongside the existing `./ui:/home/jovyan/ui` line):

```yaml
      - ./results:/home/jovyan/results
```

- [ ] **Step 2: Recreate the container so the new mount takes effect**

```bash
docker compose up -d --force-recreate pyspark
```
Expected: `docker compose ps` shows `pyspark` as `Up`.

- [ ] **Step 3: Create `ui/analytics_section.py`**

```python
# ui/analytics_section.py
import pandas as pd
import altair as alt
import streamlit as st

from work.analytics_helpers import (
    add_day_name,
    order_distance_buckets,
    pivot_hourly_demand,
    top_zones_by_revenue,
)

ACCENT = "#3987e5"


@st.cache_data
def _load_results():
    hourly = pd.read_csv("results/q1_hourly_demand.csv")
    zones = pd.read_csv("results/q2_zone_revenue.csv")
    tipping = pd.read_csv("results/q3_tipping_by_distance.csv")
    fare_per_mile = pd.read_csv("results/q4_fare_per_mile.csv")
    zone_lookup = pd.read_csv("data/taxi_zone_lookup.csv")
    return hourly, zones, tipping, fare_per_mile, zone_lookup


def render_analytics_section() -> None:
    st.header("📊 Analytics: NYC Taxi Trends (2019–2022)")
    st.caption("Computed at cloud scale (~177M trips) via PySpark on EMR Serverless.")

    try:
        hourly, zones, tipping, fare_per_mile, zone_lookup = _load_results()
    except FileNotFoundError:
        st.warning(
            "Analytics results not found under `results/` or "
            "`data/taxi_zone_lookup.csv` — run `make download-zones` and "
            "ensure `results/*.csv` exist."
        )
        return

    st.subheader("Q1 — Demand by hour and day of week")
    st.write(
        "A single evening peak (5-7pm) dominates both weekdays and weekends, "
        "rather than the classic AM/PM commute double-hump."
    )
    hourly_long = add_day_name(hourly)
    heatmap = (
        alt.Chart(hourly_long)
        .mark_rect()
        .encode(
            x=alt.X("pickup_hour:O", title="Hour of day"),
            y=alt.Y(
                "day_name:N",
                title=None,
                sort=["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
            ),
            color=alt.Color("trip_count:Q", title="Trips", scale=alt.Scale(scheme="blues")),
            tooltip=["day_name", "pickup_hour", "trip_count"],
        )
        .properties(height=260)
    )
    st.altair_chart(heatmap, use_container_width=True)
    st.dataframe(pivot_hourly_demand(hourly), use_container_width=True)

    st.subheader("Q2 — Revenue by pickup zone")
    st.write(
        "Airport zones (JFK, LaGuardia) generate the most total revenue despite "
        "far fewer trips than busy Manhattan zones, since airport fares run 3-4x higher."
    )
    top_zones = top_zones_by_revenue(zones, zone_lookup, n=15)
    zone_bar = (
        alt.Chart(top_zones)
        .mark_bar(color=ACCENT)
        .encode(
            y=alt.Y("zone_label:N", sort="-x", title=None),
            x=alt.X("total_revenue:Q", title="Total revenue ($)"),
            tooltip=["zone_label", "total_revenue", "avg_fare", "trip_count"],
        )
        .properties(height=400)
    )
    st.altair_chart(zone_bar, use_container_width=True)
    zones_with_names = zones.merge(
        zone_lookup[["LocationID", "Zone", "Borough"]],
        left_on="PULocationID",
        right_on="LocationID",
        how="left",
    ).drop(columns="LocationID")
    st.dataframe(zones_with_names, use_container_width=True)

    st.subheader("Q3 — Tipping behavior by trip distance")
    st.write(
        "Tip percentage falls as trips get longer, even though tip dollar amount "
        "rises — riders tip a smaller share of a larger fare on long trips."
    )
    tipping_ordered = order_distance_buckets(tipping)
    tip_bar = (
        alt.Chart(tipping_ordered)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X("distance_bucket:N", sort=["short", "medium", "long"], title="Distance bucket"),
            y=alt.Y("avg_tip_pct:Q", title="Avg tip %"),
            tooltip=["distance_bucket", "avg_tip_pct", "avg_tip_amount", "trip_count"],
        )
        .properties(height=260)
    )
    st.altair_chart(tip_bar, use_container_width=True)
    st.dataframe(tipping_ordered, use_container_width=True)

    st.subheader("Q4 — Fare efficiency by trip distance")
    st.write(
        "Short trips cost ~6x more per mile than long trips because the flat "
        "base fare dominates the total for short distances."
    )
    fare_ordered = order_distance_buckets(fare_per_mile)
    fare_bar = (
        alt.Chart(fare_ordered)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X("distance_bucket:N", sort=["short", "medium", "long"], title="Distance bucket"),
            y=alt.Y("avg_fare_per_mile:Q", title="Avg fare per mile ($)"),
            tooltip=["distance_bucket", "avg_fare_per_mile", "avg_fare", "avg_distance", "trip_count"],
        )
        .properties(height=260)
    )
    st.altair_chart(fare_bar, use_container_width=True)
    st.dataframe(fare_ordered, use_container_width=True)
```

- [ ] **Step 4: Wire `render_analytics_section()` into `ui/predict_app.py`**

Add this import near the top of `ui/predict_app.py`, alongside the existing `work.predict_helpers` import:

```python
from ui.analytics_section import render_analytics_section
```

Add this at the very end of `ui/predict_app.py` (after the existing `if submitted:` block that renders the prediction):

```python
st.divider()
render_analytics_section()
```

- [ ] **Step 5: Run the app and verify manually**

```bash
make run-ui
```
Open `http://localhost:8501` and confirm:
- The existing prediction form still works (fill it in, click Predict, see a `$X.XX` result — same as before).
- Below it, a "📊 Analytics: NYC Taxi Trends (2019–2022)" header appears with 4 subsections (Q1-Q4), each showing a chart and a table with real numbers (not a `st.warning` about missing files, since `results/*.csv` are already committed).
- Q1's heatmap shows a visible evening peak band around hour 17-19 across most days.
- Q2's bar chart shows JFK Airport and LaGuardia Airport at or near the top.
- Q3 and Q4's bar charts show bars in short/medium/long order (not alphabetical).

Stop the app with `Ctrl+C` once verified.

- [ ] **Step 6: Commit**

```bash
git add ui/analytics_section.py ui/predict_app.py docker-compose.yml
git commit -m "feat: display analytics results (tables + charts) on the prediction page"
```

---

## Self-Review Against Spec

| Spec requirement | Covered by |
|---|---|
| Data source: `results/*.csv`, read via pandas (no Spark) | Task 2, `_load_results()` |
| `results/` mounted into Docker | Task 2, Step 1 |
| `order_distance_buckets`, `top_zones_by_revenue`, `pivot_hourly_demand` pure + tested | Task 1 |
| Q1 heatmap (hour x day, trip_count color) + table | Task 2, `render_analytics_section()` Q1 block |
| Q2 top-15 zone bar chart with zone names + full sortable table | Task 2, Q2 block |
| Q3/Q4 bar charts in short/medium/long order + tables | Task 2, Q3/Q4 blocks |
| One-sentence finding per query, adapted from `analytics_results.md` | Task 2, `st.write(...)` calls |
| Missing-file error handling (doesn't crash the page) | Task 2, `try/except FileNotFoundError` |
| No new dependencies (Altair ships with Streamlit) | Task 2 imports only `altair`, already available |
| No changes to `work/analytics.py` | Not touched in either task |
