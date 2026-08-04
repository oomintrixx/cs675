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
    weather = pd.read_csv("results/q5_weather_demand.csv")
    zone_lookup = pd.read_csv("data/taxi_zone_lookup.csv")
    return hourly, zones, tipping, fare_per_mile, weather, zone_lookup


def render_analytics_section() -> None:
    st.header("📊 Analytics: NYC Taxi Trends (2019–2022)")
    st.caption("Computed at cloud scale (~177M trips) via PySpark on EMR Serverless.")

    try:
        hourly, zones, tipping, fare_per_mile, weather, zone_lookup = _load_results()
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

    st.subheader("Q5 — Demand by weather condition")
    st.caption("Local Jan 2022 sample — computed separately from the Q1-Q4 cloud-scale run.")
    st.write(
        "Raw trip counts (1,670,147 clear vs. 407,351 rain vs. 345,742 snow) "
        "mostly just track how many days of each type occurred that month "
        "(20 clear / 6 rain / 5 snow). Normalizing to trips/day isolates the "
        "real demand effect: ~83,500 trips/day on clear days vs. ~67,900/day "
        "on rain (-19%) and ~69,100/day on snow (-17%). Avg fare rises "
        "genuinely too — $12.56 (clear) to $13.68 (rain) / $12.72 (snow), a "
        "robust number since fare is capped in preprocessing. Avg distance "
        "is less trustworthy: trip_distance isn't capped, and a handful of "
        "outlier trips (up to 306,159 mi) inflate the mean. The median "
        "(1.73 / 1.89 / 1.80 mi) shows only a modest, real rain increase — "
        "the apparent snow-distance effect mostly disappears once outliers "
        "are excluded."
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
