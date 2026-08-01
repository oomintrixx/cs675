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
