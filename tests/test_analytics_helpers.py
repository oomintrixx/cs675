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
