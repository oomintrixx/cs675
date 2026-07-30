# tests/test_analytics.py
import datetime
import pytest
from pyspark.sql import Row


def make_clean_row(spark, **kw):
    defaults = dict(
        pickup_date=datetime.date(2022, 1, 15),
        pickup_hour=10,
        day_of_week=7,
        time_of_day="morning",
        PULocationID=161,
        DOLocationID=236,
        trip_distance=2.5,
        distance_bucket="medium",
        fare_amount=12.0,
        fare_norm=0.5,
        tip_amount=2.0,
        total_amount=15.0,
        passenger_count=1,
        payment_type=1,
        pay_credit_card=1,
        pay_cash=0,
        pay_no_charge=0,
        dist_norm=0.5,
    )
    defaults.update(kw)
    return spark.createDataFrame([Row(**defaults)])


class TestQuery1HourlyDemand:
    def test_output_has_required_columns(self, spark):
        from work.analytics import query_hourly_demand
        result = query_hourly_demand(make_clean_row(spark))
        assert "pickup_hour" in result.columns
        assert "day_of_week" in result.columns
        assert "trip_count" in result.columns

    def test_counts_correctly(self, spark):
        from work.analytics import query_hourly_demand
        result = query_hourly_demand(make_clean_row(spark))
        assert result.first()["trip_count"] == 1


class TestQuery2ZoneRevenue:
    def test_output_has_required_columns(self, spark):
        from work.analytics import query_zone_revenue
        result = query_zone_revenue(make_clean_row(spark))
        assert "PULocationID" in result.columns
        assert "total_revenue" in result.columns
        assert "trip_count" in result.columns

    def test_revenue_matches_fare(self, spark):
        from work.analytics import query_zone_revenue
        result = query_zone_revenue(make_clean_row(spark, fare_amount=20.0))
        assert result.first()["total_revenue"] == pytest.approx(20.0)


class TestQuery3TippingByDistance:
    def test_output_has_required_columns(self, spark):
        from work.analytics import query_tipping_by_distance
        result = query_tipping_by_distance(make_clean_row(spark))
        assert "distance_bucket" in result.columns
        assert "avg_tip_pct" in result.columns
        assert "trip_count" in result.columns

    def test_tip_pct_computed(self, spark):
        from work.analytics import query_tipping_by_distance
        # tip=2, fare=10 → tip_pct=20%
        result = query_tipping_by_distance(make_clean_row(spark, tip_amount=2.0, fare_amount=10.0))
        assert result.first()["avg_tip_pct"] == pytest.approx(20.0, abs=0.1)


class TestQuery4FarePerMile:
    def test_output_has_required_columns(self, spark):
        from work.analytics import query_fare_per_mile
        result = query_fare_per_mile(make_clean_row(spark))
        assert "distance_bucket" in result.columns
        assert "avg_fare_per_mile" in result.columns

    def test_fare_per_mile_computed(self, spark):
        from work.analytics import query_fare_per_mile
        # fare=10, distance=2 → fare_per_mile=5
        result = query_fare_per_mile(make_clean_row(spark, fare_amount=10.0, trip_distance=2.0))
        assert result.first()["avg_fare_per_mile"] == pytest.approx(5.0, abs=0.1)
