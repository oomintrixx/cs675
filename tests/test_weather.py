# tests/test_weather.py
import datetime
import pytest
from pyspark.sql import Row


def make_weather_row(**kw):
    defaults = dict(STATION="USW00094728", DATE=datetime.date(2022, 1, 15), PRCP=0, SNOW=0)
    defaults.update(kw)
    return Row(**defaults)


def make_taxi_row(pickup_date, fare_amount=10.0, trip_distance=2.0):
    return Row(pickup_date=pickup_date, fare_amount=fare_amount, trip_distance=trip_distance)


class TestCategorizeWeather:
    def test_clear_when_no_precip(self, spark):
        from work.weather_helpers import categorize_weather
        df = spark.createDataFrame([make_weather_row(PRCP=0, SNOW=0)])
        result = categorize_weather(df)
        assert result.first()["weather_condition"] == "clear"

    def test_rain_when_prcp_only(self, spark):
        from work.weather_helpers import categorize_weather
        df = spark.createDataFrame([make_weather_row(PRCP=50, SNOW=0)])
        result = categorize_weather(df)
        assert result.first()["weather_condition"] == "rain"

    def test_snow_when_snow_only(self, spark):
        from work.weather_helpers import categorize_weather
        df = spark.createDataFrame([make_weather_row(PRCP=0, SNOW=20)])
        result = categorize_weather(df)
        assert result.first()["weather_condition"] == "snow"

    def test_snow_takes_priority_over_rain(self, spark):
        from work.weather_helpers import categorize_weather
        df = spark.createDataFrame([make_weather_row(PRCP=94, SNOW=20)])
        result = categorize_weather(df)
        assert result.first()["weather_condition"] == "snow"


class TestLoadWeather:
    def test_loads_and_fills_nulls(self, spark, tmp_path):
        from work.weather_helpers import load_weather
        csv_path = tmp_path / "weather.csv"
        csv_path.write_text(
            "STATION,DATE,PRCP,SNOW\n"
            "USW00094728,2022-01-15,50,0\n"
            "USW00094728,2022-01-16,,\n"
        )
        result = load_weather(spark, str(csv_path))
        rows = {row["DATE"]: row for row in result.collect()}
        assert rows[datetime.date(2022, 1, 15)]["PRCP"] == 50
        assert rows[datetime.date(2022, 1, 16)]["PRCP"] == 0
        assert rows[datetime.date(2022, 1, 16)]["SNOW"] == 0


class TestQueryDemandByWeather:
    def test_joins_and_groups_by_condition(self, spark):
        from work.analytics import query_demand_by_weather
        from work.weather_helpers import categorize_weather

        taxi_df = spark.createDataFrame([
            make_taxi_row(datetime.date(2022, 1, 1), fare_amount=10.0, trip_distance=2.0),
            make_taxi_row(datetime.date(2022, 1, 1), fare_amount=20.0, trip_distance=4.0),
            make_taxi_row(datetime.date(2022, 1, 2), fare_amount=25.0, trip_distance=3.0),
            make_taxi_row(datetime.date(2022, 1, 3), fare_amount=30.0, trip_distance=6.0),
        ])
        weather_df = categorize_weather(spark.createDataFrame([
            make_weather_row(DATE=datetime.date(2022, 1, 1), PRCP=0, SNOW=0),   # clear
            make_weather_row(DATE=datetime.date(2022, 1, 2), PRCP=50, SNOW=0),  # rain
            make_weather_row(DATE=datetime.date(2022, 1, 3), PRCP=0, SNOW=30),  # snow
        ]))

        rows = query_demand_by_weather(taxi_df, weather_df).collect()

        # The query orders by weather_condition; verify that ordering survives
        # without collapsing into a dict (which would discard it).
        assert [row["weather_condition"] for row in rows] == ["clear", "rain", "snow"]

        result = {row["weather_condition"]: row for row in rows}

        assert result["clear"]["trip_count"] == 2
        assert result["clear"]["avg_fare"] == pytest.approx(15.0)
        assert result["rain"]["trip_count"] == 1
        assert result["rain"]["avg_fare"] == pytest.approx(25.0)
        assert result["snow"]["trip_count"] == 1
        assert result["snow"]["avg_distance"] == pytest.approx(6.0)
        # clear and rain must be distinguishable by avg_fare alone, not just trip_count
        assert result["clear"]["avg_fare"] != result["rain"]["avg_fare"]

    def test_unmatched_taxi_date_is_dropped(self, spark):
        """Inner join: a taxi row whose pickup_date has no matching weather
        row must be silently excluded from every bucket, not counted anywhere."""
        from work.analytics import query_demand_by_weather
        from work.weather_helpers import categorize_weather

        taxi_df = spark.createDataFrame([
            make_taxi_row(datetime.date(2022, 1, 1), fare_amount=10.0, trip_distance=2.0),
            # No weather row for this date below:
            make_taxi_row(datetime.date(2022, 6, 15), fare_amount=999.0, trip_distance=999.0),
        ])
        weather_df = categorize_weather(spark.createDataFrame([
            make_weather_row(DATE=datetime.date(2022, 1, 1), PRCP=0, SNOW=0),  # clear
        ]))

        rows = query_demand_by_weather(taxi_df, weather_df).collect()
        total_trip_count = sum(row["trip_count"] for row in rows)

        assert total_trip_count == 1
        result = {row["weather_condition"]: row for row in rows}
        assert result["clear"]["trip_count"] == 1
        assert result["clear"]["avg_fare"] == pytest.approx(10.0)

    def test_uses_broadcast_join(self, spark):
        from work.analytics import query_demand_by_weather
        from work.weather_helpers import categorize_weather

        taxi_df = spark.createDataFrame([make_taxi_row(datetime.date(2022, 1, 1))])
        weather_df = categorize_weather(spark.createDataFrame([make_weather_row(DATE=datetime.date(2022, 1, 1))]))
        plan = query_demand_by_weather(taxi_df, weather_df)._jdf.queryExecution().executedPlan().toString()
        assert "BroadcastHashJoin" in plan or "BroadcastExchange" in plan
