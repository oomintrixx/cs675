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
            make_taxi_row(datetime.date(2022, 1, 2), fare_amount=15.0, trip_distance=3.0),
            make_taxi_row(datetime.date(2022, 1, 3), fare_amount=30.0, trip_distance=6.0),
        ])
        weather_df = categorize_weather(spark.createDataFrame([
            make_weather_row(DATE=datetime.date(2022, 1, 1), PRCP=0, SNOW=0),   # clear
            make_weather_row(DATE=datetime.date(2022, 1, 2), PRCP=50, SNOW=0),  # rain
            make_weather_row(DATE=datetime.date(2022, 1, 3), PRCP=0, SNOW=30),  # snow
        ]))

        result = {
            row["weather_condition"]: row
            for row in query_demand_by_weather(taxi_df, weather_df).collect()
        }

        assert result["clear"]["trip_count"] == 2
        assert result["clear"]["avg_fare"] == pytest.approx(15.0)
        assert result["rain"]["trip_count"] == 1
        assert result["rain"]["avg_fare"] == pytest.approx(15.0)
        assert result["snow"]["trip_count"] == 1
        assert result["snow"]["avg_distance"] == pytest.approx(6.0)
