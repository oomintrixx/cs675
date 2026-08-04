# tests/test_weather.py
import datetime
from pyspark.sql import Row


def make_weather_row(**kw):
    defaults = dict(STATION="USW00094728", DATE=datetime.date(2022, 1, 15), PRCP=0, SNOW=0)
    defaults.update(kw)
    return Row(**defaults)


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
