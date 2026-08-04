# work/weather_helpers.py
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import StructType, StructField, StringType, DateType, IntegerType

WEATHER_SCHEMA = StructType([
    StructField("STATION", StringType()),
    StructField("DATE", DateType()),
    StructField("PRCP", IntegerType()),
    StructField("SNOW", IntegerType()),
])


def load_weather(spark: SparkSession, path: str) -> DataFrame:
    """
    Reads the filtered NOAA GHCN-Daily weather CSV (STATION, DATE, PRCP, SNOW)
    and fills any null PRCP/SNOW with 0 (no precipitation recorded that day).
    """
    return (
        spark.read.csv(path, header=True, schema=WEATHER_SCHEMA)
        .fillna({"PRCP": 0, "SNOW": 0})
    )


def categorize_weather(df: DataFrame) -> DataFrame:
    """
    Adds weather_condition: 'snow' if SNOW > 0, else 'rain' if PRCP > 0,
    else 'clear'. Snow takes priority on days with both.
    """
    return df.withColumn(
        "weather_condition",
        when(col("SNOW") > 0, "snow")
        .when(col("PRCP") > 0, "rain")
        .otherwise("clear")
    )
