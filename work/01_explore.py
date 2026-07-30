# work/01_explore.py
from work.spark_helper import get_spark
from work.constants import TAXI_PATH
from pyspark.sql.functions import col, count, when, isnan, lit

spark = get_spark("01_explore")

taxi = spark.read.parquet(TAXI_PATH)

print("Schema:")
taxi.printSchema()

print(f"\nRow count: {taxi.count():,}")

print("\nDescriptive stats:")
taxi.describe("trip_distance", "fare_amount", "tip_amount", "passenger_count").show()

print("\nNull counts:")
float_cols = {"trip_distance", "fare_amount"}
null_check_cols = ["trip_distance", "fare_amount", "passenger_count",
                    "tpep_pickup_datetime", "PULocationID"]
taxi.select([
    count(when(
        col(c).isNull() | (isnan(c) if c in float_cols else lit(False)), c
    )).alias(c)
    for c in null_check_cols
]).show()

print("\nSample rows:")
taxi.select(
    "tpep_pickup_datetime", "PULocationID", "DOLocationID",
    "trip_distance", "fare_amount", "tip_amount", "payment_type"
).show(10)

spark.stop()
