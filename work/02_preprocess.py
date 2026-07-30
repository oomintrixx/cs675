# work/02_preprocess.py
from work.spark_helper import get_spark
from work.constants import TAXI_PATH, OUTPUT_PATH
from work.preprocess_steps import impute, remove_outliers, normalize, encode, bin_distance, bin_time_of_day
from pyspark.sql.functions import to_date, hour, dayofweek, col

spark = get_spark("02_preprocess")

print("Loading raw taxi data...")
raw = spark.read.parquet(TAXI_PATH)
print(f"  Raw rows: {raw.count():,}")

df = (
    raw
    .withColumn("pickup_date",  to_date(col("tpep_pickup_datetime")))
    .withColumn("pickup_hour",  hour(col("tpep_pickup_datetime")))
    .withColumn("day_of_week",  dayofweek(col("tpep_pickup_datetime")))
)
df = impute(df)
df = remove_outliers(df)
df = normalize(df)
df = encode(df)
df = bin_distance(df)
df = bin_time_of_day(df)

print(f"  After preprocessing: {df.count():,} rows")
df.select(
    "pickup_date", "pickup_hour", "day_of_week", "time_of_day",
    "PULocationID", "trip_distance", "distance_bucket",
    "fare_amount", "fare_norm", "pay_credit_card"
).show(5)

df.write.mode("overwrite").parquet(OUTPUT_PATH + "taxi_clean/")
print(f"Written to {OUTPUT_PATH}taxi_clean/")
spark.stop()
