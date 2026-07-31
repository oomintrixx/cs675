# cloud/02_preprocess_cloud.py
# Same logic as work/02_preprocess.py — S3 paths, no local Spark config.
import os, sys
sys.path.insert(0, "/home/hadoop/")

from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, hour, dayofweek, col
from pyspark.sql.types import (
    StructType, StructField, LongType, DoubleType, StringType, TimestampNTZType,
)
from preprocess_steps import impute, remove_outliers, normalize, encode, bin_distance, bin_time_of_day

spark = SparkSession.builder.appName("cs675-preprocess-cloud").getOrCreate()
# TLC parquet files drift in type for some columns across years/months
# (e.g. airport_fee is int in some files, double in others). Schema
# inference/merging can't reconcile int vs double for the same column, so
# we pin an explicit schema and disable the vectorized reader — that lets
# Spark safely upcast the int-typed files to double on read.
spark.conf.set("spark.sql.parquet.enableVectorizedReader", "false")

TAXI_SCHEMA = StructType([
    StructField("VendorID", LongType()),
    StructField("tpep_pickup_datetime", TimestampNTZType()),
    StructField("tpep_dropoff_datetime", TimestampNTZType()),
    StructField("passenger_count", DoubleType()),
    StructField("trip_distance", DoubleType()),
    StructField("RatecodeID", DoubleType()),
    StructField("store_and_fwd_flag", StringType()),
    StructField("PULocationID", LongType()),
    StructField("DOLocationID", LongType()),
    StructField("payment_type", LongType()),
    StructField("fare_amount", DoubleType()),
    StructField("extra", DoubleType()),
    StructField("mta_tax", DoubleType()),
    StructField("tip_amount", DoubleType()),
    StructField("tolls_amount", DoubleType()),
    StructField("improvement_surcharge", DoubleType()),
    StructField("total_amount", DoubleType()),
    StructField("congestion_surcharge", DoubleType()),
    StructField("airport_fee", DoubleType()),
])

BUCKET  = os.environ["CS675_BUCKET"]
TAXI_IN = f"s3://{BUCKET}/data/taxi/"
OUT     = f"s3://{BUCKET}/output/taxi_clean/"

print("Loading taxi from S3...")
raw = spark.read.schema(TAXI_SCHEMA).parquet(TAXI_IN)
print(f"  Raw rows: {raw.count():,}")

df = (
    raw
    .withColumn("pickup_date", to_date(col("tpep_pickup_datetime")))
    .withColumn("pickup_hour", hour(col("tpep_pickup_datetime")))
    .withColumn("day_of_week", dayofweek(col("tpep_pickup_datetime")))
)
df = impute(df)
df = remove_outliers(df)
df = normalize(df)
df = encode(df)
df = bin_distance(df)
df = bin_time_of_day(df)

print(f"  Clean rows: {df.count():,}")
df.write.mode("overwrite").partitionBy("year", "month").parquet(OUT)
print("Cloud preprocessing complete.")
spark.stop()
