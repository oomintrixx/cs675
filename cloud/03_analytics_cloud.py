# cloud/03_analytics_cloud.py
import os, sys
sys.path.insert(0, "/home/hadoop/")

from pyspark.sql import SparkSession
from analytics import (
    query_hourly_demand, query_zone_revenue,
    query_tipping_by_distance, query_fare_per_mile,
)

spark = SparkSession.builder.appName("cs675-analytics-cloud").getOrCreate()

BUCKET = os.environ["CS675_BUCKET"]
IN     = f"s3://{BUCKET}/output/taxi_clean/"
OUT    = f"s3://{BUCKET}/output/results/"

print("Reading clean taxi data from S3...")
df = spark.read.parquet(IN)
print(f"  Rows: {df.count():,}")

query_hourly_demand(df).write.mode("overwrite").parquet(OUT + "q1/")
query_zone_revenue(df).write.mode("overwrite").parquet(OUT + "q2/")
query_tipping_by_distance(df).write.mode("overwrite").parquet(OUT + "q3/")
query_fare_per_mile(df).write.mode("overwrite").parquet(OUT + "q4/")

print("Cloud analytics complete. Results at:", OUT)
spark.stop()
