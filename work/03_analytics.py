# work/03_analytics.py
from work.spark_helper import get_spark
from work.constants import OUTPUT_PATH, WEATHER_PATH
from work.analytics import (
    query_hourly_demand,
    query_zone_revenue,
    query_tipping_by_distance,
    query_fare_per_mile,
    query_demand_by_weather,
)
from work.weather_helpers import load_weather, categorize_weather

spark = get_spark("03_analytics")

print("Loading clean taxi data...")
df = spark.read.parquet(OUTPUT_PATH + "taxi_clean/")
print(f"  Rows: {df.count():,}")

print("\n── Q1: Trip Demand by Hour and Day of Week ───────────────────────────")
query_hourly_demand(df).show(48)

print("\n── Q2: Revenue by Pickup Zone (top 20) ───────────────────────────────")
query_zone_revenue(df).show(20)

print("\n── Q3: Tipping Behavior by Distance Bucket ───────────────────────────")
query_tipping_by_distance(df).show()

print("\n── Q4: Fare Per Mile by Distance Bucket ──────────────────────────────")
query_fare_per_mile(df).show()

print("\n── Q5: Trip Demand by Weather Condition ────────────────────────────────")
weather_df = categorize_weather(load_weather(spark, WEATHER_PATH))
query_demand_by_weather(df, weather_df).show()

# Persist results
query_hourly_demand(df).write.mode("overwrite").parquet(OUTPUT_PATH + "q1_hourly_demand/")
query_zone_revenue(df).write.mode("overwrite").parquet(OUTPUT_PATH + "q2_zone_revenue/")
query_tipping_by_distance(df).write.mode("overwrite").parquet(OUTPUT_PATH + "q3_tipping/")
query_fare_per_mile(df).write.mode("overwrite").parquet(OUTPUT_PATH + "q4_fare_per_mile/")
query_demand_by_weather(df, weather_df).write.mode("overwrite").parquet(OUTPUT_PATH + "q5_weather_demand/")

print("\nAll results written to", OUTPUT_PATH)
spark.stop()
