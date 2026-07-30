# work/analytics.py
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, avg, count, sum as spark_sum, round as spark_round


def query_hourly_demand(df: DataFrame) -> DataFrame:
    """
    Q1: Trip count by hour of day and day of week.
    Shows commute spikes (Mon–Fri 7–9am, 5–7pm) vs weekend leisure patterns.
    """
    return (
        df.groupBy("pickup_hour", "day_of_week")
          .agg(count("*").alias("trip_count"))
          .orderBy("day_of_week", "pickup_hour")
    )


def query_zone_revenue(df: DataFrame) -> DataFrame:
    """
    Q2: Total revenue and trip count per pickup zone (PULocationID).
    Identifies highest-value zones for drivers and fleet operators.
    """
    return (
        df.groupBy("PULocationID")
          .agg(
              spark_round(spark_sum("fare_amount"), 2).alias("total_revenue"),
              spark_round(avg("fare_amount"), 2).alias("avg_fare"),
              count("*").alias("trip_count"),
          )
          .orderBy("total_revenue", ascending=False)
    )


def query_tipping_by_distance(df: DataFrame) -> DataFrame:
    """
    Q3: Average tip percentage by distance bucket (short/medium/long).
    Tests whether longer trips earn proportionally more tips.
    tip_pct = (tip_amount / fare_amount) * 100
    """
    return (
        df.groupBy("distance_bucket")
          .agg(
              spark_round(avg(col("tip_amount") / col("fare_amount") * 100), 2).alias("avg_tip_pct"),
              spark_round(avg("tip_amount"), 2).alias("avg_tip_amount"),
              count("*").alias("trip_count"),
          )
          .orderBy("distance_bucket")
    )


def query_fare_per_mile(df: DataFrame) -> DataFrame:
    """
    Q4: Average fare per mile by distance bucket.
    Short trips should have higher fare/mile due to base fare dominating.
    """
    return (
        df.groupBy("distance_bucket")
          .agg(
              spark_round(avg(col("fare_amount") / col("trip_distance")), 2).alias("avg_fare_per_mile"),
              spark_round(avg("fare_amount"), 2).alias("avg_fare"),
              spark_round(avg("trip_distance"), 2).alias("avg_distance"),
              count("*").alias("trip_count"),
          )
          .orderBy("distance_bucket")
    )
