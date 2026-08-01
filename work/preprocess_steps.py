# work/preprocess_steps.py
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, lit, hour, to_date,
    min as spark_min, max as spark_max, least,
)


def impute(df: DataFrame) -> DataFrame:
    """
    Fill passenger_count nulls with 1 (solo rider is the mode).
    Drop rows where fare_amount is null — it is the primary outcome variable.
    """
    return (
        df.fillna({"passenger_count": 1})
          .filter(col("fare_amount").isNotNull())
    )


def remove_outliers(df: DataFrame) -> DataFrame:
    """
    Drop trips with distance <= 0 (impossible), negative fare, or negative
    total (refund/adjustment rows — not a real trip cost to predict).
    Cap fare at $500 and total at $600 — the highest legitimate fare (JFK
    flat rate ~$70; >$500 is a data error), with headroom on the total for
    tip on top of it. Uncapped, a handful of corrupt total_amount rows
    (seen up to $401,095) dominate squared-error loss and blow up
    tree-based models like GBT.
    """
    return (
        df.filter(col("trip_distance") > 0)
          .filter(col("fare_amount") >= 0)
          .filter(col("total_amount") >= 0)
          .withColumn("fare_amount", least(col("fare_amount"), lit(500.0)))
          .withColumn("total_amount", least(col("total_amount"), lit(600.0)))
    )


def normalize(df: DataFrame) -> DataFrame:
    """
    Min-max normalize fare_amount and trip_distance to [0, 1].
    Allows comparison across years and enables future ML use.
    """
    stats = df.agg(
        spark_min("fare_amount").alias("fare_min"),
        spark_max("fare_amount").alias("fare_max"),
        spark_min("trip_distance").alias("dist_min"),
        spark_max("trip_distance").alias("dist_max"),
    ).first()

    fare_range = float(stats["fare_max"] - stats["fare_min"]) or 1.0
    dist_range = float(stats["dist_max"] - stats["dist_min"]) or 1.0

    return (
        df.withColumn("fare_norm",
                      (col("fare_amount") - lit(float(stats["fare_min"]))) / lit(fare_range))
          .withColumn("dist_norm",
                      (col("trip_distance") - lit(float(stats["dist_min"]))) / lit(dist_range))
    )


def encode(df: DataFrame) -> DataFrame:
    """
    One-hot encode payment_type (1=credit card, 2=cash, 3=no charge).
    Unknown types produce all zeros. Converts categorical to numeric for analysis.
    """
    return (
        df.withColumn("pay_credit_card", when(col("payment_type") == 1, 1).otherwise(0))
          .withColumn("pay_cash",        when(col("payment_type") == 2, 1).otherwise(0))
          .withColumn("pay_no_charge",   when(col("payment_type") == 3, 1).otherwise(0))
    )


def bin_distance(df: DataFrame) -> DataFrame:
    """
    Bin trip_distance: short (<1mi), medium (1–10mi), long (>10mi).
    Thresholds align with TLC fare structure (base fare tiers).
    """
    return df.withColumn(
        "distance_bucket",
        when(col("trip_distance") < 1.0, "short")
        .when(col("trip_distance") <= 10.0, "medium")
        .otherwise("long")
    )


def bin_time_of_day(df: DataFrame) -> DataFrame:
    """
    Bin pickup hour into time-of-day segments:
      overnight (0–5), morning (6–11), afternoon (12–17), evening (18–21), night (22–23).
    Captures demand patterns that differ by commute vs leisure vs late-night.
    """
    h = hour(col("tpep_pickup_datetime"))
    return df.withColumn(
        "time_of_day",
        when(h < 6,  "overnight")
        .when(h < 12, "morning")
        .when(h < 18, "afternoon")
        .when(h < 22, "evening")
        .otherwise("night")
    )
