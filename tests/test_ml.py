# tests/test_ml.py
import pytest
import datetime
from pyspark.sql import Row


def make_clean_rows(spark, n=3):
    """Create n rows of preprocessed taxi data with all columns needed for ML."""
    rows = [
        Row(
            pickup_date=datetime.date(2022, 1, 15),
            pickup_hour=10,
            day_of_week=7,
            time_of_day="morning",
            PULocationID=161,
            DOLocationID=236,
            trip_distance=2.5,
            distance_bucket="medium",
            fare_amount=12.0,
            fare_norm=0.5,
            tip_amount=2.0,
            total_amount=15.5,
            passenger_count=1,
            payment_type=1,
            pay_credit_card=1,
            pay_cash=0,
            pay_no_charge=0,
            dist_norm=0.5,
        )
        for _ in range(n)
    ]
    return spark.createDataFrame(rows)


class TestBuildFeatures:
    def test_features_column_exists(self, spark):
        from work.ml_features import build_features
        result = build_features(make_clean_rows(spark))
        assert "features" in result.columns

    def test_features_vector_size_is_8(self, spark):
        from work.ml_features import build_features
        from pyspark.ml.linalg import Vector
        result = build_features(make_clean_rows(spark))
        vec = result.first()["features"]
        assert vec.size == 8

    def test_no_null_features(self, spark):
        from work.ml_features import build_features
        from pyspark.sql.functions import col
        result = build_features(make_clean_rows(spark))
        null_count = result.filter(col("features").isNull()).count()
        assert null_count == 0

    def test_target_column_present(self, spark):
        from work.ml_features import build_features
        result = build_features(make_clean_rows(spark))
        assert "total_amount" in result.columns

    def test_pu_idx_and_do_idx_columns_created(self, spark):
        from work.ml_features import build_features
        result = build_features(make_clean_rows(spark))
        assert "pu_idx" in result.columns
        assert "do_idx" in result.columns
