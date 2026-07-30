# tests/test_preprocess.py
import pytest
import datetime
from pyspark.sql import Row
from pyspark.sql.types import (
    StructType, StructField, TimestampType, DoubleType, LongType,
)

ROW_SCHEMA = StructType([
    StructField("tpep_pickup_datetime", TimestampType()),
    StructField("tpep_dropoff_datetime", TimestampType()),
    StructField("trip_distance", DoubleType()),
    StructField("fare_amount", DoubleType()),
    StructField("tip_amount", DoubleType()),
    StructField("total_amount", DoubleType()),
    StructField("passenger_count", DoubleType()),
    StructField("payment_type", LongType()),
    StructField("PULocationID", LongType()),
    StructField("DOLocationID", LongType()),
])


def make_row(spark, **overrides):
    defaults = dict(
        tpep_pickup_datetime=datetime.datetime(2022, 1, 15, 10, 0, 0),
        tpep_dropoff_datetime=datetime.datetime(2022, 1, 15, 10, 30, 0),
        trip_distance=2.5,
        fare_amount=12.0,
        tip_amount=2.0,
        total_amount=15.0,
        passenger_count=1.0,
        payment_type=1,
        PULocationID=161,
        DOLocationID=236,
    )
    defaults.update(overrides)
    return spark.createDataFrame([Row(**defaults)], schema=ROW_SCHEMA)


class TestImputation:
    def test_null_passenger_count_filled_with_1(self, spark):
        from work.preprocess_steps import impute
        df = make_row(spark, passenger_count=None)
        assert impute(df).first()["passenger_count"] == 1

    def test_null_fare_amount_row_dropped(self, spark):
        from work.preprocess_steps import impute
        df = make_row(spark, fare_amount=None)
        assert impute(df).count() == 0


class TestOutlierRemoval:
    def test_negative_distance_dropped(self, spark):
        from work.preprocess_steps import remove_outliers
        df = make_row(spark, trip_distance=-1.0)
        assert remove_outliers(df).count() == 0

    def test_zero_distance_dropped(self, spark):
        from work.preprocess_steps import remove_outliers
        df = make_row(spark, trip_distance=0.0)
        assert remove_outliers(df).count() == 0

    def test_extreme_fare_capped_at_500(self, spark):
        from work.preprocess_steps import remove_outliers
        df = make_row(spark, fare_amount=9999.0)
        assert remove_outliers(df).first()["fare_amount"] == 500.0

    def test_valid_row_kept(self, spark):
        from work.preprocess_steps import remove_outliers
        df = make_row(spark)
        assert remove_outliers(df).count() == 1


class TestNormalization:
    def test_fare_normalized_to_0_and_1(self, spark):
        from work.preprocess_steps import normalize
        from functools import reduce
        df = reduce(lambda a, b: a.union(b), [
            make_row(spark, fare_amount=0.0),
            make_row(spark, fare_amount=100.0),
        ])
        norms = sorted(r["fare_norm"] for r in normalize(df).collect())
        assert norms[0] == pytest.approx(0.0)
        assert norms[1] == pytest.approx(1.0)


class TestEncoding:
    def test_credit_card_encoded(self, spark):
        from work.preprocess_steps import encode
        row = encode(make_row(spark, payment_type=1)).first()
        assert row["pay_credit_card"] == 1
        assert row["pay_cash"] == 0

    def test_cash_encoded(self, spark):
        from work.preprocess_steps import encode
        row = encode(make_row(spark, payment_type=2)).first()
        assert row["pay_credit_card"] == 0
        assert row["pay_cash"] == 1

    def test_unknown_payment_type_all_zero(self, spark):
        from work.preprocess_steps import encode
        row = encode(make_row(spark, payment_type=99)).first()
        assert row["pay_credit_card"] == 0
        assert row["pay_cash"] == 0


class TestBinning:
    def test_short_trip(self, spark):
        from work.preprocess_steps import bin_distance
        assert bin_distance(make_row(spark, trip_distance=0.5)).first()["distance_bucket"] == "short"

    def test_medium_trip(self, spark):
        from work.preprocess_steps import bin_distance
        assert bin_distance(make_row(spark, trip_distance=5.0)).first()["distance_bucket"] == "medium"

    def test_long_trip(self, spark):
        from work.preprocess_steps import bin_distance
        assert bin_distance(make_row(spark, trip_distance=15.0)).first()["distance_bucket"] == "long"

    def test_morning_time_of_day(self, spark):
        from work.preprocess_steps import bin_time_of_day
        import datetime
        df = make_row(spark, tpep_pickup_datetime=datetime.datetime(2022, 1, 15, 8, 0, 0))
        assert bin_time_of_day(df).first()["time_of_day"] == "morning"

    def test_night_time_of_day(self, spark):
        from work.preprocess_steps import bin_time_of_day
        import datetime
        df = make_row(spark, tpep_pickup_datetime=datetime.datetime(2022, 1, 15, 23, 0, 0))
        assert bin_time_of_day(df).first()["time_of_day"] == "night"
