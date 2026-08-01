# work/spark_helper.py
from pyspark.sql import SparkSession
import os

def get_spark(app_name: str = "cs675") -> SparkSession:
    """
    Local: sets master, event log, and shuffle partitions.
    Cloud (EMR Serverless): calls getOrCreate() only — EMR provides the session.
    """
    builder = SparkSession.builder.appName(app_name)
    if os.environ.get("CS675_ENV") != "cloud":
        builder = (
            builder
            .master("local[*]")
            .config("spark.eventLog.enabled", "true")
            .config("spark.eventLog.dir", "/tmp/spark-events")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.driver.memory", "4g")
        )
    return builder.getOrCreate()
