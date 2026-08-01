# cloud/04_ml_cloud.py
# Same logic as work/04_ml.py — only difference is S3 paths and no local Spark config.
import os, sys
sys.path.insert(0, "/home/hadoop/")

from pyspark.sql import SparkSession
from ml_features import build_features
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("cs675-ml-cloud").getOrCreate()

BUCKET = os.environ["CS675_BUCKET"]
IN     = f"s3://{BUCKET}/output/taxi_clean/"
OUT    = f"s3://{BUCKET}/output/"

print("Loading clean taxi data from S3...")
df = spark.read.parquet(IN)
print(f"  Rows: {df.count():,}")

featured = build_features(df)
train_df, test_df = featured.randomSplit([0.8, 0.2], seed=42)
print(f"  Train: {train_df.count():,} | Test: {test_df.count():,}")

evaluator_rmse = RegressionEvaluator(
    labelCol="total_amount", predictionCol="prediction", metricName="rmse"
)
evaluator_r2 = RegressionEvaluator(
    labelCol="total_amount", predictionCol="prediction", metricName="r2"
)

print("Training Linear Regression...")
lr_model = LinearRegression(
    featuresCol="features", labelCol="total_amount",
    maxIter=100, regParam=0.1, elasticNetParam=0.0,
).fit(train_df)
lr_preds = lr_model.transform(test_df)
lr_rmse = evaluator_rmse.evaluate(lr_preds)
lr_r2   = evaluator_r2.evaluate(lr_preds)
print(f"  LR  — RMSE: ${lr_rmse:.2f} | R²: {lr_r2:.4f}")

print("Training Gradient Boosted Trees...")
gbt_model = GBTRegressor(
    featuresCol="features", labelCol="total_amount",
    maxIter=50, maxDepth=5, maxBins=300, seed=42,
).fit(train_df)
gbt_preds = gbt_model.transform(test_df)
gbt_rmse = evaluator_rmse.evaluate(gbt_preds)
gbt_r2   = evaluator_r2.evaluate(gbt_preds)
print(f"  GBT — RMSE: ${gbt_rmse:.2f} | R²: {gbt_r2:.4f}")

print(f"\n── Model Comparison ──────────────────────────────────────────────────")
print(f"{'Linear Regression':<30} RMSE=${lr_rmse:.2f}  R²={lr_r2:.4f}")
print(f"{'Gradient Boosted Trees':<30} RMSE=${gbt_rmse:.2f}  R²={gbt_r2:.4f}")

lr_model.write().overwrite().save(OUT + "models/lr/")
gbt_model.write().overwrite().save(OUT + "models/gbt/")
print(f"Models saved to {OUT}models/")

spark.stop()
