# work/04_ml.py
import json

from work.spark_helper import get_spark
from work.constants import OUTPUT_PATH
from work.ml_features import build_pipeline

from pyspark.ml.regression import LinearRegression, GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = get_spark("04_ml")

# ── Load preprocessed data ──────────────────────────────────────────────────────
print("Loading clean taxi data...")
df = spark.read.parquet(OUTPUT_PATH + "taxi_clean/")
print(f"  Rows: {df.count():,}")

# ── Train / test split ──────────────────────────────────────────────────────────
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"  Train: {train_df.count():,} | Test: {test_df.count():,}")

evaluator_rmse = RegressionEvaluator(
    labelCol="total_amount", predictionCol="prediction", metricName="rmse"
)
evaluator_r2 = RegressionEvaluator(
    labelCol="total_amount", predictionCol="prediction", metricName="r2"
)

# maxBins=300 must exceed the ~262 distinct PU/DOLocationID zones
MODELS = {
    "lr": LinearRegression(
        featuresCol="features", labelCol="total_amount",
        maxIter=100, regParam=0.1, elasticNetParam=0.0,
    ),
    "gbt": GBTRegressor(
        featuresCol="features", labelCol="total_amount",
        maxIter=50, maxDepth=5, maxBins=300, seed=42,
    ),
    "rf": RandomForestRegressor(
        featuresCol="features", labelCol="total_amount",
        numTrees=100, maxDepth=8, maxBins=300, seed=42,
    ),
}

metrics = {}
print("\n── Model Comparison ─────────────────────────────────────────────────────")
print(f"{'Model':<30} {'RMSE':>10} {'R²':>10}")
print("-" * 52)
for name, regressor in MODELS.items():
    pipeline = build_pipeline(regressor)
    model = pipeline.fit(train_df)
    preds = model.transform(test_df)
    rmse = evaluator_rmse.evaluate(preds)
    r2 = evaluator_r2.evaluate(preds)
    metrics[name] = {"rmse": rmse, "r2": r2}
    print(f"{name:<30} ${rmse:>9.2f} {r2:>10.4f}")
    model.write().overwrite().save(OUTPUT_PATH + f"models/{name}/")

best = min(metrics, key=lambda k: metrics[k]["rmse"])
improvement = (metrics["lr"]["rmse"] - metrics[best]["rmse"]) / metrics["lr"]["rmse"] * 100
print(f"\nBest model: {best} — reduced RMSE by {improvement:.1f}% over LR baseline.")

metrics_path = OUTPUT_PATH + "models/metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nModels saved to {OUTPUT_PATH}models/, metrics written to {metrics_path}")
spark.stop()
