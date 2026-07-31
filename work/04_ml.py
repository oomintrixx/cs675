# work/04_ml.py
from work.spark_helper import get_spark
from work.constants import OUTPUT_PATH
from work.ml_features import build_features

from pyspark.ml.regression import LinearRegression, GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = get_spark("04_ml")

# ── Load preprocessed data ──────────────────────────────────────────────────────
print("Loading clean taxi data...")
df = spark.read.parquet(OUTPUT_PATH + "taxi_clean/")
print(f"  Rows: {df.count():,}")

# ── Feature assembly ────────────────────────────────────────────────────────────
print("Building feature vectors...")
featured = build_features(df)
print(f"  Rows after feature assembly: {featured.count():,}")

# ── Train / test split ──────────────────────────────────────────────────────────
train_df, test_df = featured.randomSplit([0.8, 0.2], seed=42)
print(f"  Train: {train_df.count():,} | Test: {test_df.count():,}")

evaluator_rmse = RegressionEvaluator(
    labelCol="total_amount", predictionCol="prediction", metricName="rmse"
)
evaluator_r2 = RegressionEvaluator(
    labelCol="total_amount", predictionCol="prediction", metricName="r2"
)

# ── Baseline: Linear Regression ─────────────────────────────────────────────────
print("\nTraining Linear Regression (baseline)...")
lr = LinearRegression(
    featuresCol="features",
    labelCol="total_amount",
    maxIter=100,
    regParam=0.1,
    elasticNetParam=0.0,
)
lr_model = lr.fit(train_df)
lr_preds = lr_model.transform(test_df)
lr_rmse = evaluator_rmse.evaluate(lr_preds)
lr_r2   = evaluator_r2.evaluate(lr_preds)
print(f"  LR  — RMSE: ${lr_rmse:.2f} | R²: {lr_r2:.4f}")

# ── Main model: Gradient Boosted Trees ──────────────────────────────────────────
print("\nTraining Gradient Boosted Trees...")
gbt = GBTRegressor(
    featuresCol="features",
    labelCol="total_amount",
    maxIter=50,
    maxDepth=5,
    maxBins=300,  # must exceed the ~262 distinct PU/DOLocationID zones
    seed=42,
)
gbt_model = gbt.fit(train_df)
gbt_preds = gbt_model.transform(test_df)
gbt_rmse = evaluator_rmse.evaluate(gbt_preds)
gbt_r2   = evaluator_r2.evaluate(gbt_preds)
print(f"  GBT — RMSE: ${gbt_rmse:.2f} | R²: {gbt_r2:.4f}")

# ── Alternative: Random Forest ──────────────────────────────────────────────────
# Bags many trees over bootstrap samples and averages them, rather than
# sequentially fitting residuals like GBT — much less sensitive to the
# handful of extreme total_amount values still left in the tail.
print("\nTraining Random Forest...")
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="total_amount",
    numTrees=100,
    maxDepth=8,
    maxBins=300,  # must exceed the ~262 distinct PU/DOLocationID zones
    seed=42,
)
rf_model = rf.fit(train_df)
rf_preds = rf_model.transform(test_df)
rf_rmse = evaluator_rmse.evaluate(rf_preds)
rf_r2   = evaluator_r2.evaluate(rf_preds)
print(f"  RF  — RMSE: ${rf_rmse:.2f} | R²: {rf_r2:.4f}")

# ── Comparison table ─────────────────────────────────────────────────────────────
print("\n── Model Comparison ─────────────────────────────────────────────────────")
print(f"{'Model':<30} {'RMSE':>10} {'R²':>10}")
print("-" * 52)
print(f"{'Linear Regression (baseline)':<30} ${lr_rmse:>9.2f} {lr_r2:>10.4f}")
print(f"{'Gradient Boosted Trees':<30} ${gbt_rmse:>9.2f} {gbt_r2:>10.4f}")
print(f"{'Random Forest':<30} ${rf_rmse:>9.2f} {rf_r2:>10.4f}")
best_rmse = min(gbt_rmse, rf_rmse)
improvement = (lr_rmse - best_rmse) / lr_rmse * 100
print(f"\nBest tree model reduced RMSE by {improvement:.1f}% over baseline.")

# ── Save models ──────────────────────────────────────────────────────────────────
lr_model.write().overwrite().save(OUTPUT_PATH + "models/lr/")
gbt_model.write().overwrite().save(OUTPUT_PATH + "models/gbt/")
rf_model.write().overwrite().save(OUTPUT_PATH + "models/rf/")
print(f"\nModels saved to {OUTPUT_PATH}models/")

spark.stop()
