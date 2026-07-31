# work/ml_features.py
from pyspark.sql import DataFrame
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline


# The 8 features fed to both models.
# Order matters — must stay consistent between local and cloud runs.
FEATURE_COLS = [
    "trip_distance",
    "pickup_hour",
    "day_of_week",
    "passenger_count",
    "pu_idx",       # StringIndexer output for PULocationID
    "do_idx",       # StringIndexer output for DOLocationID
    "pay_credit_card",
    "pay_cash",
]


def build_features(df: DataFrame) -> DataFrame:
    """
    Transforms preprocessed taxi DataFrame into ML-ready form:
      1. StringIndexer: PULocationID → pu_idx, DOLocationID → do_idx
      2. VectorAssembler: FEATURE_COLS → features vector

    Returns the DataFrame with added columns: pu_idx, do_idx, features.
    The target column (total_amount) is left untouched.
    """
    pu_indexer = StringIndexer(
        inputCol="PULocationID",
        outputCol="pu_idx",
        handleInvalid="keep",   # unseen zones at inference time get index 0
    )
    do_indexer = StringIndexer(
        inputCol="DOLocationID",
        outputCol="do_idx",
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="features",
        handleInvalid="skip",   # drop rows where any feature is null
    )
    pipeline = Pipeline(stages=[pu_indexer, do_indexer, assembler])
    return pipeline.fit(df).transform(df)


def build_pipeline(regressor) -> Pipeline:
    """
    Returns an untrained Pipeline: StringIndexer(PULocationID) + StringIndexer(DOLocationID)
    + VectorAssembler(FEATURE_COLS) + the given regressor stage.

    Unlike build_features(), this pipeline is meant to be fit once on raw
    training data and saved whole — feature transform and model travel
    together, so a loaded PipelineModel can predict directly from raw
    input columns (no separately-tracked StringIndexer mapping needed).
    """
    pu_indexer = StringIndexer(
        inputCol="PULocationID",
        outputCol="pu_idx",
        handleInvalid="keep",
    )
    do_indexer = StringIndexer(
        inputCol="DOLocationID",
        outputCol="do_idx",
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="features",
        handleInvalid="skip",
    )
    return Pipeline(stages=[pu_indexer, do_indexer, assembler, regressor])
