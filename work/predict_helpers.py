# work/predict_helpers.py
"""Pure helper functions bridging the Streamlit form to the trained PipelineModel's
expected raw input columns. Kept free of Spark/Streamlit imports so they're fast
to unit test."""

DAY_NAME_TO_SPARK_DOW = {
    "Sunday": 1,
    "Monday": 2,
    "Tuesday": 3,
    "Wednesday": 4,
    "Thursday": 5,
    "Friday": 6,
    "Saturday": 7,
}


def payment_flags(method: str) -> dict:
    if method == "Credit Card":
        return {"pay_credit_card": 1, "pay_cash": 0}
    if method == "Cash":
        return {"pay_credit_card": 0, "pay_cash": 1}
    raise ValueError(f"Unknown payment method: {method}")


def build_input_row(
    pu_location_id: int,
    do_location_id: int,
    trip_distance: float,
    pickup_hour: int,
    day_name: str,
    passenger_count: int,
    payment_method: str,
) -> dict:
    row = {
        "PULocationID": pu_location_id,
        "DOLocationID": do_location_id,
        "trip_distance": trip_distance,
        "pickup_hour": pickup_hour,
        "day_of_week": DAY_NAME_TO_SPARK_DOW[day_name],
        "passenger_count": passenger_count,
    }
    row.update(payment_flags(payment_method))
    return row


def load_best_model_name(metrics: dict) -> str:
    return min(metrics, key=lambda name: metrics[name]["rmse"])


def zone_dropdown_options(zone_df) -> list:
    options = [
        (f"{row.Zone}, {row.Borough}", int(row.LocationID))
        for row in zone_df.itertuples()
    ]
    return sorted(options, key=lambda pair: pair[0])
