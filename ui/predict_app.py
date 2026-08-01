# ui/predict_app.py
import json

import pandas as pd
import streamlit as st
from pyspark.ml import PipelineModel
from pyspark.sql import Row

from work.spark_helper import get_spark
from work.constants import OUTPUT_PATH
from work.predict_helpers import (
    DAY_NAME_TO_SPARK_DOW,
    build_input_row,
    load_best_model_name,
    zone_dropdown_options,
)
from ui.analytics_section import render_analytics_section

st.set_page_config(page_title="Taxi Fare Predictor", page_icon="🚕")
st.title("🚕 NYC Yellow Taxi — Predicted Total Amount")


@st.cache_resource
def load_model_and_metrics():
    metrics_path = OUTPUT_PATH + "models/metrics.json"
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
    except FileNotFoundError:
        return None, None, None

    best_name = load_best_model_name(metrics)
    spark = get_spark("predict_ui")
    model = PipelineModel.load(OUTPUT_PATH + f"models/{best_name}/")
    best_info = {"name": best_name, **metrics[best_name]}
    return spark, model, best_info


@st.cache_resource
def load_zone_options():
    zones = pd.read_csv("data/taxi_zone_lookup.csv")
    return zone_dropdown_options(zones)


spark, model, best_info = load_model_and_metrics()

if model is None:
    st.error(
        "No trained models found at `data/output/models/`. "
        "Run `make run-ml` first, then reload this page."
    )
    st.stop()

zone_options = load_zone_options()
zone_labels = [label for label, _ in zone_options]
zone_id_by_label = dict(zone_options)

with st.form("predict_form"):
    pu_label = st.selectbox("Pickup zone", zone_labels)
    do_label = st.selectbox("Dropoff zone", zone_labels)
    trip_distance = st.number_input(
        "Trip distance (miles)", min_value=0.1, value=2.5, step=0.1
    )
    pickup_hour = st.slider("Pickup hour", 0, 23, 10)
    day_name = st.selectbox("Day of week", list(DAY_NAME_TO_SPARK_DOW.keys()))
    passenger_count = st.number_input(
        "Passenger count", min_value=1, max_value=6, value=1, step=1
    )
    payment_method = st.radio("Payment method", ["Credit Card", "Cash"])
    submitted = st.form_submit_button("Predict")

if submitted:
    row = build_input_row(
        pu_location_id=zone_id_by_label[pu_label],
        do_location_id=zone_id_by_label[do_label],
        trip_distance=trip_distance,
        pickup_hour=pickup_hour,
        day_name=day_name,
        passenger_count=int(passenger_count),
        payment_method=payment_method,
    )
    input_df = spark.createDataFrame([Row(**row)])
    prediction = model.transform(input_df).first()["prediction"]

    st.metric("Predicted total amount", f"${prediction:.2f}")
    st.caption(
        f"Served by {best_info['name'].upper()} — "
        f"RMSE ${best_info['rmse']:.2f}, R² {best_info['r2']:.4f}"
    )

st.divider()
render_analytics_section()
