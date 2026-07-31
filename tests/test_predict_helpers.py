# tests/test_predict_helpers.py
import pytest
import pandas as pd


class TestPaymentFlags:
    def test_credit_card(self):
        from work.predict_helpers import payment_flags
        assert payment_flags("Credit Card") == {"pay_credit_card": 1, "pay_cash": 0}

    def test_cash(self):
        from work.predict_helpers import payment_flags
        assert payment_flags("Cash") == {"pay_credit_card": 0, "pay_cash": 1}

    def test_unknown_method_raises(self):
        from work.predict_helpers import payment_flags
        with pytest.raises(ValueError):
            payment_flags("Bitcoin")


class TestBuildInputRow:
    def test_maps_all_fields(self):
        from work.predict_helpers import build_input_row
        row = build_input_row(
            pu_location_id=161,
            do_location_id=236,
            trip_distance=2.5,
            pickup_hour=10,
            day_name="Monday",
            passenger_count=1,
            payment_method="Credit Card",
        )
        assert row == {
            "PULocationID": 161,
            "DOLocationID": 236,
            "trip_distance": 2.5,
            "pickup_hour": 10,
            "day_of_week": 2,
            "passenger_count": 1,
            "pay_credit_card": 1,
            "pay_cash": 0,
        }


class TestLoadBestModelName:
    def test_picks_lowest_rmse(self):
        from work.predict_helpers import load_best_model_name
        metrics = {
            "lr": {"rmse": 4.82, "r2": 0.91},
            "gbt": {"rmse": 2.41, "r2": 0.96},
            "rf": {"rmse": 2.55, "r2": 0.96},
        }
        assert load_best_model_name(metrics) == "gbt"


class TestZoneDropdownOptions:
    def test_formats_and_sorts_labels(self):
        from work.predict_helpers import zone_dropdown_options
        df = pd.DataFrame([
            {"LocationID": 236, "Borough": "Manhattan", "Zone": "Upper East Side North"},
            {"LocationID": 132, "Borough": "Queens", "Zone": "JFK Airport"},
        ])
        options = zone_dropdown_options(df)
        assert options == [
            ("JFK Airport, Queens", 132),
            ("Upper East Side North, Manhattan", 236),
        ]
