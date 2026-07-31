# cs675

## Run the Prediction UI

Prerequisites: models must already be trained (`make run-ml`) and the zone lookup downloaded (`make download-zones`).

```bash
make download-zones
make run-ui
```

Open http://localhost:8501, fill in trip details, and click "Predict" to see the estimated `total_amount` from the best-performing trained model.
