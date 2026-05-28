# Prediction Experiments

Run locally:
- `python -m pip install -e .`
- `python -m pytest tests -q`
- `python -m eidos_brain.prediction.run_world_telemetry --fixture --out ../../artifacts/predictions/world_telemetry_smoke`
- `python -m eidos_brain.prediction.run_market_forecast --fixture --out ../../artifacts/predictions/market_forecast_smoke`
- `python -m eidos_brain.prediction.evaluate_due_predictions --ledger ../../artifacts/predictions/ledger --out ../../artifacts/predictions/evaluations_smoke`

GitHub Actions workflow: `.github/workflows/eidos_prediction_experiments.yml`.

Optional secrets: `RCLONE_CONFIG_DRIVE`, `EIDOS_GDRIVE_REMOTE_PATH`.
If missing, sync is skipped safely and manifests record `google_drive_sync: skipped_missing_secret`.

Financial disclaimer: Research experiment only. Not financial advice. No trading execution.
