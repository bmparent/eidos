# Prediction Experiments

Run locally:
- `python -m pip install -e .`
- `python -m pytest tests -q`
- `python -m eidos_brain.prediction.run_world_telemetry --config config/prediction/world_sources.example.yaml --out ../../artifacts/predictions/world_telemetry_real_world_corpus_v0`
- `python -m eidos_brain.prediction.run_world_telemetry --fixture --out ../../artifacts/predictions/world_telemetry_smoke`
- `python -m eidos_brain.prediction.run_market_forecast --fixture --out ../../artifacts/predictions/market_forecast_smoke`
- `python -m eidos_brain.prediction.evaluate_due_predictions --ledger ../../artifacts/predictions/ledger --out ../../artifacts/predictions/evaluations_smoke`

GitHub Actions workflow: `.github/workflows/eidos_prediction_experiments.yml`.

Optional secrets: `RCLONE_CONFIG_DRIVE`, `EIDOS_GDRIVE_REMOTE_PATH`.
If missing, sync is skipped safely and manifests record `google_drive_sync: skipped_missing_secret`.

## Real World Corpus v0

World telemetry can now read a low-cost public corpus without paid API keys.
Configured source adapters:

- `rss`
- `atom`
- `gdelt_doc_api`
- `arxiv_api`
- `local_jsonl`
- `fixture`

The fixture adapter remains the deterministic CI/smoke path. Non-fixture runs read sources from `config/prediction/world_sources.example.yaml` or from a compatible config passed with `--config`.

Each run writes:

- `source_events.jsonl` with normalized source rows, including skipped/error rows.
- `predictions.jsonl` with append-compatible prediction rows.
- `manifest.json` with corpus counts, status counts, config hash, data snapshot hash, and `google_drive_sync`.
- `summary.csv`, `experiment_report.md`, and `experiment_status.md`.

Normalized event schema:

```json
{
  "event_id": "stable deterministic id",
  "source_id": "source identifier",
  "source_type": "rss|atom|gdelt_doc_api|arxiv_api|local_jsonl|fixture",
  "url": "canonical URL if available",
  "title": "title",
  "summary": "summary or abstract",
  "text": "combined usable text",
  "published_at_utc": "source publication time if available",
  "observed_at_utc": "time Eidos saw it",
  "language": "optional language",
  "domain": "optional domain/source domain",
  "raw_hash": "hash of raw source record",
  "license_note": "short source/license note",
  "ingest_status": "ok|skipped|error",
  "error": null
}
```

The adapters fail closed at the source-row level: a network or parse issue becomes an `ingest_status: error` row in `source_events.jsonl` and the run continues. If no usable events are collected, the command still writes a manifest and report but records no pending predictions.

Financial disclaimer: Research experiment only. Not financial advice. No trading execution.
