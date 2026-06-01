# Codex Journal — 2026-05-22

## What happened today

Real World Corpus v0 was added to the existing Eidos Brain prediction experiment framework. The work stayed inside the current `eidos_brain.prediction` package and did not rebuild the prediction framework.

## What was accomplished

- Added source adapters for `rss`, `atom`, `gdelt_doc_api`, `arxiv_api`, `local_jsonl`, and `fixture`.
- Normalized source rows into an auditable common event schema.
- Preserved fixture mode for deterministic smoke tests.
- Wired non-fixture world telemetry runs to read configured public sources.
- Added `source_events.jsonl` and richer corpus metadata in `manifest.json`.
- Updated docs, example config, and the GitHub Actions world telemetry command.
- Added focused tests for all source adapters and a non-fixture local JSONL CLI smoke run.

## Tests and commands run

- `python -m pip install -e .` — passed.
- `python -m pytest tests/test_prediction_sources.py tests/test_world_telemetry_smoke.py -q` — initial pre-install subprocess import failure, then passed with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/test_prediction_sources.py tests/test_world_telemetry_smoke.py tests/test_market_forecast_smoke.py tests/test_prediction_ledger.py tests/test_prediction_reports.py tests/test_prediction_scoring.py -q` — 14 passed.
- `python -m eidos_brain.prediction.run_world_telemetry --fixture --out artifacts/proof_runs/2026-05-22/real_world_corpus_v0/world_telemetry_fixture_smoke` — passed.
- `python -m eidos_brain.prediction.run_world_telemetry --config config/prediction/world_sources.example.yaml --out artifacts/proof_runs/2026-05-22/real_world_corpus_v0/world_telemetry_public_smoke --max-events 25 --timeout-seconds 10` — passed.
- `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests -q` — 78 passed, 4 skipped.

## Problems encountered

The local pytest environment still auto-loads a broken `pytest_recording` plugin unless `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is set. The first public source run also revealed non-chronological feed order, so the source-window calculation was corrected to use min/max timestamps. GDELT timed out during live smoke validation and was recorded as an audited error row.

## What changed

The change added ingestion and reporting code only. It did not change reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, forecasting logic, or domain-profile behavior.

## What did not change

Core model behavior was untouched. Existing ledger append semantics and prediction rows remain intact. Fixture mode remains available and tested.

## Artifacts generated

- `artifacts/proof_runs/2026-05-22/real_world_corpus_v0/test_results.txt`
- `artifacts/proof_runs/2026-05-22/real_world_corpus_v0/drive_manifest.json`
- `artifacts/proof_runs/2026-05-22/real_world_corpus_v0/world_telemetry_fixture_smoke/20260522T200347Z/`
- `artifacts/proof_runs/2026-05-22/real_world_corpus_v0/world_telemetry_public_smoke/20260522T200400Z/`

## Google Drive archive status

Google Drive copy was skipped. `EIDOS_PROOF_DRIVE_DIR` and `EIDOS_ARTIFACT_ROOT` were not set, and `/content/drive/MyDrive` was not present. No files were copied.

## Thoughts on improvement

The ingestion layer is intentionally small and auditable. The most useful next improvement would be source-health tracking so repeated source failures become visible without inspecting every JSONL row.

## Where to improve next

Add retries/backoff and a compact source-health report for scheduled prediction experiments.

## Anything that stands out

The public smoke run still produced usable predictions even when one source timed out because source errors are now rows, not process failures. That is the right behavior for a low-cost public corpus.

## End-of-task summary

1. Files changed: source adapters, world telemetry runner, tests, docs, config, workflow, proof receipts.
2. Whether core behavior changed: no.
3. Tests added or skipped: source adapter tests and local JSONL CLI smoke added; optional dependency tests still skipped as before.
4. Repo-root commands run: editable install, focused pytest, full pytest, fixture smoke, public corpus smoke.
5. Artifacts generated: local proof folder under `artifacts/proof_runs/2026-05-22/real_world_corpus_v0/`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: skipped because no configured Drive path was available.
9. Known limitations: live public sources can time out; current behavior records this but does not retry.
10. Follow-up tasks not implemented: source-health history, retry/backoff, richer corpus quality metrics.
