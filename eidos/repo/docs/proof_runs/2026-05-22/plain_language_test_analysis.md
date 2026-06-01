# Plain-Language Test Analysis — 2026-05-22

## What the task attempted

The task extended the existing Eidos Brain prediction experiment framework from PR #19 so the world telemetry experiment can ingest a real public corpus instead of depending only on two fixture events.

## Why the test matters

The prediction experiment is only useful as proof if the source data is auditable. The new ingestion path records where each source row came from, when Eidos observed it, whether it was usable, and why any source failed.

## What was tested

Tests covered fixture ingestion, RSS, Atom, GDELT DOC API JSON, arXiv API Atom output, local JSONL ingestion, error rows, the world telemetry fixture CLI, and a local JSONL corpus CLI smoke run. The full repo test suite was also run.

## What passed

The focused prediction tests passed: 14 passed. The full repo test suite passed with 78 passed and 4 skipped. Fixture mode still produced world telemetry artifacts. A public corpus smoke run collected 26 source rows, 25 usable events, and wrote 12 pending predictions.

## What failed

The first pytest command failed before collection because a user-site `pytest_recording` plugin was incompatible with the installed `urllib3`. Rerunning with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` passed. The public GDELT source timed out during the live smoke run and was recorded as one `ingest_status=error` row instead of failing the run.

## Artifacts generated

Local artifacts were saved under:

`artifacts/proof_runs/2026-05-22/real_world_corpus_v0/`

The latest fixture smoke run is:

`artifacts/proof_runs/2026-05-22/real_world_corpus_v0/world_telemetry_fixture_smoke/20260522T200347Z/`

The latest public corpus smoke run is:

`artifacts/proof_runs/2026-05-22/real_world_corpus_v0/world_telemetry_public_smoke/20260522T200400Z/`

## What was saved locally

The public corpus smoke saved `source_events.jsonl`, `predictions.jsonl`, `manifest.json`, `summary.csv`, `experiment_report.md`, and `experiment_status.md`. The proof folder also includes `test_results.txt` and `drive_manifest.json`.

## What was saved to Google Drive

Nothing was copied to Google Drive. `EIDOS_PROOF_DRIVE_DIR` and `EIDOS_ARTIFACT_ROOT` were not set, and `/content/drive/MyDrive` was not present.

## What remains uncertain

Live public feeds can be temporarily unavailable or slow. The code records those failures as source rows, but a future hardening pass could add retries or per-source backoff.

## What should happen next

The next PR-sized step is to add a small source-health report that summarizes which sources are reliable over time and which are repeatedly timing out.
