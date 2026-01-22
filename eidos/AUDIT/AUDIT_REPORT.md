# Eidos Audit Report (v0.4.7.02)

## Summary
This report documents correctness, demo readiness, and claim alignment for the engine and service entrypoints. All findings list severity, file/function, reproduction, and patch status.

## Findings

### CRITICAL
- None found in static review.

### HIGH
1) **Receipts missing mandatory fields (session/hash/attribution bundle).**
   - Location: `SessionRecorder.record_anomaly()` in `repo/src/eidos_brain/engine/eidos_v0_4_7_02.py`.
   - Repro: Run any session and inspect `anomalies.jsonl`; fields like `session_id`, `config_hash`, `synaptic_hash_current`, `baseline`, `evidence_paths` are absent.
   - Patch: Added receipt bundle fields and ensured attribution + fingerprints are always present.

2) **Continuity hash not recorded; no continuity mismatch detection.**
   - Location: `run_eidos_sentinel()` in engine.
   - Repro: Run a session; no continuity hash in summary or state capsule output.
   - Patch: Added `continuity_hash`, `reservoir_state_hash`, `reservoir_weights_hash` to summary and a `state_capsule.json` artifact; optional mismatch detection with `continuity_expected_hash`.

### MEDIUM
1) **Demo local config points to missing folder.**
   - Location: `repo/configs/demo_local.yaml` references `./demo_data/sample_folder`.
   - Repro: Run demo local; fails due to missing directory.
   - Patch: Created `repo/demo_data/sample_folder` and populated sample files.

2) **No retry/backoff for network fetches.**
   - Location: stream + kaggle ingestion in engine.
   - Repro: interrupt network; engine fails without retry.
   - Patch: Documented (future optional).

### LOW
- Minor duplication in config paths and log output; no functional impact.

## Tests
- Added `tools/demo_smoke_test.py` to validate artifacts and receipt completeness.
