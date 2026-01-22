# Core Loop Spec (Source → Artifacts)

This document maps the end-to-end dataflow for the v0.4.7.02 engine.

## Dataflow

**Source → Adapter/Projector → Normalizer → Reservoir → Predictor → Error → Surprise → Monitors → Gate → Recorder → Artifacts**

### 1) Source
- Sources: DRIVE, LOCAL (ARCHIVE/LONGTXT), KAGGLE, STREAM.
- Entry: `run_eidos_sentinel()` in `repo/src/eidos_brain/engine/eidos_v0_4_7_02.py`.
- Failure modes:
  - Missing/invalid paths for archive streams (throws `FileNotFoundError`).
  - Kaggle dependency missing when `use_kagglehub=True`.
  - Empty dataset (`ValueError`) or no numeric columns.

### 2) Adapter / Projector
- Adapter interface: `eidos_domain_adapters.get_domain_adapter()` (when available).
- Archive adapters: `_ArchiveProjector` + `_iter_*` loaders for tabular/text/binary.
- Behavior:
  - Numeric vectors are padded/projected to `FEATURES` via `AutoProjector`.
  - Text streams are embedded into fixed-size char ordinal vectors.
- Failure modes:
  - Non-numeric values in tabular data lead to exceptions on conversion.
  - Oversized binary files are capped by `ARCHIVE_BINARY_MAX_BYTES`.

### 3) Normalizer
- Stream normalizer: `OnlineVectorNormalizer` (Welford) in `eidos_v0_4_7_02.py`.
- Archive/table normalization: mean/std computed on input arrays (per-file or per-dataset).
- Failure modes:
  - Zero std values are guarded by epsilon replacement.
  - NaN handling is not explicitly configured beyond `np.nanmean` in some contexts.

### 4) Reservoir
- State update: RLS-based reservoir in `RLSReservoir` (Torch).
- State evolves via `listen()` and `adapt()` depending on surprise.
- Failure modes:
  - Potential dtype/device mismatches if `DTYPE` changes mid-run.

### 5) Predictor (Newtonian / Left Brain)
- Predictor: `NewtonianPredictor` uses last frame(s) to extrapolate.
- Residual computed as difference between predicted and actual frame.

### 6) Error & Surprise
- Error: residual norm is computed (L2/RMS) from model outputs.
- Surprise: `z_score` computed from EMA error + sigma baseline.
- Adaptive quantile gate: `z_thresh` updated every step to match target surprise rate.

### 7) Monitors
- Spectral monitor: FFT-based entropy/flatness metrics.
- Geometry monitor: optional reservoir state samples + box-count dimension.
- Thermodynamics / plasticity: `update_thermodynamics()` adjusts forgetting/lambda.

### 8) Gate
- Gate uses `is_surprise` + status color to control learning rate scale.
- Hippocampus can inhibit updates when a pattern is recognized.

### 9) Recorder
- `SessionRecorder` writes steps.csv, anomalies.jsonl, summary.json, report.txt.
- Receipts enriched with attribution + fingerprint + continuity metadata.

### 10) Artifacts
- See `AUDIT/INVENTORY_MAP.md` for output list.

## Unit Tests
- `forensic_test_suite.py` covers:
  - RLS initialization, adapt, listen.
  - Hippocampus encoding and recall.
  - Regime detection logic.
- Missing unit tests:
  - Adapters/projectors for archive streams.
  - Receipt bundle completeness (now covered by `tools/demo_smoke_test.py`).
