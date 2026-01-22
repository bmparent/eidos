# Inventory Map

## Entrypoints

### CLI / main entrypoints
- `EIDOS_BRAIN_UNIFIED_v0_4.7.02.py` → `run_eidos_sentinel()` and `if __name__ == "__main__"` for monolithic runs.
- `repo/src/eidos_brain/engine/eidos_v0_4_7_02.py` → `run(config)` + `run_eidos_sentinel()` and `if __name__ == "__main__"`.
- `repo/src/eidos_brain/service/daemon.py` → `main()` for daemon sessions (`--once`, `--loop`).
- `repo/src/eidos_brain/service/api.py` → FastAPI `POST /run` and `main()` for API server.
- `repo/src/eidos_brain/demo/demo_runner.py` → `main()` launches engine + dashboard.
- `repo/src/eidos_brain/demo/demo_streamer.py` → `main()` launches local demo streaming server.
- `repo/scripts/test_connectivity.py` → CLI connectivity probe.
- `forensic_test_suite.py` → `unittest` runner for engine diagnostics.

### Config files & validation
- `repo/configs/default.yaml`, `demo_local.yaml`, `demo_stream.yaml` load via `repo/src/eidos_brain/utils/config.py` (`load_config`).
- Runtime config overrides are applied in `repo/src/eidos_brain/engine/eidos_v0_4_7_02.py::_apply_runtime_config()`.

### Outputs / artifacts
- Session artifacts are written under `<artifact_root>/eidos_brain_archive/<session_id>_...`:
  - `session_meta.json`
  - `steps.csv`
  - `anomalies.jsonl`
  - `clusters.json`
  - `summary.json`
  - `report.txt`
  - `state_capsule.json` (new)
- Additional artifacts:
  - `compression/` artifacts (`bicameral_stream_*`) via `store_memory_artifact()`.
  - `sentinel_forensics/` top-k surprise artifacts.
  - `geometry/` (fractal metrics) when enabled.
  - `reservoir_checkpoints/` and `hippocampus/` snapshots.
  - `run_manifest.json` via `write_run_manifest()`.

## Dependency / Dataflow Graph (imports + runtime)

```
service/daemon.py
  -> utils/config.load_config
  -> engine.adapters.run_session
     -> engine/eidos_v0_4_7_02.run
        -> run_eidos_sentinel
           -> domain adapters (eidos_domain_adapters)
           -> OnlineVectorNormalizer
           -> RLSReservoir + Newtonian predictor
           -> Surprise gate (z-score + quantile)
           -> Spectral + geometry monitors
           -> Hippocampus (HDC/VSA)
           -> SessionRecorder (steps/anomalies/report)
           -> store_memory_artifact (compression/forensics)

service/api.py
  -> same engine path via adapters

demo/demo_runner.py
  -> service/daemon.py (engine)
  -> demo/dashboard.py (Streamlit)
```

## Required Environment Variables (observed)
- `EIDOS_DATA_SOURCE` / `EIDOS_DATA_SOURCE_TYPE`
- `EIDOS_ARTIFACT_ROOT`
- `GOOGLE_API_KEY` (if NL config or Gemini features)
- `HIVE_IMAGE_DIGEST` (optional provenance metadata)
