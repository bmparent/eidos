# RNG Null Proof: Eidos Brain Adapter Runtime

The RNG null-proof harness has two runtime layers:

- **Baseline predictor mode does not need `torch`.** The default online frequency/transition baseline is pure Python and is suitable for smoke tests, null controls, and reproducibility checks when optional numerical dependencies are not installed.
- **The Sentinel-backed scaffold can run without `torch`.** Sentinel evidence and conservative proof reporting remain available even when the full Eidos Brain engine cannot be imported.
- **The Eidos Brain-backed proof requires `torch`.** The Brain-backed path depends on the unified engine file, `EIDOS_BRAIN_UNIFIED_v0_4.7.02.py`, and its `RLS_Reservoir` implementation. That engine imports `torch`, so Brain-backed proof runs require the optional Eidos Brain dependency set.
- **`official_proof_ready` stays `false` until the current gate passes.** A run should not be promoted to official Brain-backed proof status unless `torch` imports successfully, the engine path/import succeeds, `RLS_Reservoir` is available, and the Eidos Brain predictor can initialize.

## Install optional Eidos Brain runtime dependencies

```bash
python -m pip install -r requirements-eidos-brain.txt
```

## Check the runtime environment

```bash
python scripts/check_eidos_brain_runtime.py
```

The check prints the Python version, `torch` import status/version, CUDA availability, selected device, engine-file existence, `RLS_Reservoir` import status, and whether the Eidos Brain RNG predictor can initialize.

Exit codes:

- `0`: Eidos Brain predictor can initialize.
- `2`: an optional dependency such as `torch` is missing.
- `3`: the engine file, import path, or engine initialization failed.

## Brain-backed smoke command

```bash
python tools/run_rng_null_proof.py --suite smoke --seed 42 --frames 2000 --predictor eidos_brain --out artifacts/rng_null_proof_smoke_eidos_brain
```

If the Brain-backed gate does not pass, keep the run marked as not official rather than forcing `official_proof_ready` to `true`.
