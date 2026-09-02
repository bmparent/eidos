# Eidos Sentinel Runner v0.2

This service is the resource-qualified execution plane for Sentinel Lab. Vercel remains the UI/control plane; this container downloads one explicitly versioned Kaggle file and runs the full `EIDOS_BRAIN_UNIFIED_v0_4.7.02.py` engine in a child process.

## Safety boundary

- Every request carries a canonical SHA-256 run lock, recomputed by the runner.
- The exact Kaggle dataset version and file path are mandatory. Missing files fail closed; no "first file" fallback exists.
- Labels are removed before frames or metadata enter the Eidos engine.
- Imputation and z-score parameters are fit on the calibration partition only.
- Evaluation labels are used only after the engine returns frozen step predictions.
- The sealed holdout partition is committed by digest and never sent to the engineering run.
- Every job is a separate Python process because the current engine's programmatic helper temporarily replaces module-global configuration.
- Results remain `REAL_DATA_ENGINEERING`, keep G0-G6 locked, and preserve `BLOCKED_RESOURCE_BEFORE_HELDOUT`.

## Required environment

```text
EIDOS_RUNNER_TOKEN=<long random bearer token>
KAGGLE_API_TOKEN=<Kaggle API token>
EIDOS_ENGINE_PATH=/absolute/path/to/EIDOS_BRAIN_UNIFIED_v0_4.7.02.py
EIDOS_JOB_ROOT=/durable/job/storage
EIDOS_MAX_CONCURRENT_JOBS=1
```

The Vercel project receives only:

```text
EIDOS_RUNNER_URL=https://runner.example.com
EIDOS_RUNNER_TOKEN=<same bearer token>
EIDOS_OPERATOR_TOKEN=<separate credential held by the operator>
```

Kaggle credentials belong on the runner, not in the browser. `KAGGLE_API_TOKEN` may also be added to Vercel if authenticated/private catalog search is needed, but it is never returned to the client.

## Local verification

From this directory:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

From the repository root, build the execution image with:

```bash
docker build -f services/sentinel-runner/Dockerfile -t eidos-sentinel-runner:0.2 .
```

The container needs a persistent volume at `/var/lib/eidos/jobs` and enough CPU/GPU memory for the configured 2,000-unit reservoir. Production deployments should place TLS and workload/resource controls in front of the single API worker.
