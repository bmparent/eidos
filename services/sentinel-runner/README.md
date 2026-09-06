# Eidos Sentinel Runner v0.2

This service is the resource-qualified execution plane for Sentinel Lab. Vercel remains the UI/control plane; this container downloads one explicitly versioned Kaggle file and runs the full `EIDOS_BRAIN_UNIFIED_v0_4.7.02.py` engine in a child process.

The same package also includes `sentinel_runner.sandbox_launcher`, the bootstrap used by the Vercel Sandbox backend. It creates an isolated environment with `uv`, resolves the pinned CPU-only Torch build, records authenticated diagnostic artifacts on failure, and then hands the immutable request to the same CLI/job implementation used by the container.

## Safety boundary

- Every request carries a canonical SHA-256 run lock, recomputed by the runner.
- The exact Kaggle dataset version and file path are mandatory. Missing files fail closed; no "first file" fallback exists.
- Labels are removed before frames or metadata enter the Eidos engine.
- Imputation and z-score parameters are fit on the calibration partition only.
- Prediction telemetry is committed and hashed before evaluation labels are used. Missing, duplicate, non-finite, out-of-partition, or inconsistent evaluation predictions fail closed.
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

For the Vercel Sandbox backend, the project receives:

```text
EIDOS_EXECUTION_BACKEND=sandbox
EIDOS_OPERATOR_TOKEN=<separate credential held by the operator>
KAGGLE_API_TOKEN=<Kaggle API token>
```

For the external-runner fallback, the Vercel project receives:

```text
EIDOS_RUNNER_URL=https://runner.example.com
EIDOS_RUNNER_TOKEN=<same bearer token>
EIDOS_OPERATOR_TOKEN=<separate credential held by the operator>
```

Kaggle credentials belong on the runner, not in the browser. `KAGGLE_API_TOKEN` may also be added to Vercel if authenticated/private catalog search is needed, but it is never returned to the client.

## Local verification

From the repository root, using Python 3.14 and the pinned CPU dependencies:

```bash
python -m pip install uv
uv pip install --python python --torch-backend cpu services/sentinel-runner
python -m unittest discover -s services/sentinel-runner/tests -v
python services/sentinel-runner/scripts/verify_full_engine.py --profile cpu_engineering --output artifacts/sentinel-ci/cpu_engineering
python services/sentinel-runner/scripts/verify_full_engine.py --profile cpu_mechanisms --output artifacts/sentinel-ci/cpu_mechanisms
```

The full-engine script generates 1,000 synthetic rows locally and runs 800 through the actual Torch engine, scoring all 600 evaluation rows while excluding the 200-row holdout. It emits the input, metrics, effective configuration, code/input hashes, and measured geometry/telemetry. These are integration checks, not real-world detection validation.

From the repository root, build the execution image with:

```bash
docker build -f services/sentinel-runner/Dockerfile -t eidos-sentinel-runner:0.2 .
```

The bridge accepts only the profiles in `sentinel_runner/profiles.py`: standard 256/2,048, experimental 256/2,048 with four leak bands and TraceSeal, and full-size 2,000/10,000. Requested overrides must match the configuration returned by the engine. The full-size profile requires `EIDOS_ENABLE_FULL_CAPACITY=1` on both this runner and the control plane, and substantially larger dedicated compute; it has not been qualified by the included checks. See [the complete contract](../../apps/sentinel-lab/docs/real-data-experiments.md).

Authenticated artifact retrieval includes `engine_trace.jsonl`, `engine_diagnostics.json`, evaluation metrics/trace, manifests, and failure logs. The container needs a persistent volume at `/var/lib/eidos/jobs`; production deployments should place TLS and workload/resource controls in front of the API worker. The existing active-job scan is not a distributed admission lock; concurrent multi-instance deployment requires a queue or shared admission controller.
