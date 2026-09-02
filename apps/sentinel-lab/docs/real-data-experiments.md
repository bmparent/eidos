# Sentinel Lab v0.2 real-data experiments

## Topology

Sentinel Lab is split deliberately:

1. The Next.js application on Vercel searches the catalog, validates operator input, computes a canonical experiment lock, and displays status/artifacts.
2. The primary execution adapter launches one persistent Vercel Sandbox per experiment. The Sandbox clones the exact deployment commit into Vercel's managed Python image and owns Kaggle credentials, dataset bytes, a pinned CPU-only Torch runtime, resource limits, job artifacts, and the full Eidos process.
3. A FastAPI/Docker runner remains available when experiments outgrow Sandbox or need dedicated GPU/durable infrastructure.
4. Kaggle is an input registry, not a live backend. The runner downloads an explicit version and exact file once per locked job.

The Next.js request does not execute the full engine. The current `run_stream_once` helper temporarily replaces module-global configuration, so the request starts an isolated VM process and returns a job receipt; the interface polls status separately. The v0.2 Sandbox profile runs the full Eidos 0.4.7.02 code path with 256 reservoir units and 2,048 hippocampal dimensions so a 25,000-row CPU diagnostic fits the engineering budget. The effective overrides are written into the run manifest and do not represent the full-scale 2,000/10,000 configuration.

## Evidence states

| State | Input | Engine | Proof effect |
|---|---|---|---|
| `ENGINEERING_SMOKE` | Deterministic synthetic scenarios | Browser-safe observer projection | Zero gates |
| `REAL_DATA_ENGINEERING` | Pinned Kaggle version/file | Full Eidos 0.4.7.02 | Zero gates |
| Held-out proof | Sealed preregistered corpus | Resource-qualified locked protocol | Not enabled in v0.2 |

The v0.2 runner never converts a real-data engineering result into held-out proof.

## Causal data contract

For rows `0..N-1`, the immutable split is:

- calibration: `[0, floor(0.20N))`
- evaluation: `[floor(0.20N), floor(0.80N))`
- sealed holdout: remaining rows

Source order is preserved unless the operator locks an explicit ordering column. Shuffling and class balancing are unavailable in the v0.2 real-data path.

For feature column `j`, imputation, mean, and scale use calibration rows only:

`median_j = median(X_cal[:, j])`

`mu_j = mean(impute(X_cal[:, j], median_j))`

`sigma_j = max(std(impute(X_cal[:, j], median_j)), 1e-12)`

Every engine row is transformed with those frozen values. The label column, declared exclusions, ordering column, and label-like columns are never allowed into the feature matrix. If source width exceeds 64, a seed-locked Gaussian projection scaled by `1/sqrt(source_width)` produces the engine frame; narrower inputs are zero-padded.

## Label isolation

The runner constructs two objects:

- `PreparedDataset`: transformed frames, source row indices, and label-free metadata.
- `LabelVault`: evaluation labels and a one-way commitment to held-out labels/row indices.

Only `PreparedDataset.make_gen_factory()` is passed to `run_stream_once`; its metadata contains neither labels nor calibration/evaluation membership. After the engine returns step rows, `evaluate_frozen_predictions` aligns evaluation frames and opens evaluation labels. The trace exported to the application contains scores and thresholds but no labels. Held-out labels are never evaluated in this evidence class.

## Vercel Sandbox deployment contract

Set `EIDOS_EXECUTION_BACKEND=sandbox`, `EIDOS_OPERATOR_TOKEN`, and `KAGGLE_API_TOKEN` in Vercel. Git-based deployments supply `VERCEL_GIT_COMMIT_SHA` automatically; a manual deployment must set `EIDOS_SOURCE_COMMIT` to an exact 40-character SHA. Optional budget variables control vCPU, session timeout, maximum rows, and concurrency.

The default budget is intentionally conservative: 4 vCPU/8 GB, 45 minutes, 25,000 rows, and one concurrent job. The default dataset lock pins `dhoogla/cicids2017` version 3, `WebAttacks-Thursday-no-metadata.parquet`, and SHA-256 `7db47b2bf97ad58c3556ee25e8e1eb1e697cd391670733833865d0e84d8ed82a`. Each persistent Sandbox keeps only its newest snapshot and expires it after seven days. Terminal status and artifact reads stop the resumed VM after retrieval. Failed runs expose `runner.log` plus a traceback only through the operator-authenticated artifact route.

Before accepting the detached engine process, the app verifies the managed Python interpreter and the exact launcher file. It records `launcher_command.json` with the Vercel command ID, inspects that command at a bounded cadence, and converts missing or prematurely exited launchers into an explicit failed receipt plus `launcher_failure.log`. A stale `QUEUED` job is retired automatically instead of occupying runner capacity indefinitely.

The application exposes status and artifacts only through operator-authenticated route handlers. The Kaggle and Vercel credentials never enter the browser.

## External-runner fallback

Set `EIDOS_EXECUTION_BACKEND=external`, `EIDOS_RUNNER_URL`, `EIDOS_RUNNER_TOKEN`, and a separate `EIDOS_OPERATOR_TOKEN`. The browser supplies the operator credential only for dispatch/status calls; Vercel uses the runner credential server-to-server. The runner requires its bearer token, `KAGGLE_API_TOKEN`, a canonical engine path, persistent job storage, and a conservative concurrency limit. TLS, network restrictions, resource quotas, and artifact retention belong to that deployment.

Until every selected-backend preflight condition passes, the application can prepare an experiment lock but refuses dispatch instead of pretending a full-engine run occurred.
