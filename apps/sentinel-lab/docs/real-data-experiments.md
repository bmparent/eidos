# Sentinel Lab v0.2 real-data experiments

## Topology

Sentinel Lab is split deliberately:

1. The Next.js application on Vercel searches the catalog, validates operator input, computes a canonical experiment lock, and displays status/artifacts.
2. The external runner owns Kaggle credentials, dataset bytes, Torch, resource limits, durable job directories, and the full Eidos engine process.
3. Kaggle is an input registry, not a live backend. The runner downloads an explicit version and exact file once per locked job.

Vercel must not execute the full engine inside a request. A 2,000-unit reservoir plus HDC state is a resource-qualified batch workload, and the current `run_stream_once` helper temporarily replaces module-global configuration. The runner therefore gives each request a separate child process.

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

Only `PreparedDataset.make_gen_factory()` is passed to `run_stream_once`. After the engine returns step rows, `evaluate_frozen_predictions` aligns evaluation frames and opens evaluation labels. The trace exported to the application contains scores and thresholds but no labels. Held-out labels are never evaluated in this evidence class.

## Deployment contract

Vercel requires `EIDOS_RUNNER_URL`, `EIDOS_RUNNER_TOKEN`, and a separate `EIDOS_OPERATOR_TOKEN`. The browser supplies the operator credential only for dispatch/status calls; Vercel uses the runner credential server-to-server. The runner requires the runner bearer token, `KAGGLE_API_TOKEN`, a canonical engine path, persistent job storage, and a conservative concurrency limit. TLS, network restrictions, resource quotas, and artifact retention belong to the runner deployment.

Until both Vercel variables exist, the application can prepare and export an experiment lock but returns `RUNNER_NOT_CONFIGURED` instead of pretending a full-engine run occurred.
