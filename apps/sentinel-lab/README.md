# Eidos / Sentinel Lab

An evidence-first Next.js operator console for the Eidos Brain / Sentinel Grand Proof workflow.

## What v0.2 does

- Runs deterministic, past-only engineering smoke projections for registered scenarios.
- Restricts the UI/API to engineering seeds `0` and `1`; held-out seeds are not accepted.
- Visualizes raw residual, quotient, persistence, and threshold traces.
- Produces a calibrated five-field incident-evidence card.
- Compares the engineering observer with rolling-z, EWMA, and CUSUM projections.
- Imports local JSON, JSONL, and text proof artifacts for operator inspection.
- Keeps G0–G6 locked and the current verdict at `BLOCKED_RESOURCE_BEFORE_HELDOUT`.
- Searches the public Kaggle catalog and pins an explicit dataset version plus exact file path.
- Produces a canonical SHA-256 real-data experiment lock.
- Enforces calibration/evaluation/sealed-held-out partitions and dispatches only to a separately configured full-engine runner.
- Ships the process-isolated Python runner under `services/sentinel-runner` for `EIDOS_BRAIN_UNIFIED_v0_4.7.02.py`.

## What it does not do

The browser-facing smoke simulator still does not run the full Torch reservoir/HDC engine. Real-data requests are sent to the separate runner only when `EIDOS_RUNNER_URL` and `EIDOS_RUNNER_TOKEN` are configured. Even a successful real-data engineering run cannot establish Grand Proof acceptance, held-out generalization, or production readiness.

## Real-data safety contract

- Kaggle downloads use `owner/dataset/versions/N` and an exact file path; missing files fail closed.
- Labels are removed before engine frames and metadata are generated.
- Missing-value, mean, and scale parameters are fit on calibration rows only.
- Evaluation labels are opened only after the full engine returns frozen predictions.
- Held-out rows are digest-committed but excluded from engineering execution.
- Each full-engine job runs in its own process because the current engine helper temporarily replaces module-global configuration.
- Dispatch and status APIs require a separate operator bearer token so the public lab cannot launch compute jobs.
- Every result remains `REAL_DATA_ENGINEERING`, advances zero gates, and preserves `BLOCKED_RESOURCE_BEFORE_HELDOUT`.

## Local checks

```bash
npm install
npm test
npm run lint
npm run build
npm run dev
```

Runner checks:

```bash
cd ../../services/sentinel-runner
PYTHONPATH=. python -m unittest discover -s tests -v
```

For a repeatable browser pass, point `CHROME_BIN` at Chrome or Chrome Headless Shell and run `npm run qa:browser`. The script checks desktop and mobile rendering, the smoke-run path, tab navigation, locked gates, and held-out seed rejection.

## Repository placement

The app lives at `apps/sentinel-lab`. The external execution service lives at `services/sentinel-runner` and imports the repository's canonical monolithic 0.4.7.02 engine. Development occurs on a feature branch; `main` is changed only through review and merge.
