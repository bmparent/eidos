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
- Enforces calibration/evaluation/sealed-held-out partitions and dispatches to an isolated Vercel Sandbox or a separately configured external runner.
- Ships the process-isolated Python runner under `services/sentinel-runner` for `EIDOS_BRAIN_UNIFIED_v0_4.7.02.py`.
- Polls live job stages, renders frozen-prediction metrics, and retrieves authenticated audit artifacts in the app.

## What it does not do

The browser-facing smoke simulator still does not run the full Torch reservoir/HDC engine. The real-data tab launches isolated compute only when the selected backend, Kaggle credential, exact Git commit, and operator authorization all pass preflight. Even a successful real-data engineering run cannot establish Grand Proof acceptance, held-out generalization, or production readiness.

## Real-data safety contract

- Kaggle downloads use `owner/dataset/versions/N` and an exact file path; missing files fail closed.
- Labels and split membership are removed before engine frames and metadata are generated.
- Missing-value, mean, and scale parameters are fit on calibration rows only.
- Evaluation labels are opened only after the full engine returns frozen predictions.
- Held-out rows are digest-committed but excluded from engineering execution.
- Each full-engine job runs in its own process because the current engine helper temporarily replaces module-global configuration.
- The Sandbox backend clones the deployment's exact Git commit, receives Kaggle credentials only as server-side environment state, and retains one expiring snapshot per job.
- The default Sandbox budget is one concurrent job, 4 vCPU/8 GB, 25,000 rows, and 45 minutes.
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

The app lives at `apps/sentinel-lab`. The execution package lives at `services/sentinel-runner` and imports the repository's canonical monolithic 0.4.7.02 engine. Vercel Sandbox is the primary backend; the FastAPI/Docker service remains an external-compute fallback. Development occurs on a feature branch; `main` is changed only through review and merge.
