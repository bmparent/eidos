# Eidos / Sentinel Lab

An evidence-first Next.js operator console for the Eidos Brain / Sentinel Grand Proof workflow.

## What v0.1 does

- Runs deterministic, past-only engineering smoke projections for registered scenarios.
- Restricts the UI/API to engineering seeds `0` and `1`; held-out seeds are not accepted.
- Visualizes raw residual, quotient, persistence, and threshold traces.
- Produces a calibrated five-field incident-evidence card.
- Compares the engineering observer with rolling-z, EWMA, and CUSUM projections.
- Imports local JSON, JSONL, and text proof artifacts for operator inspection.
- Keeps G0–G6 locked and the current verdict at `BLOCKED_RESOURCE_BEFORE_HELDOUT`.

## What it does not do

The browser-facing smoke simulator does not run the full Torch reservoir/HDC engine and cannot establish Grand Proof acceptance, natural-domain value, or production readiness. It is an operator surface and deterministic engineering tool. Full-engine artifacts can be imported now; a remote resource-qualified runner can be attached later.

## Local checks

```bash
npm install
npm test
npm run lint
npm run build
npm run dev
```

For a repeatable browser pass, point `CHROME_BIN` at Chrome or Chrome Headless Shell and run `npm run qa:browser`. The script checks desktop and mobile rendering, the smoke-run path, tab navigation, locked gates, and held-out seed rejection.

## Repository placement

The app is intended to live at `apps/sentinel-lab` on a feature branch based on `codex/eidos-meaningful-surprise-grand-proof-v1`. The repository root remains unchanged and `main` is not modified.
