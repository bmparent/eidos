# Plain-Language Test Analysis - 2026-05-14

## What the task attempted

The task attempted to rebuild the Colab-style Eidos Life Lab experiment as a local development app that runs in a normal browser window. The new app needed a backend simulation, live WebSocket streaming, HTTP fallback, a Three.js board, direct editing tools, metrics, Sentinel regime labels, exports, checkpoints, and clear local run commands.

## Why the test matters

This matters because the lab is meant to become a reliable proving ground for Eidos Brain and Sentinel ideas. If the local app cannot start, stream state, accept edits, or save artifacts, it is not useful as a proof environment.

## What was tested

- The backend engine can produce a complete 72x72 snapshot.
- The backend engine includes the required metrics.
- Board-edit commands mutate the world.
- The backend can run from a local FastAPI server.
- `GET /api/state` returns JSON.
- `POST /api/command` accepts a cell edit and changes state.
- `ws://127.0.0.1:8787/ws` accepts a connection, returns a snapshot, accepts a command, and returns an ack plus a fresh snapshot.
- `POST /api/export` writes a JSON export.
- `POST /api/checkpoint` writes a JSON checkpoint.
- Drive manifests are written even when Drive copy is skipped.
- The frontend dependencies install.
- The frontend builds.
- The frontend dependency audit is clean at moderate severity and above.

## What passed

- Backend unit smoke tests passed: 2 tests.
- Backend virtual environment install passed.
- Frontend install passed after updating Vite.
- Frontend production build passed.
- `npm audit --audit-level moderate` passed with 0 vulnerabilities.
- Live HTTP state endpoint passed.
- Live WebSocket connection and command path passed.
- Live board mutation through HTTP command passed.
- Live export/checkpoint endpoint checks passed.

## What failed

No acceptance-critical check failed after the Vite update. The first `npm install` surfaced moderate Vite/esbuild advisories, and that was fixed by moving the frontend dev dependency to Vite `^8.0.12`.

## What artifacts were generated

- `artifacts/eidos_life_lab/exports/export_20260514T003115_723576Z.json`
- `artifacts/eidos_life_lab/exports/export_20260514T003115_723576Z_drive_manifest.json`
- `artifacts/eidos_life_lab/checkpoints/checkpoint_20260514T003116_275192Z.json`
- `artifacts/eidos_life_lab/checkpoints/checkpoint_20260514T003116_275192Z_drive_manifest.json`

## What was saved locally

The local app source was saved under `eidos-life-lab/`. The validation export and checkpoint were saved under `artifacts/eidos_life_lab/`. The proof journal and this analysis were saved under `docs/proof_runs/2026-05-14/`.

## What was saved to Google Drive

Nothing was saved to Google Drive in this run.

## What remains uncertain

- The frontend was build-tested, but not visually screenshot-tested in a browser.
- The app has not yet been run as a long-duration benchmark.
- The current Sentinel regime logic is deliberately simple and should be treated as a baseline display layer, not a proven detector.
- The replay buffer is client-side only and is not yet a durable timeline artifact.

## What should happen next

The next useful step is a rare structure detector that produces artifacts from local runs. After that, a persistent replay timeline would make the lab stronger as a repeatable proof tool.
