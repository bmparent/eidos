# Codex Journal - 2026-05-14

## What happened today

Built a new local development app named `eidos-life-lab` for the Eidos Life Lab experiment. The work created a FastAPI backend, a NumPy simulation engine, a WebSocket live stream, HTTP fallback endpoints, a Vite frontend, and a Three.js board renderer that runs in a normal local browser window instead of Colab.

## What was accomplished

- Added a standalone Eidos Life Lab backend under `eidos-life-lab/backend`.
- Added a standalone Vite frontend under `eidos-life-lab/frontend`.
- Added local run scripts for Windows and Unix/macOS.
- Added a README with setup, controls, artifact paths, performance notes, Sentinel relationship, and roadmap.
- Added backend smoke tests for snapshot shape and command mutation behavior.
- Proved `/api/state`, `/ws`, and `toggle_cell` against a live local backend.
- Proved export and checkpoint endpoints write JSON artifacts and Drive manifests.
- Added `.gitignore` entries for generated local dependencies, build output, Python bytecode, and generated Life Lab artifact JSON.

## Tests and commands run

- `python -m unittest discover -s eidos-life-lab\backend\tests` - passed, 2 tests.
- `python -m venv .venv` from `eidos-life-lab\backend` - passed.
- `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` from `eidos-life-lab\backend` - passed.
- `.\eidos-life-lab\backend\.venv\Scripts\python.exe -m unittest discover -s eidos-life-lab\backend\tests` - passed, 2 tests.
- `npm install` from `eidos-life-lab\frontend` - passed after Vite was bumped to a fixed current major.
- `npm run build` from `eidos-life-lab\frontend` - passed. Vite reported a chunk-size warning from the Three.js bundle.
- `npm audit --audit-level moderate` from `eidos-life-lab\frontend` - passed, 0 vulnerabilities.
- Live backend smoke script from repo root - passed:
  - Started `python -m uvicorn app:app --host 127.0.0.1 --port 8787`.
  - Confirmed `GET /api/state` returned a 72x72 snapshot.
  - Confirmed `POST /api/command` with `toggle_cell` changed cell `(5, 5)`.
  - Confirmed `ws://127.0.0.1:8787/ws` accepted a connection, returned a snapshot, accepted a `step` command, and returned an ack plus snapshot.
- Live artifact smoke script from repo root - passed:
  - Confirmed `POST /api/export` created a JSON export.
  - Confirmed `POST /api/checkpoint` created a JSON checkpoint.
  - Confirmed both artifact calls wrote Drive manifests.

## Problems encountered

- Initial `npm install` with Vite 5 reported two moderate dev-server advisories through Vite/esbuild. I updated Vite to `^8.0.12`, reran install, and confirmed `npm audit --audit-level moderate` reports 0 vulnerabilities.
- `npm run build` reports a chunk-size warning because Three.js is bundled into the first version. The build still succeeds and this is acceptable for the local lab baseline.
- Google Drive copy was skipped because no writable Drive root was configured in this local environment.

## What changed

The change is additive. It introduces a new local app under `eidos-life-lab/`, proof-run notes under `docs/proof_runs/2026-05-14/`, and a root `.gitignore` for generated local build/dependency/artifact byproducts.

## What did not change

Existing Eidos Brain core model behavior was not changed. No reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, forecasting logic, or domain-profile behavior were modified.

## Artifacts generated

- `artifacts/eidos_life_lab/exports/export_20260514T003115_723576Z.json`
- `artifacts/eidos_life_lab/exports/export_20260514T003115_723576Z_drive_manifest.json`
- `artifacts/eidos_life_lab/checkpoints/checkpoint_20260514T003116_275192Z.json`
- `artifacts/eidos_life_lab/checkpoints/checkpoint_20260514T003116_275192Z_drive_manifest.json`

## Google Drive archive status

Drive copy was skipped.

- Drive root used: unknown.
- Drive folder used: unknown.
- Files copied: none.
- Files skipped: the local export/checkpoint artifacts were not mirrored.
- Reason: `EIDOS_PROOF_DRIVE_DIR` not set, `EIDOS_ARTIFACT_ROOT` not set, and no mounted Colab Drive path found.

## Thoughts on improvement

The local app is now good enough for interactive proof work, but the next useful improvements are persistence and detection rather than more visuals. The first follow-up should be a rare structure detector with artifact receipts, followed by a persistent replay timeline.

## Where to improve next

Add a rare structure detector that can identify gliders, blinkers, blocks, r-pentomino-like structures, and acorn-derived growth in exported runs.

## Anything that stands out

The local FastAPI/WebSocket route avoids the Colab iframe/WebSocket fragility cleanly. The main remaining frontend limitation is that browser interaction was validated through build and live endpoint smoke tests, not a full visual browser screenshot pass.

## End-of-task summary

1. Files changed: new `eidos-life-lab/` app, root `.gitignore`, this journal, and the plain-language analysis.
2. Whether core behavior changed: no existing Eidos Brain core behavior changed.
3. Tests added or skipped: backend smoke tests added; browser screenshot testing skipped for this first pass.
4. Repo-root commands run: backend unittest, live backend smoke, live artifact smoke; frontend commands run from `eidos-life-lab/frontend`.
5. Artifacts generated: export/checkpoint JSON plus Drive manifests under `artifacts/eidos_life_lab/`.
6. Plain-language analysis written: yes, `docs/proof_runs/2026-05-14/plain_language_test_analysis.md`.
7. Journal entry written: yes, this file.
8. Google Drive copy status: skipped because no Drive root was configured or mounted.
9. Known limitations: no persistent replay file yet, no rare structure detector yet, no visual screenshot QA yet, and the Three.js bundle triggers a build-size warning.
10. Follow-up tasks not implemented: rare structure detector, persistent replay timeline, Eidos Brain stream adapter, incident cards, and long-run benchmark mode.
