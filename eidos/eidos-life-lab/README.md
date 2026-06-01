# Eidos Life Lab

Eidos Life Lab is a local development app for exploring a synthetic Eidos Brain / Sentinel proving ground outside Colab. It runs as a normal browser app at `localhost`, with a Python FastAPI backend, a WebSocket live stream, HTTP fallback commands, a NumPy simulation engine, and a Vite + Three.js frontend.

The first version focuses on reliability: a visible 72x72 world, direct board editing, live metrics, checkpoint/export receipts, and no Colab iframe or notebook-specific transport code.

## Local Run

From the repo root:

```powershell
cd eidos-life-lab\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8787 --reload
```

In a second terminal:

```powershell
cd eidos-life-lab\frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The frontend proxies `/api` and `/ws` to the backend at `127.0.0.1:8787`.

## Convenience Scripts

Windows:

```powershell
.\eidos-life-lab\scripts\dev.ps1
```

Unix/macOS:

```bash
./eidos-life-lab/scripts/dev.sh
```

The scripts assume dependencies are already installed. Run the manual setup commands first if the backend virtual environment or frontend `node_modules` folder does not exist.

## Controls

- Play, pause, step, reset, clear, and random seed the world.
- Select scenario presets: `evolutionary_garden`, `rare_structure_emergence`, `stress_test`, `sparse_seed`, and `dense_seed`.
- Change mutation pressure and intervention mode.
- Toggle top-down and tilted camera views, zoom with the mouse wheel, and pan with right or middle drag.
- Use the board tools to inspect, toggle, birth, kill, paint disks, and inject glider, blinker, block, r-pentomino, or acorn patterns.
- Switch field views across lineage, energy, memory, signal, nutrient, waste, stress, and alive/dead monochrome.
- Scrub the client-side replay buffer without mutating backend state, then return to live mode.

## Artifacts

Exports and checkpoints are saved locally under:

```text
artifacts/eidos_life_lab/exports/
artifacts/eidos_life_lab/checkpoints/
```

Each export/checkpoint includes the flattened snapshot, genome registry, seed, timestamp, and artifact metadata. A matching `*_drive_manifest.json` is written next to each artifact. If `EIDOS_PROOF_DRIVE_DIR`, `EIDOS_ARTIFACT_ROOT`, or a mounted Colab Drive path is available and writable, the backend mirrors the JSON artifact to:

```text
Eidos_Brain_Proof_Phase/YYYY-MM-DD/<run_id>/
```

Drive is never required for the local lab to run.

## Performance Notes

- Default board size is 72x72.
- The backend broadcasts at 12 FPS by default, separate from simulation speed.
- The frontend keeps a maximum of 500 replay frames.
- Events and genome registry entries are capped.
- Rendering uses a single Three.js `InstancedMesh`.
- Render quality can be set to low, medium, or high.
- The first version intentionally avoids expensive postprocessing and external visualization libraries.

## How This Relates To Eidos Brain / Sentinel

The lab is a synthetic proving ground, not a change to the existing Eidos Brain model. It exposes regime labels and world metrics in a controlled, replayable setting so Sentinel-style ideas can be watched, edited, exported, and later connected to benchmark or incident-card workflows.

Current sentinel regimes are intentionally simple:

- `RED_COLLAPSE` when density is below `0.003`.
- `RED_BLOOM` when density is above `0.25`.
- `AMBER_THRASH` when birth candidates exceed `0.75` of alive cells.
- `AMBER_DOMINANCE` when the largest component exceeds `0.25` of alive cells.
- `GREEN_EDGE` otherwise.

## Backend Smoke Checks

From the repo root:

```powershell
python -m unittest discover -s eidos-life-lab\backend\tests
```

After starting the backend:

```powershell
Invoke-RestMethod http://127.0.0.1:8787/api/state
```

## Roadmap

1. Rare structure detector.
2. Persistent replay timeline.
3. Lineage takeover detector improvements.
4. Eidos Brain stream adapter.
5. Incident cards.
6. Long-run benchmark mode.
