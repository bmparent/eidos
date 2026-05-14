# Eidos Life v0.4 - Evolutionary World Layer

Experiment-only static browser sandbox for a small self-evolving Eidos Life world. It combines the original typed-array Life engine, Eidos-style monitor metrics, the Three.js observatory renderer, and a v0.4 evolutionary layer with heritable genomes, ecology fields, persistent organisms, local regimes, prediction ghosts, and telemetry receipts.

## What v0.4 adds
- Heritable compact genomes for living cells, with trait-driven birth, survival, energy use, stress tolerance, mutation, cohesion, and color identity.
- Genome and lineage IDs stored in typed arrays, without one heavy object per cell.
- Birth inheritance from neighboring live cells, with mutation pressure from local anomaly, stress, regime, and selected mutation mode.
- Ecology fields for nutrients, waste, signal, anomaly, and memory residue.
- Low-resolution local regime zones (`GREEN`, `AMBER`, `RED`, `BLUE`, `VIOLET`, `CALIBRATING`) that alter local physics and selection pressure.
- Persistent organism tracking with approximate identity, split/merge/death events, mass/centroid histories, lineage graph data, and fitness/novelty/threat summaries.
- Evolution telemetry for diversity, dominant genomes/lineages, birth/death/split/merge rates, mutation pressure, extinction count, local regime diversity, and prediction error.
- Heuristic prediction ghost and prediction-error sparks so the observatory shows where the world was expected to move next.
- Event feed cards driven by real tracker/telemetry events.
- Autonomous world controls for evolution, mutation pressure, Eidos intervention, speed, organism inspection, and world export/import.

## View the actual Three.js render locally

From repo root:

```bash
python -m http.server 5173
```

Then open:

```text
http://localhost:5173/experiments/eidos-life/
```

Or from the experiment folder:

```bash
npm run serve
```

The page uses the real modular Three.js implementation in `src/app.js` and `src/visualization.js`. Drag the observatory canvas to orbit the scene, and use the mouse wheel or trackpad scroll to zoom.

On Windows, if `python` is unavailable, use:

```powershell
py -m http.server 5173
```

## Three.js dependency

The experiment vendors `three.module.js` under `vendor/` so the demo can run even when the CDN is unavailable. If you want to refresh it, use Three.js `0.160.0` to match the original implementation.

## Controls
- `Evolution`: enables or disables inheritance, ecology selection pressure, local regimes, and genome-aware behavior.
- `Mutation`: sets low, medium, high, or adaptive mutation pressure.
- `Intervention`: keeps Eidos passive, guardian-style, or experimental as a regulator of conditions rather than a direct organism designer.
- `World speed`: slows down, normalizes, or accelerates simulation steps.
- `Inspect`: follows the largest, oldest, newest, or most novel active organism.
- Field toggles: show surprise/anomaly, memory, energy/nutrients, organism outlines, and prediction ghost/error overlays.
- `Eidos Pulse`: injects an anomaly pulse into the world and triggers a visible field ripple.
- `Export Bundle`: downloads telemetry, interesting events, evolution data, and world state.
- `Export World` / `Import World`: save and restore a reproducible browser-side world state.

## Run tests

```bash
cd experiments/eidos-life
npm test
```

## Export data

The telemetry export includes:

```json
{
  "manifest": {},
  "summary": {},
  "telemetry": [],
  "interestingEvents": [],
  "evolution": {
    "genomes": [],
    "lineages": [],
    "organisms": []
  },
  "worldState": {}
}
```

## Boundary with the real engine

This experiment does not modify or prove the production Eidos Brain engine. It uses an Eidos-style monitor/controller loop as a browser-side experiment layer. A future backend bridge may connect browser frames to the real Python engine, but that bridge remains disabled by default and intentionally stubbed.

## Long-run observation workflow (v0.4 harness hardening)

- **Summary Export is the recommended long-run artifact.** It is compact and generated directly from summary methods (not from full bundle payloads).
- **Export Bundle** is still available for deep debugging but can be very large on long runs.
- **Autosave is metadata-only by default** every **5,000** generations (`CHECKPOINT_MODE_DEFAULT = "metadata_only"`). It stores run metadata, compact telemetry/world stats, and a warning that full world state is not autosaved.
- **Manual Save Checkpoint** is explicit: it attempts a guarded local save only for smaller payloads and otherwise downloads a checkpoint JSON file.
- **Recommended cadence**:
  - Summary Export every **5,000–10,000** generations.
  - Export World every **25,000–50,000** generations.
  - Avoid Export Bundle unless specifically needed.
- HUD run counters:
  - `gen`: current engine generation.
  - `total`: monotonic in-session generation counter.
  - `epoch`: increments per detected or explicit reset/import/seed/scenario reset.
  - `resets`: total reset events this session.
- On full page reload, session counters may restart unless reloaded from saved state.
- `RED` regime indicates collapse risk / sparse-risk pressure and does **not** inherently mean the world is dead.

> This remains a browser-side experiment harness (v0.4), not production Eidos Brain proof.
