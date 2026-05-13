# Eidos Life v0.2 — Living Stream Lab

Experiment-only Three.js + Game of Life sandbox that adds an Eidos-style monitor, regime controller, novelty memory, organism tracking, and telemetry receipts.

## What v0.2 adds
- Modular ES modules in `src/` (engine, monitor, memory, scenarios, visualization, app).
- Multi-state typed-array cells (`alive`, `age`, `energy`, `species`, `memory`, `stress`) and invisible fields (energy/signal/anomaly/memory residue).
- Regime-driven adaptive rules (`CALIBRATING`, `GREEN`, `AMBER`, `RED`, `BLUE`, `VIOLET`).
- Pattern memory fingerprints + approximate known pattern tags.
- Approximate organism clustering with centroid/mass/threat stats.
- Telemetry recorder with browser export bundle: `{ manifest, summary, telemetry, interestingEvents }`.
- Three.js observatory renderer with instanced 3D live cells, translucent field overlays, organism boundary cages, regime lighting/fog, camera drag/zoom, and pulse rings.
- HUD toggles for surprise/memory/energy/outlines and a compact visual regime timeline strip.
- Disabled `EidosBackendBridge` stub for future backend integration.

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

The page uses the real modular Three.js implementation in `src/app.js` and `src/visualization.js`.

Drag the observatory canvas to orbit the scene, and use the mouse wheel or trackpad scroll to zoom.

On Windows, if `python` is unavailable, use:

```powershell
py -m http.server 5173
```

## Three.js dependency

The experiment vendors `three.module.js` under `vendor/` so the demo can run even when the CDN is unavailable. If you want to refresh it, use Three.js `0.160.0` to match the original implementation.

## Run tests
```bash
cd experiments/eidos-life
npm test
```

## Boundary with the real engine

This experiment does not modify the production Eidos Brain engine. It uses an Eidos-style monitor/controller loop to visualize the idea of a self-monitoring streaming intelligence codec. A future backend bridge may connect browser frames to the real Python engine, but that bridge is disabled by default and intentionally stubbed in this version.
