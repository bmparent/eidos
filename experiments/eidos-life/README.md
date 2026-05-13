# Eidos Life v0.2 — Living Stream Lab

Experiment-only Three.js + Game of Life sandbox that adds an Eidos-style monitor, regime controller, novelty memory, organism tracking, and telemetry receipts.

## What v0.2 adds
- Modular ES modules in `src/` (engine, monitor, memory, scenarios, visualization, app).
- Multi-state typed-array cells (`alive`, `age`, `energy`, `species`, `memory`, `stress`) and invisible fields (energy/signal/anomaly/memory residue).
- Regime-driven adaptive rules (`CALIBRATING`, `GREEN`, `AMBER`, `RED`, `BLUE`, `VIOLET`).
- Pattern memory fingerprints + approximate known pattern tags.
- Approximate organism clustering with centroid/mass/threat stats.
- Telemetry recorder with browser export bundle: `{ manifest, summary, telemetry, interestingEvents }`.
- HUD toggles for surprise/memory/energy/outlines and a compact regime timeline strip.
- Disabled `EidosBackendBridge` stub for future backend integration.

## Run browser demo
```bash
python -m http.server 5173
```
Open: `http://localhost:5173/experiments/eidos-life/`

## Run tests
```bash
cd experiments/eidos-life
npm test
```

## Boundary with the real engine

This experiment does not modify the production Eidos Brain engine. It uses an Eidos-style monitor/controller loop to visualize the idea of a self-monitoring streaming intelligence codec. A future backend bridge may connect browser frames to the real Python engine, but that bridge is disabled by default and intentionally stubbed in this version.
