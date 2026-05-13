# Eidos Life Engine

A self-contained Three.js Game of Life experiment where an Eidos-inspired monitor acts as the **life engine**.

## Concept

The world is a Conway-style cellular automaton. Each generation is treated as a streaming frame. The Eidos layer watches that stream and computes a small set of brain-like operating metrics:

- **surprise** — how much the grid changed from the previous frame
- **entropy** — how alive/dead and balanced the grid is
- **plasticity** — how aggressively the world should adapt or explore
- **compression** — an approximate RLE-style compressibility score
- **novelty** — how different the current grid fingerprint is from recent memory
- **collapse risk** — whether the world is freezing, dying, or saturating

The resulting regime color changes the rules of life:

- **GREEN** — close to normal Conway B3/S23 behavior
- **AMBER** — higher exploration/mutation during unstable dynamics
- **RED** — collapse protection; low-energy reseeding prevents dead worlds
- **BLUE** — novel but controlled geometry
- **VIOLET** — rare/high-novelty geometry with more exploratory pressure
- **CALIBRATING** — early warmup period

## Run locally

From the repo root:

```bash
python -m http.server 5173
```

Then open:

```text
http://localhost:5173/experiments/eidos-life/
```

A static server is required because the demo uses ES modules/import maps.

## Files

```text
experiments/eidos-life/
  index.html   # standalone Three.js demo
  README.md    # this file
```

## How this connects to Eidos Brain

The repo’s proof plan describes Eidos Brain as a self-monitoring streaming intelligence codec that learns live streams, compresses predictable behavior, preserves meaningful anomalies, monitors its own internal state, and emits human-readable incident receipts. This demo is the visual version of that idea: the automaton is the stream, the Eidos layer is the self-monitor, and the regime color decides how the living world adapts.

## Next upgrade path

This first version intentionally avoids touching the production Eidos benchmark/proof code. The clean next steps are:

1. Split the Eidos life rules into `src/life-engine.js`.
2. Add a Vite app wrapper for easier development and deployment.
3. Add a JSON telemetry export so the demo emits the same style of receipts as the proof work.
4. Bridge to the real Python Eidos engine through a local websocket service.
5. Add scenario presets: stable oscillator, glider storm, collapse, noisy regime shift, and rare structure emergence.
6. Add tests for rule transitions and metric ranges.

## Product idea

This can become the visual explanation layer for Eidos Brain: a living world where compression, anomaly preservation, sentinel regimes, and self-monitoring are immediately visible instead of abstract.
