# Eidos Life v0.2 task notes

## Scope
- Strictly experiment-only work under `experiments/eidos-life/`.
- Do not alter production Eidos Brain engine files, sentinel monolith, benchmarks, proofs, or trading modules.

## v0.2 architecture
- `src/life-engine.js`: wrapped-grid Conway core with multi-state channels and field layers.
- `src/eidos-monitor.js`: computes stream metrics + regime selection + rule presets.
- `src/pattern-memory.js`: bounded fingerprint memory and novelty estimate.
- `src/organisms.js`: approximate connected-cluster tracking.
- `src/scenarios.js`: scenario presets and world seeding.
- `src/telemetry-recorder.js`: run telemetry + interesting event extraction + export bundle.
- `src/visualization.js`: Three.js mesh and overlay coloring.
- `src/eidos-backend-bridge.js`: disabled contract stub for future Python backend.
- `src/app.js`: browser wiring/HUD lifecycle.

## Test command
```bash
cd experiments/eidos-life
npm test
```

## Known limitations
- Pattern detection is heuristic and approximate.
- Organism identity persistence is nearest-centroid approximate.
- Timeline is visual-only (no rewind yet).
- Backend bridge is intentionally disabled/no-op.

## Next steps
- Add stronger motif detection and explainability snippets.
- Add timeline rewind buffer and frame scrubbing.
- Add optional backend-connected mode behind explicit feature flag.
