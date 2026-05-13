# Eidos Life v0.4 task notes

## Scope
- Strictly experiment-only work under `experiments/eidos-life/`.
- Do not alter production Eidos Brain engine files, sentinel monolith, benchmarks, proofs, trading modules, or existing proof artifacts.
- Keep Eidos as a regulator of world conditions, not a puppetmaster that spawns predesigned successful organisms.

## v0.4 architecture
- `src/life-engine.js`: typed-array world core, baseline Conway-compatible mode, evolutionary birth/survival mode, world state import/export.
- `src/genome.js`: compact genome registry, founder genomes, inheritance, bounded mutation, serialization.
- `src/ecology-fields.js`: nutrient, waste, signal, anomaly, and memory field updates with cheap diffusion/decay.
- `src/local-regimes.js`: low-resolution local regime map and local rule modifiers.
- `src/organism-tracker.js`: persistent approximate organism identity, active/dead organism records, split/merge/death/birth events, lineage graph.
- `src/evolution-telemetry.js`: world-level evolution metrics, event aggregation, export payloads.
- `src/prediction-ghost.js`: heuristic next-frame prediction, prediction error, ghost layer, surprise spark positions.
- `src/eidos-monitor.js`: global stream metrics and global regime selection.
- `src/pattern-memory.js`: bounded fingerprint memory and novelty estimate.
- `src/organisms.js`: compatibility cluster tracker retained for older paths/tests.
- `src/scenarios.js`: scenario presets and world seeding, including an evolutionary garden preset.
- `src/telemetry-recorder.js`: telemetry + interesting event bundle export with evolution/world-state data.
- `src/visualization.js`: Three.js observatory with instanced live-cell geometry, field overlays, local regimes, organism outlines/centroids, prediction ghost/sparks, and regime atmosphere.
- `src/eidos-backend-bridge.js`: disabled contract stub for future Python backend.
- `src/app.js`: browser wiring, controls, event feed, inspection panel, export/import lifecycle.

## Test command

```bash
cd experiments/eidos-life
npm test
```

## Manual verification

```bash
python -m http.server 5173
```

Open:

```text
http://localhost:5173/experiments/eidos-life/
```

Check that the observatory renders, cells show lineage/genome color differences, organism outlines and centroid markers are visible, event feed cards update from real tracker events, prediction ghost/error overlays can be toggled, Eidos Pulse changes the fields and visuals, local regime patches appear, and exports include evolution data plus world state.

## Known limitations
- Organism identity persistence is intentionally approximate: nearest centroid, mass similarity, and dominant genome/lineage overlap.
- Split and merge attribution is heuristic and tuned for readability rather than perfect biological accounting.
- Prediction ghost is a browser-side heuristic, not the production Eidos backend.
- Import restores enough state to continue a world, but UI controls may still reflect the current browser settings until changed.
- Novelty archive tracking is bounded and lightweight; it is meant as a signal, not proof of open-ended evolution.

## Next steps
- Add optional timeline scrub/rewind for organism lineage history.
- Add richer organism hover/raycast inspection.
- Add more scenario seeds for long-run ecology experiments.
- Add optional backend-connected mode behind an explicit feature flag.
