# Eidos Life Lab v0.2 Codex Journal

## Repo inspection notes
- Experiment is implemented in `experiments/eidos-life` as a browser-based JS simulation.
- Core engine is `experiments/eidos-life/src/life-engine.js`.
- Field dynamics are in `experiments/eidos-life/src/ecology-fields.js`.
- UI/controls are in `experiments/eidos-life/index.html` and `experiments/eidos-life/src/app.js`.
- Visualization/Three.js renderer is in `experiments/eidos-life/src/visualization.js`.
- Scenario presets are in `experiments/eidos-life/src/scenarios.js`.
- There is no FastAPI service for this lab in current repo; /state and /step compatibility is represented by engine/export state usage.
- Existing tests are node tests under `experiments/eidos-life/tests/*.test.mjs`.

## Implementation plan
1. Extend simulation state with Higgs-like scalar inertia field (phi), toxicity, heat/stress, and memory residue while preserving current arrays and behavior.
2. Add organism-level life variables (mass, coupling, familiarity/novelty, regime, action, reproduction counters) in compact typed arrays and include summary/sample exports.
3. Implement metabolism, action costs, death/recycling, and reproduction/mutation coupled to phi/effective mass.
4. Add deterministic seed support and scenario presets for v0.2-focused regimes.
5. Add event receipts + artifact writer (`events.jsonl`, `summary.json`, `run_manifest.json`) with bounded in-memory event storage.
6. Keep existing state export fields stable; add optional detail endpoints in exported state shape (`detail` argument / additional sections).
7. Update frontend HUD + layer selector + controls for nutrients/toxicity/phi/memory/stress/regime layers and Higgs toggles.
8. Add/extend tests for metabolism, Higgs effects, reproduction, regimes, API compatibility-style state keys, artifacts, and determinism.

## Risks / notes
- Current project is frontend-centric; API semantics are implemented as state export methods rather than HTTP routes.
- Keep changes incremental and avoid heavy deps.

## Test run notes
- Ran `npm test` in `experiments/eidos-life`.
- Most tests pass after v0.2 changes; remaining failing legacy tests:
  1. `evolutionary birth inherits parent genome and lineage` (expects stronger parent genome inheritance wiring than current v0.2 simplified reproduction path).
  2. `evolution telemetry export includes genomes lineages and organisms` (expects telemetry payload section still named/structured exactly as previous export contract).
- New v0.2 tests in `tests/life-v0_2.test.mjs` pass.
