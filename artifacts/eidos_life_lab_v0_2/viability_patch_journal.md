# Viability Patch Journal

## Likely collapse cause
The prior world collapsed from lifecycle imbalance: high metabolism/death pressure, weak reproductive throughput, and no robust abiogenesis/reseeding path. This produced nutrient-rich, low-toxicity dead worlds with no recovery trigger.

## Files changed
- `experiments/eidos-life/src/life-engine.js`
- `experiments/eidos-life/src/scenarios.js`
- `experiments/eidos-life/src/app.js`
- `experiments/eidos-life/index.html`
- `experiments/eidos-life/package.json`
- `experiments/eidos-life/scripts/smoke_stable_ecology.mjs`
- `experiments/eidos-life/scripts/smoke_primordial_recovery.mjs`
- `experiments/eidos-life/scripts/smoke_extinction_event.mjs`
- `experiments/eidos-life/scripts/smoke_export_diagnostics.mjs`

## Added counters/config
Added cumulative counters: births, deaths, mutations, reseeds, primordial_blooms, extinction_events, near_extinction_events, collapse_events, recovery_events.

Added viability configuration for abiogenesis, near-extinction recovery, tuned metabolism/reproduction/death recycling, plus preset configuration snapshots exported in world state.

## Tests / smoke runs
- `node scripts/smoke_stable_ecology.mjs`
- `node scripts/smoke_primordial_recovery.mjs`
- `node scripts/smoke_extinction_event.mjs`
- `node scripts/smoke_export_diagnostics.mjs`
