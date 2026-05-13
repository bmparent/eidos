# Codex Task — Upgrade Eidos Life Engine

You are working in the existing `bmparent/eidos` repo. The current prototype lives at:

```text
experiments/eidos-life/index.html
```

It is a standalone Three.js Game of Life demo where an Eidos-inspired monitor acts as the life engine. Do **not** rewrite the core Eidos proof/benchmark system. Treat this as a contained visual/demo layer unless explicitly instructed otherwise.

## Goal

Turn the standalone Eidos Life prototype into a clean, testable, repo-friendly demo that visually explains Eidos Brain as a self-monitoring streaming intelligence codec.

The experience should show:

1. A Conway-style cellular automaton rendered in Three.js.
2. The grid treated as a live stream.
3. Eidos-style metrics computed each generation:
   - surprise
   - entropy
   - plasticity
   - compression estimate
   - novelty/familiarity
   - collapse risk
4. Sentinel-style regimes:
   - CALIBRATING
   - GREEN
   - AMBER
   - RED
   - BLUE
   - VIOLET
5. Rule modulation based on the regime.
6. A telemetry receipt/export path that can later align with Eidos proof artifacts.

## Constraints

- Keep this work contained under `experiments/eidos-life/` unless there is a clear reason to add shared docs.
- Do not modify trading/Kalshi/PolySentinel behavior.
- Do not claim this proves Eidos performance. It is a visualization and experimental interface.
- Keep it runnable locally with simple commands.
- Prefer deterministic seeds for scenario presets.
- Keep dependencies light.

## Phase 1 — Modularize the current prototype

Create this structure:

```text
experiments/eidos-life/
  index.html
  package.json
  src/
    main.js
    render/
      three-scene.js
      hud.js
    engine/
      life-grid.js
      eidos-life-engine.js
      metrics.js
      regimes.js
      scenarios.js
    telemetry/
      recorder.js
      export.js
  tests/
    metrics.test.js
    regimes.test.js
  README.md
  CODEX_TASK.md
```

Use Vite for the app wrapper unless the repo already has a preferred frontend build system.

## Phase 2 — Engine interface

Create a small engine API:

```js
const engine = new EidosLifeEngine({
  width: 72,
  height: 72,
  seed: 42,
  scenario: 'glider_storm',
});

const frame = engine.step();

// frame shape
{
  generation: 123,
  grid: Uint8Array,
  metrics: {
    liveRatio: 0.18,
    surprise: 0.12,
    entropy: 0.68,
    plasticity: 0.31,
    compressionRatio: 2.4,
    novelty: 0.22,
    collapseRisk: 0.0
  },
  regime: 'GREEN',
  rule: {
    birth: [3],
    survive: [2, 3],
    mutationRate: 0.0002,
    reseedRate: 0
  }
}
```

## Phase 3 — Scenarios

Add scenario presets:

- `classic_random`
- `glider_storm`
- `oscillator_lab`
- `collapse_test`
- `noisy_regime_shift`
- `rare_structure_emergence`

Each preset should produce predictable, visually different dynamics.

## Phase 4 — Telemetry receipts

Add a telemetry recorder that can export:

```text
experiments/eidos-life/artifacts/sample_run/
  run_manifest.json
  telemetry.jsonl
  summary.json
```

In browser mode, use a downloadable JSON/JSONL export rather than writing to disk.

The summary should answer:

- total generations
- regime counts
- mean surprise
- mean entropy
- mean plasticity
- mean compression ratio
- collapse events
- top novelty events

## Phase 5 — Optional real Eidos bridge

Add a clearly optional bridge plan, not enabled by default:

```text
browser Three.js demo
   -> websocket client
      -> local Python Eidos adapter
         -> real Eidos metrics / Sentinel regime
         -> telemetry artifact folder
```

Do not implement the Python bridge unless specifically instructed. Add interface stubs only:

```js
class EidosBackendClient {
  connect(url) {}
  sendFrame(gridFrame) {}
  onMetrics(callback) {}
}
```

## Phase 6 — Tests

Add lightweight tests for:

- entropy is near zero for all-dead/all-live grids
- entropy rises for balanced grids
- surprise is zero when grids match
- surprise rises when many cells change
- RED triggers on collapse-like states
- GREEN triggers on stable moderate states
- AMBER triggers on high plasticity/surprise
- BLUE/VIOLET trigger on novelty thresholds

## Acceptance criteria

- `npm install` works from `experiments/eidos-life/`.
- `npm run dev` launches the demo.
- `npm test` runs the metrics/regime tests.
- The demo still looks premium and dimensional in Three.js.
- The UI clearly communicates that Eidos is the life engine, not just a visual skin.
- The original standalone `index.html` behavior is preserved or improved.
- No unrelated files are changed.

## Design direction

Visual style should feel like Eidos Brain: dark, luminous, scientific, alive, premium, not gimmicky. It should communicate intelligence, monitoring, compression, anomaly preservation, and self-regulation.

Use tasteful Three.js effects:

- instanced cell meshes
- depth/fog
- subtle bloom-like emissive material if available without heavy dependencies
- regime color shifts
- pulse/ripple effects on surprise
- optional orbit controls

Avoid AI-slop visuals, excessive neon, unreadable HUDs, and heavy dependencies.
