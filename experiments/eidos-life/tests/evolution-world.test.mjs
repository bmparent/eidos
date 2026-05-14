import test from 'node:test';
import assert from 'node:assert/strict';
import { GenomeRegistry } from '../src/genome.js';
import { LifeEngine, DEFAULT_RULE } from '../src/life-engine.js';
import { OrganismTracker } from '../src/organism-tracker.js';
import { LocalRegimeMap } from '../src/local-regimes.js';
import { EvolutionTelemetry } from '../src/evolution-telemetry.js';
import { PredictionGhost } from '../src/prediction-ghost.js';
import { TelemetryRecorder } from '../src/telemetry-recorder.js';

function withRandom(value, fn) {
  const original = Math.random;
  Math.random = () => value;
  try {
    return fn();
  } finally {
    Math.random = original;
  }
}

test('genome creation and mutation stay bounded', () => {
  const registry = new GenomeRegistry();
  const founderId = registry.createFounderGenome({ birthBias: 0.9, mutationRate: 0.2, colorHue: 0.1 });
  const child = withRandom(0.01, () => registry.inherit([founderId], { mutationPressure: 0.5, generation: 3 }));
  const genome = registry.get(child.genomeId);

  assert.ok(child.mutated);
  assert.ok(genome.parentGenomeIds.includes(founderId));
  for (const value of Object.values(genome.traits)) assert.ok(value >= 0 && value <= 1);
});

test('evolutionary birth inherits parent genome and lineage', () => {
  const engine = new LifeEngine({ width: 5, height: 5, evolutionEnabled: true });
  engine.seed([[1, 2, 1], [2, 2, 1], [3, 2, 1]]);
  const parentGenome = engine.genomeId[engine.idx(2, 2)];

  withRandom(0.01, () => engine.step(DEFAULT_RULE, { evolutionEnabled: true, mutationPressure: 'high' }));
  const born = engine.idx(2, 1);
  const childGenome = engine.genomeRegistry.get(engine.genomeId[born]);

  assert.equal(engine.alive[born], 1);
  assert.ok(engine.genomeId[born] > 0);
  assert.ok(engine.lineageId[born] > 0);
  assert.ok(childGenome.parentGenomeIds.includes(parentGenome));
});

test('ecology fields consume nutrients, produce waste, and expose mutation pressure', () => {
  const engine = new LifeEngine({ width: 5, height: 5, evolutionEnabled: true });
  engine.seed([[2, 2, 1]]);
  const i = engine.idx(2, 2);
  const beforeNutrient = engine.nutrientField[i];
  engine.anomalyField[i] = 0.8;
  engine.ecology.update(engine, engine.genomeRegistry);
  const context = engine.ecology.contextAt(engine, i, engine.genomeRegistry.get(engine.genomeId[i]));

  assert.ok(engine.nutrientField[i] < beforeNutrient);
  assert.ok(engine.wasteField[i] > 0);
  assert.ok(engine.stress[i] > 0);
  assert.ok(context.mutationPressure > 0.2);
});

test('local regime map detects collapse and chaos zones', () => {
  const width = 12, height = 12;
  const local = new LocalRegimeMap(width, height, { tilesX: 3, tilesY: 3 });
  const empty = new Uint8Array(width * height);
  const field = new Float32Array(width * height);
  local.update({ width, height, alive: empty, anomalyField: field, memoryField: field, stress: field }, { generation: 40 });
  assert.ok(local.export().regimes.every(regime => regime === 'RED'));

  const alive = new Uint8Array(width * height);
  for (let i = 0; i < alive.length; i++) alive[i] = i % 2;
  const anomaly = new Float32Array(width * height).fill(0.5);
  const memory = new Float32Array(width * height).fill(0.4);
  const stress = new Float32Array(width * height);
  local.previousAlive.fill(0);
  local.update({ width, height, alive, anomalyField: anomaly, memoryField: memory, stress }, { generation: 40, novelty: 0.4 });
  assert.ok(local.export().regimes.includes('VIOLET') || local.export().regimes.includes('AMBER'));
});

test('organism tracker persists identity and detects split/death', () => {
  const engine = new LifeEngine({ width: 8, height: 8, evolutionEnabled: true });
  const tracker = new OrganismTracker();
  engine.seed([[2, 2, 1], [3, 2, 1], [4, 2, 1]]);
  let organisms = tracker.update(engine.snapshot(), { regime: 'GREEN' }, engine.genomeRegistry);
  const firstId = organisms[0].id;

  organisms = tracker.update(engine.snapshot(), { regime: 'GREEN' }, engine.genomeRegistry);
  assert.equal(organisms[0].id, firstId);

  engine.clear();
  engine.setAliveCell(engine.idx(2, 2), 1);
  engine.setAliveCell(engine.idx(4, 2), 1);
  for (let i=0;i<3;i++) tracker.update(engine.snapshot(), { regime: 'GREEN' }, engine.genomeRegistry);
  const splitSummary = tracker.getEventSummary(100);
  assert.ok((splitSummary.candidateEventTypeCounts.organism_split || 0) > 0);

  engine.clear();
  for (let i=0;i<3;i++) tracker.update(engine.snapshot(), { regime: 'GREEN' }, engine.genomeRegistry);
  const deathSummary = tracker.getEventSummary(100);
  assert.ok((deathSummary.candidateEventTypeCounts.organism_death || 0) > 0);
});

test('prediction ghost reports prediction error and spark positions', () => {
  const engine = new LifeEngine({ width: 5, height: 5 });
  const ghost = new PredictionGhost(5, 5);
  engine.seed([[1, 2], [2, 2], [3, 2]]);
  ghost.predict(engine, DEFAULT_RULE, engine.localRegimes);
  engine.step(DEFAULT_RULE);
  const result = ghost.compare(engine.alive, engine.generation);

  assert.ok(result.predicted.length === 25);
  assert.ok(result.predictionError >= 0);
  assert.ok(Array.isArray(result.sparks));
});

test('evolution telemetry export includes genomes lineages and organisms', () => {
  const engine = new LifeEngine({ width: 8, height: 8, evolutionEnabled: true });
  const tracker = new OrganismTracker();
  const evo = new EvolutionTelemetry();
  const telemetry = new TelemetryRecorder();
  engine.seed([[2, 2, 1], [3, 2, 1], [4, 2, 1]]);
  const organisms = tracker.update(engine.snapshot(), { regime: 'GREEN', novelty: 0.2 }, engine.genomeRegistry);
  const prediction = { predictionError: 0, sparks: [] };
  const evolution = evo.record({ engine, organisms, organismEvents: tracker.getEvents(), localRegimes: engine.localRegimes, prediction, metrics: { regime: 'GREEN' } });
  telemetry.record({ generation: 1, regime: 'GREEN', surprise: 0, entropy: 0, compressionRatio: 1, novelty: 0, ...evolution.metrics }, organisms, {
    events: evolution.events,
    evolution: evo.exportData({ engine, organismTracker: tracker }),
  });
  const bundle = telemetry.exportBundle(engine.exportState());

  assert.ok(bundle.evolution.genomes.length > 0);
  assert.ok(bundle.evolution.lineages.length > 0);
  assert.ok(bundle.evolution.organisms.length > 0);
  assert.ok(bundle.worldState.genomeRegistry.genomes.length > 0);
});
