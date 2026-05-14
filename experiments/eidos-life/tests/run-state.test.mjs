import test from 'node:test';
import assert from 'node:assert/strict';
import { RunState } from '../src/run-state.js';

function memStore(seed = {}) {
  const map = new Map(Object.entries(seed));
  return { getItem: (k) => map.get(k) ?? null, setItem: (k, v) => map.set(k, v), removeItem: (k) => map.delete(k) };
}

test('run state detects durable startup generation drops and increments epoch/resetCount', () => {
  const storage = memStore({ 'eidos-life:active-run-meta': JSON.stringify({ runId:'r1', runEpoch:0, resetCount:0, totalGenerations:25000, highestObservedGeneration:25000, lastObservedGeneration:25000, resetEvents:[] }) });
  const run = new RunState({ storage });
  run.initialize({ currentGeneration: 2000, scenario: 'a', settingsHash: 's' });
  assert.equal(run.resetCount, 1);
  assert.equal(run.runEpoch, 1);
  assert.equal(run.resetEvents.at(-1).resetReason, 'app_restart_generation_drop');
  assert.ok(run.totalGenerations >= 25000);
});

test('continuous startup does not false trigger', () => {
  const run = new RunState({ storage: memStore() });
  run.initialize({ currentGeneration: 0 });
  assert.equal(run.resetCount, 0);
});
