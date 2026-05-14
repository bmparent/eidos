import test from 'node:test';
import assert from 'node:assert/strict';
import { RunState } from '../src/run-state.js';

test('run state detects generation drops and increments epoch/resetCount', () => {
  const run = new RunState();
  run.updateGeneration(10, { scenario: 'a' });
  run.updateGeneration(3, { scenario: 'a' });
  assert.equal(run.runEpoch, 1);
  assert.equal(run.resetCount, 1);
  assert.equal(run.totalGenerations, 10);
  assert.equal(run.resetEvents.at(-1).resetReason, 'detected_generation_drop');
});
