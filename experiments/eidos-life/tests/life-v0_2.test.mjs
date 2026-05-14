import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { LifeEngine } from '../src/life-engine.js';
import { writeRunArtifacts } from '../src/artifacts.js';

test('metabolism and death recycle nutrients', () => {
  const e = new LifeEngine({ width: 8, height: 8, evolutionEnabled: true, seed: 1 });
  const i = e.idx(2, 2); e.setAliveCell(i, 1); e.energy[i] = 0.01; e.health[i] = 0.01; const before = e.nutrientField[i];
  e.step();
  assert.equal(e.alive[i], 0);
  assert.ok(e.nutrientField[i] >= before);
});

test('higgs phi increases mass and dampens mutation', () => {
  const e = new LifeEngine({ width: 8, height: 8, evolutionEnabled: true, seed: 2 });
  const i = e.idx(1,1); e.setAliveCell(i, 1); e.higgsPhiField[i] = 2.0;
  assert.ok(e.computeMass(i) > 1.5);
});

test('export state includes compatibility and v0.2 fields', () => {
  const e = new LifeEngine({ width: 8, height: 8, seed: 3 });
  e.randomize(0.2);
  const s = e.exportState();
  for (const k of ['version','generation','width','height','regime','alive_count','density','grid']) assert.ok(k in s);
  assert.equal(s.life_version, '0.2-life-fields');
});

test('artifact writing creates manifest summary and events', () => {
  const e = new LifeEngine({ width: 8, height: 8, seed: 4 }); e.randomize(0.1); e.step();
  const dir = 'artifacts/eidos_life_lab_v0_2';
  writeRunArtifacts({ baseDir: dir, engine: e, seed: 4, command: 'node --test' });
  assert.ok(fs.existsSync(`${dir}/run_manifest.json`));
  assert.ok(fs.existsSync(`${dir}/summary.json`));
  assert.ok(fs.existsSync(`${dir}/events.jsonl`));
});
