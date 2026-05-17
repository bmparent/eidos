import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { parseArgs, runSeedSweep } from '../scripts/seed_sweep_viability.mjs';

test('argument parsing', () => {
  const args = parseArgs(['--scenario','primordial_soup','--mode','engine_fast','--steps','500','--seeds','1,2','--out','artifacts/tmp']);
  assert.deepEqual(args.seeds, [1,2]);
  assert.equal(args.steps, 500);
});

test('summary csv shape and generation_63 checkpoint', async () => {
  const out = await fs.mkdtemp(path.join(os.tmpdir(), 'eidos-sweep-'));
  const rows = await runSeedSweep(parseArgs(['--scenario','primordial_soup','--mode','engine_fast','--steps','100','--seeds','3','--out', out]));
  assert.equal(rows.length, 1);
  const csv = await fs.readFile(path.join(out, 'seed_sweep_summary.csv'), 'utf8');
  assert.ok(csv.split('\n')[0].includes('generation_63_diagnosis_state'));
  const gen63 = await fs.readFile(path.join(out, 'seed_3', 'generation_63_state.json'), 'utf8');
  assert.ok(gen63.includes('"generation": 63'));
});

test('run_status updates and interrupted run auditable', async () => {
  const out = await fs.mkdtemp(path.join(os.tmpdir(), 'eidos-sweep-int-'));
  process.env.EIDOS_SWEEP_INTERRUPT_AT = '70';
  await runSeedSweep(parseArgs(['--scenario','primordial_soup','--mode','engine_fast','--steps','500','--seeds','4','--out', out]));
  delete process.env.EIDOS_SWEEP_INTERRUPT_AT;
  const status = JSON.parse(await fs.readFile(path.join(out, 'seed_4', 'run_status.json'), 'utf8'));
  assert.equal(status.status, 'partial');
  assert.equal(status.interrupted, true);
  const report = await fs.readFile(path.join(out, 'seed_4', 'run_report.md'), 'utf8');
  assert.ok(report.includes('PARTIAL'));
});
