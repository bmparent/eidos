#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { execSync } from 'node:child_process';
import { LifeEngine } from '../src/life-engine.js';
import { applyScenario, SCENARIOS } from '../src/scenarios.js';

const CHECKPOINT_GENERATIONS = [63];

export function parseArgs(argv) {
  const args = { mode: 'engine_fast', scenario: 'primordial_soup', width: 72, height: 72, steps: 10000 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const v = argv[i + 1];
    if (a === '--scenario') { args.scenario = v; i++; }
    else if (a === '--mode') { args.mode = v; i++; }
    else if (a === '--width') { args.width = Number(v); i++; }
    else if (a === '--height') { args.height = Number(v); i++; }
    else if (a === '--steps') { args.steps = Number(v); i++; }
    else if (a === '--out') { args.out = v; i++; }
    else if (a === '--seeds') { args.seeds = v.split(',').map(Number); i++; }
    else if (a === '--seed-start') { args.seedStart = Number(v); i++; }
    else if (a === '--seed-end') { args.seedEnd = Number(v); i++; }
    else if (a === '--comparison') { args.comparison = v; i++; }
  }
  if (!args.out) throw new Error('--out is required');
  if (!['engine_fast', 'app_faithful'].includes(args.mode)) throw new Error('Invalid --mode');
  if (!SCENARIOS[args.scenario]) throw new Error('Invalid --scenario');
  if (args.seeds && (args.seedStart || args.seedEnd)) throw new Error('Use either --seeds or --seed-start/--seed-end');
  if (!args.seeds) {
    const start = args.seedStart ?? 1; const end = args.seedEnd ?? start;
    args.seeds = Array.from({ length: (end - start) + 1 }, (_, idx) => start + idx);
  }
  args.comparison = args.comparison || 'baseline';
  return args;
}

export function comparisonConfig(mode) {
  const map = {
    baseline: {},
    no_abiogenesis: { abiogenesis_enabled: false },
    no_near_extinction_recovery: { near_extinction_recovery_enabled: false },
    no_primordial_bloom: { primordial_bloom_enabled: false },
    no_rescue: { abiogenesis_enabled: false, near_extinction_recovery_enabled: false, primordial_bloom_enabled: false },
  };
  if (!map[mode]) throw new Error(`Unknown comparison mode: ${mode}`);
  return map[mode];
}

const toJSON = state => JSON.stringify(state, null, 2);
const bool = v => v ? 'true' : 'false';

async function writeStatus(file, payload) { await fs.writeFile(file, `${toJSON(payload)}\n`, 'utf8'); }

function csvRow(obj, cols) {
  return cols.map(c => String(obj[c] ?? '')).join(',');
}

export async function runSeedSweep(opts) {
  const outDir = path.resolve(opts.out);
  await fs.mkdir(outDir, { recursive: true });
  const gitCommit = execSync('git rev-parse HEAD', { encoding: 'utf8' }).trim();
  const envText = [`node=${process.version}`, `platform=${os.platform()}`, `arch=${os.arch()}`].join('\n');
  const rows = [];

  for (const seed of opts.seeds) {
    const runDir = path.join(outDir, `seed_${seed}`);
    const checkpointsDir = path.join(runDir, 'checkpoints');
    await fs.mkdir(checkpointsDir, { recursive: true });
    const statusFile = path.join(runDir, 'run_status.json');
    const baseConfig = { ...SCENARIOS[opts.scenario].preset, ...comparisonConfig(opts.comparison) };
    const configHash = crypto.createHash('sha256').update(JSON.stringify(baseConfig)).digest('hex');
    const engine = new LifeEngine({ width: opts.width, height: opts.height, seed, evolutionEnabled: true, config: baseConfig });
    applyScenario(engine, opts.scenario);

    const meta = { seed, scenario: opts.scenario, mode: opts.mode, width: opts.width, height: opts.height, target_steps: opts.steps, comparison: opts.comparison, config_hash: configHash, git_commit: gitCommit };
    await fs.writeFile(path.join(runDir, 'manifest.json'), `${toJSON(meta)}\n`);
    await fs.writeFile(path.join(runDir, 'config.lock.json'), `${toJSON(engine.config)}\n`);
    await fs.writeFile(path.join(runDir, 'environment.txt'), `${envText}\n`);
    await fs.writeFile(path.join(runDir, 'git_commit.txt'), `${gitCommit}\n`);
    await fs.writeFile(path.join(runDir, 'initial_state.json'), `${toJSON(engine.exportState({ scenario: opts.scenario, settings: meta }))}\n`);
    await writeStatus(statusFile, { ...meta, status: 'running', current_generation: engine.generation, interrupted: false, partial: true });

    let interrupted = false;
    let gen63 = null;
    for (let step = 0; step < opts.steps; step++) {
      engine.step();
      if (CHECKPOINT_GENERATIONS.includes(engine.generation)) {
        const s = engine.exportState({ scenario: opts.scenario, settings: meta });
        await fs.writeFile(path.join(runDir, `generation_${engine.generation}_state.json`), `${toJSON(s)}\n`);
        await fs.writeFile(path.join(checkpointsDir, `generation_${engine.generation}.json`), `${toJSON(s)}\n`);
        if (engine.generation === 63) gen63 = s;
      }
      if (step > 0 && step % 1000 === 0) {
        await writeStatus(statusFile, { ...meta, status: 'running', current_generation: engine.generation, interrupted, partial: true });
      }
      if (process.env.EIDOS_SWEEP_INTERRUPT_AT && engine.generation >= Number(process.env.EIDOS_SWEEP_INTERRUPT_AT)) { interrupted = true; break; }
    }

    const finalState = engine.exportState({ scenario: opts.scenario, settings: meta });
    const summary = {
      ...meta,
      final_generation: finalState.generation,
      interrupted,
      final_alive_count: finalState.alive_count,
      final_density: finalState.density,
      final_viability_state: finalState.viability_state,
      final_regime: finalState.regime,
      generation_63_alive_count: gen63?.alive_count ?? '',
      generation_63_viability_state: gen63?.viability_state ?? '',
      generation_63_diagnosis_state: gen63?.diagnosis?.state ?? '',
      births: finalState.births,
      deaths: finalState.deaths,
      mutations: finalState.mutations,
      reseeds: finalState.reseeds,
      primordial_blooms: finalState.primordial_blooms,
      extinction_events: finalState.extinction_events,
      near_extinction_events: finalState.near_extinction_events,
      collapse_events: finalState.collapse_events,
      recovery_events: finalState.recovery_events,
      population_min: finalState.population_min,
      population_max: finalState.population_max,
      population_mean_recent: finalState.population_mean_recent,
      passed_generation_63: finalState.generation >= 63,
      reached_target_or_interrupted_cleanly: finalState.generation === opts.steps || interrupted,
      config_hash: configHash,
      git_commit: gitCommit,
    };
    await fs.writeFile(path.join(runDir, 'final_state.json'), `${toJSON(finalState)}\n`);
    await fs.writeFile(path.join(runDir, 'final_summary.json'), `${toJSON(summary)}\n`);
    await fs.writeFile(path.join(runDir, 'run_report.md'), `# Seed ${seed} Run Report\n\n- Status: ${interrupted ? 'PARTIAL' : 'COMPLETE'}\n- Final generation: ${summary.final_generation}/${opts.steps}\n- Generation 63 diagnosis: ${summary.generation_63_diagnosis_state || 'N/A'}\n- Final diagnosis: ${finalState.diagnosis.state}\n`);
    await writeStatus(statusFile, { ...summary, status: interrupted ? 'partial' : 'complete', partial: interrupted, current_generation: finalState.generation });
    rows.push(summary);
  }

  const columns = ['seed','scenario','mode','width','height','target_steps','final_generation','interrupted','final_alive_count','final_density','final_viability_state','final_regime','generation_63_alive_count','generation_63_viability_state','generation_63_diagnosis_state','births','deaths','mutations','reseeds','primordial_blooms','extinction_events','near_extinction_events','collapse_events','recovery_events','population_min','population_max','population_mean_recent','passed_generation_63','reached_target_or_interrupted_cleanly','config_hash','git_commit'];
  const csv = [columns.join(','), ...rows.map(r => csvRow(r, columns))].join('\n');
  await fs.writeFile(path.join(outDir, 'seed_sweep_summary.csv'), `${csv}\n`);
  const md = ['# Seed Sweep Summary', '', `Comparison mode: ${opts.comparison}`, '', '| seed | final_generation | interrupted | final_alive_count | gen63_diagnosis | final_viability |', '|---:|---:|---|---:|---|---|', ...rows.map(r => `| ${r.seed} | ${r.final_generation} | ${bool(r.interrupted)} | ${r.final_alive_count} | ${r.generation_63_diagnosis_state} | ${r.final_viability_state} |`)].join('\n');
  await fs.writeFile(path.join(outDir, 'seed_sweep_summary.md'), `${md}\n`);
  return rows;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const opts = parseArgs(process.argv.slice(2));
  await runSeedSweep(opts);
}
