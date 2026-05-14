import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { execSync } from 'node:child_process';

export function writeRunArtifacts({ baseDir='artifacts/eidos_life_lab_v0_2', engine, seed=42, command='node' }={}) {
  fs.mkdirSync(baseDir, { recursive: true });
  const configHash = `cfg_${engine.width}x${engine.height}_${seed}`;
  const manifest = { git_commit: safeGit(), python_version: null, platform: os.platform(), seed, grid_size: `${engine.width}x${engine.height}`, config_hash: configHash, command };
  fs.writeFileSync(path.join(baseDir, 'run_manifest.json'), JSON.stringify(manifest, null, 2));
  const state = engine.exportState();
  const summary = { total_generations: engine.generation, seed, config_hash: configHash, start_population: 0, final_population: state.alive_count, births: state.births, deaths: state.deaths, mutations: state.mutations, average_energy: state.global_energy_mean, average_health: state.global_health_mean, average_mass: state.global_mass_mean, average_phi: state.global_phi_mean, regime_counts: state.regime_counts || {}, top_event_types: {}, runtime_stats: {} };
  fs.writeFileSync(path.join(baseDir, 'summary.json'), JSON.stringify(summary, null, 2));
  fs.writeFileSync(path.join(baseDir, 'events.jsonl'), (engine.events || []).map(e => JSON.stringify({ ts: new Date().toISOString(), ...e })).join('\n'));
  return { manifest, summary };
}

function safeGit(){ try { return execSync('git rev-parse HEAD').toString().trim(); } catch { return 'unknown'; } }
