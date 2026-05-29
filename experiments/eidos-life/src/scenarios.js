export const SCENARIOS = {
  stable_ecology: { random: 0.16, anomaly: 0.08, preset: { metabolism_scale: 0.76, nutrient_absorption_rate: 0.15, toxicity_damage_scale: 0.3, abiogenesis_enabled: true, near_extinction_recovery_enabled: true } },
  primordial_soup: { random: 0, anomaly: 0.02, initialSeeds: [[36, 36, 1]], preset: { abiogenesis_enabled: true, primordial_bloom_enabled: true, abiogenesis_seed_count: 32, primordial_bloom_seed_count: 56 }, field: { nutrient: 0.85, toxicity: 0.04 } },
  higgs_wells: { random: 0.12, anomaly: 0.1, preset: { mass_phi_gain: 1.5, metabolism_scale: 0.72 } },
  harsh_world: { random: 0.08, anomaly: 0.26, preset: { nutrient_absorption_rate: 0.1, toxicity_damage_scale: 0.5, near_extinction_recovery_enabled: true, abiogenesis_cooldown: 220 } },
  mutation_storm: { random: 0.2, anomaly: 0.18, preset: { mutation_mass_cost: 0.01, reproduction_energy_threshold: 0.64, metabolism_scale: 0.84 } },
  extinction_event: { random: 0.16, anomaly: 0.12, preset: { near_extinction_recovery_enabled: true, abiogenesis_enabled: true }, shockGeneration: 300 },
  dead_world: { random: 0, anomaly: 0, initialSeeds: [[36, 36, 1]], preset: { abiogenesis_enabled: false, near_extinction_recovery_enabled: false, viability_enabled: false } },
};

export function applyScenario(engine, name) {
  const scenario = SCENARIOS[name] || SCENARIOS.stable_ecology;
  engine.clear();
  Object.assign(engine.config, scenario.preset || {});
  engine.presetName = name;
  if (scenario.random) engine.randomize(scenario.random);
  if (scenario.field) {
    engine.nutrientField.fill(scenario.field.nutrient);
    engine.toxicityField.fill(scenario.field.toxicity);
  }
  if (scenario.initialSeeds) {
    for (const [x, y, species = 1] of scenario.initialSeeds) engine.setAliveCell(engine.idx(x, y), species);
    engine.nextAlive.set(engine.alive);
    engine.updatePopulationStats?.();
  }
  if (engine.alive.reduce((a, b) => a + b, 0) === 0 && !scenario.initialSeeds) {
    engine.seedBySuitability?.(scenario.preset?.abiogenesis_seed_count || 24);
  }
  if (scenario.anomaly) engine.pulseAnomaly?.(Math.floor(engine.width / 2), Math.floor(engine.height / 2), 8, scenario.anomaly);
  return scenario;
}
