export const SCENARIOS = {
  stable_oscillators: { points: [[10, 10, 1], [11, 10, 1], [12, 10, 1], [20, 20, 2], [21, 20, 2], [20, 21, 2], [21, 21, 2]] },
  classic_glider_storm: { points: [[1, 0, 1], [2, 1, 1], [0, 2, 1], [1, 2, 1], [2, 2, 1], [30, 30, 3], [31, 31, 3], [29, 32, 3], [30, 32, 3], [31, 32, 3]] },
  collapse_test: { random: 0.01, anomaly: 0.5 },
  noisy_regime_shift: { random: 0.24, anomaly: 0.25 },
  rare_structure_emergence: { random: 0.16, points: [[40, 40, 4], [41, 40, 4], [42, 40, 4], [42, 39, 4], [41, 38, 4]] },
  species_competition: { points: [[5, 5, 1], [6, 5, 1], [7, 5, 1], [60, 60, 2], [61, 60, 2], [62, 60, 2], [30, 20, 3], [30, 21, 3], [30, 22, 3]] },
  evolutionary_garden: {
    random: 0.18,
    anomaly: 0.16,
    points: [[12, 52, 1], [13, 52, 1], [14, 52, 1], [46, 18, 2], [47, 18, 2], [48, 18, 2], [52, 46, 5], [53, 47, 5], [51, 48, 5], [52, 48, 5], [53, 48, 5]],
  },
};

export function applyScenario(engine, name) {
  const scenario = SCENARIOS[name] || SCENARIOS.evolutionary_garden;
  engine.clear();
  if (scenario.random) engine.randomize(scenario.random);
  if (scenario.points) {
    for (const [x, y, species = 1] of scenario.points) engine.setAliveCell(engine.idx(x, y), species);
  }
  if (scenario.anomaly) engine.pulseAnomaly(Math.floor(engine.width / 2), Math.floor(engine.height / 2), 8, scenario.anomaly);
  return scenario;
}
