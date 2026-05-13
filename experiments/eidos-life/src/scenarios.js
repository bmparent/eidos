export const SCENARIOS = {
  stable_oscillators: { points:[[10,10],[11,10],[12,10],[20,20],[21,20],[20,21],[21,21]] },
  classic_glider_storm: { points:[[1,0],[2,1],[0,2],[1,2],[2,2],[30,30],[31,31],[29,32],[30,32],[31,32]] },
  collapse_test: { random:0.01, anomaly:0.5 },
  noisy_regime_shift: { random:0.24, anomaly:0.25 },
  rare_structure_emergence: { random:0.16, points:[[40,40],[41,40],[42,40],[42,39],[41,38]] },
  species_competition: { points:[[5,5,1],[6,5,1],[7,5,1],[60,60,2],[61,60,2],[62,60,2],[30,20,3],[30,21,3],[30,22,3]] }
};

export function applyScenario(engine, name){
  const s=SCENARIOS[name] || SCENARIOS.stable_oscillators;
  engine.clear();
  if (s.random) engine.randomize(s.random);
  if (s.points) engine.seed(s.points);
  if (s.anomaly) engine.pulseAnomaly(Math.floor(engine.width/2),Math.floor(engine.height/2),8,s.anomaly);
}
