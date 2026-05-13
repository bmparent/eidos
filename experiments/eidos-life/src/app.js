import { LifeEngine } from './life-engine.js';
import { EidosMonitor } from './eidos-monitor.js';
import { PatternMemory } from './pattern-memory.js';
import { trackOrganisms } from './organisms.js';
import { TelemetryRecorder } from './telemetry-recorder.js';
import { SCENARIOS, applyScenario } from './scenarios.js';
import { LifeVisualization, REGIME_CLASS } from './visualization.js';
import { EidosBackendBridge } from './eidos-backend-bridge.js';

const engine = new LifeEngine({ width: 72, height: 72 });
const monitor = new EidosMonitor();
const memory = new PatternMemory();
const telemetry = new TelemetryRecorder();
const bridge = new EidosBackendBridge({ enabled: false });
let organisms = [];
let paused = false;

applyScenario(engine, 'stable_oscillators');
const viz = new LifeVisualization({ container: document.body, engine });

const ui = id => document.getElementById(id);
const scenarioSel = ui('scenarioSelect');
const timeline = ui('timeline');
const timelineSlots = Array.from({ length: 24 }, () => {
  const slot = document.createElement('span');
  slot.className = 'timelineCell empty';
  timeline.appendChild(slot);
  return slot;
});

Object.keys(SCENARIOS).forEach(name => {
  const option = document.createElement('option');
  option.value = name;
  option.textContent = name;
  scenarioSel.appendChild(option);
});

function seedSelectedScenario() {
  applyScenario(engine, scenarioSel.value);
  monitor.prevAlive = null;
  monitor.prevEntropy = 0;
  monitor.timeline = [];
  memory.ring = [];
  organisms = [];
  viz.reset();
}

function renderRegimeTimeline(values) {
  const recent = values.slice(-timelineSlots.length);
  for (let i = 0; i < timelineSlots.length; i++) {
    const regime = recent[i] || '';
    timelineSlots[i].className = `timelineCell ${REGIME_CLASS[regime] || 'empty'}`;
    timelineSlots[i].title = regime || 'waiting';
  }
}

ui('pauseBtn').onclick = () => {
  paused = !paused;
  ui('pauseBtn').textContent = paused ? 'Resume' : 'Pause';
};
ui('seedBtn').onclick = seedSelectedScenario;
scenarioSel.onchange = seedSelectedScenario;
ui('pulseBtn').onclick = () => {
  engine.pulseAnomaly(36, 36, 8, 0.8);
  viz.pulse({ x: 36, y: 36, power: 0.9 });
};
ui('exportBtn').onclick = () => {
  const blob = new Blob([JSON.stringify(telemetry.exportBundle(), null, 2)], { type: 'application/json' });
  const anchor = document.createElement('a');
  anchor.href = URL.createObjectURL(blob);
  anchor.download = 'eidos-life-run-bundle.json';
  anchor.click();
};
for (const id of ['toggleSurprise', 'toggleMemory', 'toggleEnergy', 'toggleOutlines']) {
  ui(id).onchange = event => {
    viz.overlays[id.replace('toggle', '').toLowerCase()] = event.target.checked;
  };
}

function tick() {
  if (!paused) {
    const fp = memory.fingerprint(engine.alive, engine.width, engine.height);
    const novelty = memory.novelty(fp);
    memory.remember(fp);
    const { metrics, rulePreset } = monitor.analyze({ ...engine.snapshot(), novelty });
    if (rulePreset.reseed && metrics.collapseRisk) engine.applyReseed();
    engine.step(rulePreset, { surprise: metrics.surprise });
    organisms = trackOrganisms({ ...engine.snapshot(), previous: organisms });
    const row = {
      ...metrics,
      organismCount: organisms.length,
      largestOrganismMass: organisms.reduce((max, organism) => Math.max(max, organism.mass), 0),
    };
    telemetry.record(row, organisms);
    bridge.sendTelemetry(row);

    ui('regimeLabel').textContent = metrics.regime;
    ui('generation').textContent = `gen ${engine.generation}`;
    ui('surprise').textContent = metrics.surprise.toFixed(3);
    ui('entropy').textContent = metrics.entropy.toFixed(3);
    ui('compression').textContent = `${metrics.compressionRatio.toFixed(2)}x`;
    ui('plasticity').textContent = metrics.plasticity.toFixed(3);
    ui('aliveRatio').textContent = metrics.aliveRatio.toFixed(3);
    ui('organisms').textContent = String(organisms.length);
    renderRegimeTimeline(monitor.timeline);
    viz.render({ metrics, organisms });
  }
  requestAnimationFrame(tick);
}

requestAnimationFrame(tick);
