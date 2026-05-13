import { LifeEngine } from './life-engine.js';
import { EidosMonitor } from './eidos-monitor.js';
import { PatternMemory } from './pattern-memory.js';
import { OrganismTracker } from './organism-tracker.js';
import { TelemetryRecorder } from './telemetry-recorder.js';
import { EvolutionTelemetry } from './evolution-telemetry.js';
import { PredictionGhost } from './prediction-ghost.js';
import { SCENARIOS, applyScenario } from './scenarios.js';
import { LifeVisualization, REGIME_CLASS } from './visualization.js';
import { EidosBackendBridge } from './eidos-backend-bridge.js';

const engine = new LifeEngine({ width: 72, height: 72, evolutionEnabled: true });
const monitor = new EidosMonitor();
const memory = new PatternMemory();
const tracker = new OrganismTracker();
const telemetry = new TelemetryRecorder();
const evolutionTelemetry = new EvolutionTelemetry();
const predictionGhost = new PredictionGhost(engine.width, engine.height);
const bridge = new EidosBackendBridge({ enabled: false });
const settings = { evolutionEnabled: true, mutationPressure: 'adaptive', intervention: 'guardian', speed: 1, scenario: 'evolutionary_garden' };
let organisms = [];
let prediction = predictionGhost.compare(engine.alive, 0);
let selectedOrganism = null;
let paused = false;
let frameCount = 0;

applyScenario(engine, settings.scenario);
const viz = new LifeVisualization({ container: document.body, engine });

const ui = id => document.getElementById(id);
const scenarioSel = ui('scenarioSelect');
const timeline = ui('timeline');
const inspectPanel = ui('inspectPanel');
const eventFeed = ui('eventFeed');
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
scenarioSel.value = settings.scenario;

function resetScenario(name = scenarioSel.value) {
  settings.scenario = name;
  applyScenario(engine, name);
  monitor.prevAlive = null;
  monitor.prevEntropy = 0;
  monitor.timeline = [];
  memory.ring = [];
  tracker.reset();
  evolutionTelemetry.reset();
  organisms = [];
  selectedOrganism = null;
  prediction = predictionGhost.compare(engine.alive, engine.generation);
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

function chooseOrganism(mode) {
  if (!organisms.length) return null;
  const sorted = organisms.slice();
  if (mode === 'oldest') sorted.sort((a, b) => b.ageFrames - a.ageFrames);
  else if (mode === 'newest') sorted.sort((a, b) => b.birthGeneration - a.birthGeneration);
  else if (mode === 'novel') sorted.sort((a, b) => b.noveltyScore - a.noveltyScore);
  else sorted.sort((a, b) => b.mass - a.mass);
  return sorted[0];
}

function renderInspectPanel() {
  selectedOrganism = chooseOrganism(ui('inspectSelect').value);
  if (!selectedOrganism) {
    inspectPanel.innerHTML = '<div>ID <span>-</span></div><div>status <span>waiting</span></div>';
    return;
  }
  const o = selectedOrganism;
  inspectPanel.innerHTML = [
    ['ID', `#${o.id}`],
    ['age', o.ageFrames],
    ['mass', o.mass],
    ['genome', `G${o.dominantGenomeId}`],
    ['lineage', `L${o.dominantLineageId}`],
    ['fitness', o.fitnessScore.toFixed(2)],
    ['novelty', o.noveltyScore.toFixed(2)],
    ['threat', o.threatScore.toFixed(2)],
    ['children', o.childrenIds.length],
    ['status', o.status],
  ].map(([label, value]) => `<div>${label} <span>${value}</span></div>`).join('');
}

function renderEventFeed() {
  const events = telemetry.events.slice(-5).reverse();
  eventFeed.innerHTML = events.map(item => `<div class="eventCard ${item.severity || 'low'}">${item.description || item.type}</div>`).join('');
}

function exportJson(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const anchor = document.createElement('a');
  anchor.href = URL.createObjectURL(blob);
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(anchor.href);
}

function worldState() {
  return engine.exportState({ scenario: settings.scenario, settings });
}

ui('pauseBtn').onclick = () => {
  paused = !paused;
  ui('pauseBtn').textContent = paused ? 'Resume' : 'Pause';
};
ui('seedBtn').onclick = () => resetScenario(scenarioSel.value);
scenarioSel.onchange = () => resetScenario(scenarioSel.value);
ui('pulseBtn').onclick = () => {
  engine.pulseAnomaly(36, 36, 8, 0.8);
  viz.pulse({ x: 36, y: 36, power: 0.9 });
};
ui('exportBtn').onclick = () => exportJson('eidos-life-run-bundle.json', telemetry.exportBundle(worldState()));
ui('exportWorldBtn').onclick = () => exportJson('eidos-life-world-state.json', worldState());
ui('importWorldBtn').onclick = () => ui('importWorldFile').click();
ui('importWorldFile').onchange = async event => {
  const [file] = event.target.files;
  if (!file) return;
  const state = JSON.parse(await file.text());
  engine.importState(state);
  settings.scenario = state.scenario || settings.scenario;
  Object.assign(settings, state.settings || {});
  scenarioSel.value = settings.scenario;
  tracker.reset();
  evolutionTelemetry.reset();
  viz.reset();
};
ui('toggleEvolution').onchange = event => {
  settings.evolutionEnabled = event.target.checked;
  engine.evolutionEnabled = settings.evolutionEnabled;
};
ui('mutationSelect').onchange = event => { settings.mutationPressure = event.target.value; };
ui('interventionSelect').onchange = event => { settings.intervention = event.target.value; };
ui('speedSelect').onchange = event => { settings.speed = Number(event.target.value); };
ui('inspectSelect').onchange = renderInspectPanel;
for (const id of ['toggleSurprise', 'toggleMemory', 'toggleEnergy', 'toggleOutlines', 'togglePrediction']) {
  ui(id).onchange = event => {
    viz.overlays[id.replace('toggle', '').toLowerCase()] = event.target.checked;
  };
}

function stepWorld() {
  const fp = memory.fingerprint(engine.alive, engine.width, engine.height);
  const novelty = memory.novelty(fp);
  memory.remember(fp);
  const { metrics, rulePreset } = monitor.analyze({ ...engine.snapshot(), novelty });
  engine.applyEidosIntervention(settings.intervention, metrics);
  predictionGhost.predict(engine, rulePreset, engine.localRegimes);
  if (rulePreset.reseed && metrics.collapseRisk) engine.applyReseed();
  engine.step(rulePreset, {
    surprise: metrics.surprise,
    novelty,
    evolutionEnabled: settings.evolutionEnabled,
    mutationPressure: settings.mutationPressure,
    intervention: settings.intervention,
    collapseRisk: metrics.collapseRisk,
  });
  prediction = predictionGhost.compare(engine.alive, engine.generation);
  organisms = tracker.update(engine.snapshot(), metrics, engine.genomeRegistry);
  const evolution = evolutionTelemetry.record({
    engine,
    organisms,
    organismEvents: tracker.getEvents(),
    localRegimes: engine.localRegimes,
    prediction,
    metrics,
  });
  const row = {
    ...metrics,
    ...evolution.metrics,
    generation: engine.generation,
    organismCount: organisms.length,
    largestOrganismMass: organisms.reduce((max, organism) => Math.max(max, organism.mass), 0),
  };
  telemetry.record(row, organisms, {
    events: evolution.events,
    evolution: evolutionTelemetry.exportData({ engine, organismTracker: tracker }),
  });
  bridge.sendTelemetry(row);
  return row;
}

function updateHud(row) {
  ui('regimeLabel').textContent = row.regime;
  ui('generation').textContent = `gen ${engine.generation}`;
  ui('surprise').textContent = row.surprise.toFixed(3);
  ui('entropy').textContent = row.entropy.toFixed(3);
  ui('compression').textContent = `${row.compressionRatio.toFixed(2)}x`;
  ui('plasticity').textContent = row.plasticity.toFixed(3);
  ui('aliveRatio').textContent = row.aliveRatio.toFixed(3);
  ui('organisms').textContent = `${organisms.length} / L${row.livingLineages}`;
  renderRegimeTimeline(monitor.timeline);
  renderInspectPanel();
  renderEventFeed();
}

function tick() {
  if (!paused) {
    const steps = settings.speed === 0 ? (frameCount % 3 === 0 ? 1 : 0) : settings.speed;
    let row = telemetry.rows[telemetry.rows.length - 1] || null;
    for (let i = 0; i < steps; i++) row = stepWorld();
    if (row) {
      updateHud(row);
      viz.render({ metrics: row, organisms, prediction, localRegimes: engine.localRegimes, genomeRegistry: engine.genomeRegistry, selectedOrganism });
    }
  }
  frameCount++;
  requestAnimationFrame(tick);
}

requestAnimationFrame(tick);
