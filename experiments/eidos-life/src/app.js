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
import { RunState } from './run-state.js';

const engine = new LifeEngine({ width: 72, height: 72, evolutionEnabled: true });
const monitor = new EidosMonitor();
const memory = new PatternMemory();
const tracker = new OrganismTracker();
const telemetry = new TelemetryRecorder();
const runState = new RunState();
const evolutionTelemetry = new EvolutionTelemetry();
const predictionGhost = new PredictionGhost(engine.width, engine.height);
const bridge = new EidosBackendBridge({ enabled: false });
const settings = { evolutionEnabled: true, mutationPressure: 'adaptive', intervention: 'guardian', speed: 1, scenario: 'evolutionary_garden' };
const AUTOSAVE_METADATA_INTERVAL = 5000;
const FULL_CHECKPOINT_MAX_LOCALSTORAGE_BYTES = 1_000_000;
const CHECKPOINT_MODE_DEFAULT = 'metadata_only';
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

function resetScenario(name = scenarioSel.value, reason = 'scenario_change') {
  settings.scenario = name;
  runState.recordReset(reason, runState.lastObservedGeneration, 0, { scenario: name, settings: { ...settings } });
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
  runState.updateGeneration(engine.generation, { scenario: settings.scenario, settings: { ...settings } });
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
function setStatus(message) { ui('statusLine').textContent = message; }

function worldState() {
  return engine.exportState({ scenario: settings.scenario, settings });
}

ui('pauseBtn').onclick = () => {
  paused = !paused;
  ui('pauseBtn').textContent = paused ? 'Resume' : 'Pause';
};
ui('seedBtn').onclick = () => resetScenario(scenarioSel.value, 'seed');
scenarioSel.onchange = () => resetScenario(scenarioSel.value, 'scenario_change');
ui('pulseBtn').onclick = () => {
  engine.pulseAnomaly(36, 36, 8, 0.8);
  viz.pulse({ x: 36, y: 36, power: 0.9 });
};
ui('exportBtn').onclick = () => exportJson('eidos-life-run-bundle.json', telemetry.exportBundle(worldState()));
ui('summaryExportBtn').onclick = async () => {
  setStatus('summary export: building...');
  await new Promise(resolve => setTimeout(resolve, 0));
  try {
    const summary = telemetry.exportSummary(worldState(), runState.exportMeta(), { settings: { ...settings }, finalWorldCompact: buildFinalWorldCompact() });
    exportJson('eidos-life-summary.json', summary);
    setStatus('summary export: complete');
  } catch (error) {
    console.warn('summary export failed', error);
    setStatus('summary export: failed, see console');
  }
};
ui('exportWorldBtn').onclick = () => exportJson('eidos-life-world-state.json', worldState());
ui('importWorldBtn').onclick = () => ui('importWorldFile').click();
ui('importWorldFile').onchange = async event => {
  const [file] = event.target.files;
  if (!file) return;
  const state = JSON.parse(await file.text());
  runState.recordReset('import_world', runState.lastObservedGeneration, state.generation || 0, { scenario: state.scenario || settings.scenario, settings: { ...settings } });
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
ui('saveCheckpointBtn').onclick = saveCheckpoint;

for (const id of ['toggleSurprise', 'toggleMemory', 'toggleEnergy', 'toggleOutlines', 'togglePrediction']) {
  ui(id).onchange = event => {
    viz.overlays[id.replace('toggle', '').toLowerCase()] = event.target.checked;
  };
}

function buildFinalWorldCompact() {
  const snap = engine.snapshot();
  const aliveCount = snap.aliveCount;
  const total = engine.width * engine.height;
  const mean = (arr) => arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0;
  const liveAges=[]; const liveEnergy=[]; const liveStress=[]; const liveMemory=[];
  const q=[0,0,0,0];
  for (let y=0;y<engine.height;y++) for (let x=0;x<engine.width;x++) {
    const i=y*engine.width+x; if (!engine.alive[i]) continue;
    liveAges.push(engine.age[i]); liveEnergy.push(engine.energy[i]); liveStress.push(engine.stress[i]); liveMemory.push(engine.memory[i]);
    q[(y>=engine.height/2)*2 + (x>=engine.width/2)]++;
  }
  return { generation: engine.generation, totalGenerations: runState.totalGenerations, runEpoch: runState.runEpoch, resetCount: runState.resetCount, width: engine.width, height: engine.height, aliveCount, aliveDensity: aliveCount/total, activeGenomeCount: snap.activeGenomeCount, activeLineageCount: snap.activeLineageCount, genomeRegistrySize: engine.genomeRegistry.genomes.length, lineageRegistrySize: engine.genomeRegistry.lineages.length, nextGenomeId: engine.genomeRegistry.nextGenomeId, nextLineageId: engine.genomeRegistry.nextLineageId, oldestLiveCellAge: liveAges.length?Math.max(...liveAges):0, meanLiveAge: mean(liveAges), meanLiveEnergy: mean(liveEnergy), meanLiveStress: mean(liveStress), meanLiveMemory: mean(liveMemory), quadrantAliveCounts: q };
}

function saveMetadataCheckpoint(totalGeneration) {
  const metadata = { checkpointMode: CHECKPOINT_MODE_DEFAULT, warning: 'full checkpoint not autosaved; use Save Checkpoint for manual full snapshot file export', runId: runState.runId, runEpoch: runState.runEpoch, totalGenerations: runState.totalGenerations, generation: engine.generation, resetCount: runState.resetCount, scenario: settings.scenario, settings: { ...settings }, timestamp: new Date().toISOString(), telemetryStats: telemetry.getSummary(), finalWorldCompact: buildFinalWorldCompact() };
  try {
    localStorage.setItem('eidos-life:checkpoint-meta', JSON.stringify(metadata));
    setStatus(`checkpoint: metadata saved at total ${totalGeneration}`);
  } catch (error) {
    console.warn('metadata checkpoint failed', error);
    setStatus('checkpoint: metadata save failed');
  }
}

function saveCheckpoint() {
  const checkpoint = { worldState: worldState(), runMeta: runState.exportMeta(), compact: buildFinalWorldCompact(), savedAt: new Date().toISOString(), mode: 'manual_full_checkpoint' };
  const serialized = JSON.stringify(checkpoint);
  if (serialized.length > FULL_CHECKPOINT_MAX_LOCALSTORAGE_BYTES) {
    exportJson(`eidos-life-checkpoint-${runState.totalGenerations}.json`, checkpoint);
    setStatus('checkpoint: skipped localStorage full save, downloaded file');
    return;
  }
  try {
    localStorage.setItem('eidos-life:last-checkpoint', serialized);
    setStatus(`checkpoint: full saved locally (${serialized.length} bytes)`);
  } catch (error) {
    console.warn('full checkpoint localStorage save failed', error);
    exportJson(`eidos-life-checkpoint-${runState.totalGenerations}.json`, checkpoint);
    setStatus('checkpoint: local save failed, downloaded file');
  }
  saveMetadataCheckpoint(runState.totalGenerations);
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
  runState.updateGeneration(engine.generation, { scenario: settings.scenario, settings: { ...settings } });
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
  ui('regimeLabel').textContent = row.regime === 'RED' && row.aliveRatio > 0 ? 'RED / sparse-risk' : row.regime;
  ui('generation').textContent = `gen ${engine.generation}`;
  ui('runMeta').textContent = `total ${runState.totalGenerations} / epoch ${runState.runEpoch} / resets ${runState.resetCount}`;
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
      if (engine.generation > 0 && engine.generation % AUTOSAVE_METADATA_INTERVAL === 0) saveMetadataCheckpoint(runState.totalGenerations);
      updateHud(row);
      viz.render({ metrics: row, organisms, prediction, localRegimes: engine.localRegimes, genomeRegistry: engine.genomeRegistry, selectedOrganism });
    }
  }
  frameCount++;
  requestAnimationFrame(tick);
}

requestAnimationFrame(tick);
