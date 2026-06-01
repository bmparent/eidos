import './styles.css';
import { LabApi } from './api.js';
import { TrendChart } from './charts.js';
import { LabRenderer } from './renderer.js';
import { formatNumber, regimeClass, renderEvents, updateMetricsGrid } from './ui.js';

const api = new LabApi();
const replayBuffer = [];
const maxReplayFrames = 500;
let latestSnapshot = null;
let liveMode = true;
let wsStatus = 'connecting';

const elements = {
  mount: document.querySelector('#worldMount'),
  generation: document.querySelector('#hudGeneration'),
  alive: document.querySelector('#hudAlive'),
  density: document.querySelector('#hudDensity'),
  sentinel: document.querySelector('#sentinelBadge'),
  connection: document.querySelector('#connectionBadge'),
  metricsGrid: document.querySelector('#metricsGrid'),
  eventsFeed: document.querySelector('#eventsFeed'),
  cellInspector: document.querySelector('#cellInspector'),
  trendCanvas: document.querySelector('#trendCanvas'),
  replaySlider: document.querySelector('#replaySlider'),
  replayStatus: document.querySelector('#replayStatus'),
  artifactStatus: document.querySelector('#artifactStatus'),
  debugLog: document.querySelector('#debugLog'),
  toolSelect: document.querySelector('#toolSelect'),
  radiusInput: document.querySelector('#radiusInput'),
  fieldViewSelect: document.querySelector('#fieldViewSelect'),
  qualitySelect: document.querySelector('#qualitySelect'),
  speedInput: document.querySelector('#speedInput'),
  mutationSelect: document.querySelector('#mutationSelect'),
  interventionSelect: document.querySelector('#interventionSelect'),
  scenarioSelect: document.querySelector('#scenarioSelect')
};

const chart = new TrendChart(elements.trendCanvas);
const renderer = new LabRenderer(elements.mount, handleCellClick);

function pushReplay(snapshot) {
  replayBuffer.push(snapshot);
  if (replayBuffer.length > maxReplayFrames) {
    replayBuffer.shift();
  }
  elements.replaySlider.max = String(Math.max(0, replayBuffer.length - 1));
  if (liveMode) {
    elements.replaySlider.value = String(Math.max(0, replayBuffer.length - 1));
  }
  elements.replayStatus.textContent = `${liveMode ? 'Live' : 'Replay'} buffer: ${replayBuffer.length} frames`;
}

function displaySnapshot(snapshot, options = {}) {
  latestSnapshot = snapshot;
  if (!options.fromReplay) {
    pushReplay(snapshot);
    chart.add(snapshot.metrics);
  }
  renderer.updateSnapshot(snapshot);
  const metrics = snapshot.metrics;
  elements.generation.textContent = metrics.generation;
  elements.alive.textContent = metrics.alive;
  elements.density.textContent = formatNumber(metrics.density, 4);
  elements.sentinel.textContent = metrics.sentinelRegime;
  elements.sentinel.className = regimeClass(metrics.sentinelRegime);
  updateMetricsGrid(elements.metricsGrid, metrics);
  renderEvents(elements.eventsFeed, snapshot.events || []);
  writeDebugLine();
}

function writeDebugLine(extra = '') {
  elements.debugLog.textContent = [
    `backend HTTP: ${api.httpUrl}`,
    `backend WS:   ${api.wsUrl}`,
    `ws status:    ${wsStatus}`,
    `mode:         ${liveMode ? 'live' : 'replay'}`,
    extra
  ]
    .filter(Boolean)
    .join('\n');
}

function updateConnection(status) {
  wsStatus = status;
  elements.connection.textContent = status;
  elements.connection.className = `connection-badge connection-${status}`;
  writeDebugLine();
}

function onSocketMessage(payload) {
  if (payload.type === 'snapshot' && payload.snapshot) {
    if (liveMode) {
      displaySnapshot(payload.snapshot);
    } else {
      latestSnapshot = payload.snapshot;
      pushReplay(payload.snapshot);
    }
  } else if (payload.type === 'error') {
    console.warn('Backend command error', payload.detail);
    writeDebugLine(`last error: ${payload.detail}`);
  } else if (payload.type === 'ack') {
    if (payload.result?.path) {
      elements.artifactStatus.textContent = payload.result.path;
    }
  }
}

async function pollFallback() {
  if (api.connected) {
    return;
  }
  try {
    const snapshot = await api.fetchState();
    if (liveMode) {
      displaySnapshot(snapshot);
    }
  } catch (error) {
    console.warn('HTTP polling fallback failed', error);
    writeDebugLine(`poll error: ${error.message}`);
  }
}

async function command(payload) {
  try {
    const result = await api.sendCommand(payload);
    if (result.snapshot && liveMode) {
      displaySnapshot(result.snapshot);
    }
    return result;
  } catch (error) {
    console.warn('Command failed', payload, error);
    writeDebugLine(`command error: ${error.message}`);
    return null;
  }
}

function handleCellClick(cell) {
  if (!latestSnapshot) {
    return;
  }
  const tool = elements.toolSelect.value;
  updateInspector(cell);
  if (tool === 'inspect') {
    return;
  }
  if (!liveMode) {
    liveMode = true;
  }
  if (tool === 'toggle') {
    command({ command: 'toggle_cell', x: cell.x, y: cell.y });
  } else if (tool === 'birth') {
    command({ command: 'set_cell', x: cell.x, y: cell.y, alive: true });
  } else if (tool === 'kill') {
    command({ command: 'set_cell', x: cell.x, y: cell.y, alive: false });
  } else if (tool === 'paint_birth') {
    command({ command: 'paint_disk', x: cell.x, y: cell.y, radius: Number(elements.radiusInput.value), mode: 'birth' });
  } else if (tool === 'paint_kill') {
    command({ command: 'paint_disk', x: cell.x, y: cell.y, radius: Number(elements.radiusInput.value), mode: 'kill' });
  } else {
    command({ command: 'inject_pattern', x: cell.x, y: cell.y, pattern: tool });
  }
}

function updateInspector(cell) {
  const snapshot = latestSnapshot;
  const index = cell.index;
  const rows = [
    ['x, y', `${cell.x}, ${cell.y}`],
    ['alive', snapshot.alive[index] ? 'yes' : 'no'],
    ['genome', snapshot.genome[index]],
    ['lineage', snapshot.lineage[index]],
    ['energy', formatNumber(snapshot.energy[index], 3)],
    ['memory', formatNumber(snapshot.memory[index], 3)],
    ['signal', formatNumber(snapshot.signal[index], 3)],
    ['nutrient', formatNumber(snapshot.nutrient[index], 3)],
    ['waste', formatNumber(snapshot.waste[index], 3)],
    ['stress', formatNumber(snapshot.stress[index], 3)]
  ];
  elements.cellInspector.innerHTML = rows.map(([key, value]) => `<div><span>${key}</span><strong>${value}</strong></div>`).join('');
}

function bindControls() {
  document.querySelector('#playBtn').addEventListener('click', () => command({ command: 'play' }));
  document.querySelector('#pauseBtn').addEventListener('click', () => command({ command: 'pause' }));
  document.querySelector('#stepBtn').addEventListener('click', () => command({ command: 'step', steps: 1 }));
  document.querySelector('#resetBtn').addEventListener('click', () => {
    replayBuffer.length = 0;
    liveMode = true;
    command({ command: 'reset', scenario: elements.scenarioSelect.value });
  });
  document.querySelector('#clearBtn').addEventListener('click', () => command({ command: 'clear_world' }));
  document.querySelector('#randomBtn').addEventListener('click', () => command({ command: 'random_seed', density: 0.09 }));
  elements.speedInput.addEventListener('change', sendSettings);
  elements.mutationSelect.addEventListener('change', sendSettings);
  elements.interventionSelect.addEventListener('change', sendSettings);
  elements.qualitySelect.addEventListener('change', () => {
    renderer.setQuality(elements.qualitySelect.value);
    sendSettings();
  });
  elements.fieldViewSelect.addEventListener('change', () => renderer.setFieldView(elements.fieldViewSelect.value));
  document.querySelector('#topDownBtn').addEventListener('click', () => renderer.topDown());
  document.querySelector('#tiltBtn').addEventListener('click', () => renderer.tiltView());
  document.querySelector('#fitBoardBtn').addEventListener('click', () => renderer.fitBoard());
  elements.replaySlider.addEventListener('input', () => {
    const index = Number(elements.replaySlider.value);
    const snapshot = replayBuffer[index];
    if (!snapshot) {
      return;
    }
    liveMode = false;
    displaySnapshot(snapshot, { fromReplay: true });
    elements.replayStatus.textContent = `Replay frame ${index + 1} of ${replayBuffer.length}`;
  });
  document.querySelector('#backToLiveBtn').addEventListener('click', () => {
    liveMode = true;
    if (latestSnapshot) {
      displaySnapshot(latestSnapshot, { fromReplay: true });
    }
    elements.replayStatus.textContent = `Live buffer: ${replayBuffer.length} frames`;
  });
  document.querySelector('#exportBtn').addEventListener('click', async () => {
    elements.artifactStatus.textContent = 'Exporting...';
    try {
      const result = await api.exportState();
      elements.artifactStatus.textContent = result.result.path;
    } catch (error) {
      elements.artifactStatus.textContent = error.message;
    }
  });
  document.querySelector('#checkpointBtn').addEventListener('click', async () => {
    elements.artifactStatus.textContent = 'Checkpointing...';
    try {
      const result = await api.checkpoint();
      elements.artifactStatus.textContent = result.result.path;
    } catch (error) {
      elements.artifactStatus.textContent = error.message;
    }
  });
}

function sendSettings() {
  command({
    command: 'set',
    settings: {
      speed: Number(elements.speedInput.value),
      mutationPressure: elements.mutationSelect.value,
      interventionMode: elements.interventionSelect.value,
      renderQuality: elements.qualitySelect.value
    }
  });
}

async function boot() {
  bindControls();
  renderer.setFieldView(elements.fieldViewSelect.value);
  api.connect({ onMessage: onSocketMessage, onStatus: updateConnection });
  setInterval(pollFallback, 850);
  try {
    const snapshot = await api.fetchState();
    displaySnapshot(snapshot);
  } catch (error) {
    console.warn('Initial state fetch failed', error);
    writeDebugLine(`initial fetch error: ${error.message}`);
  }
}

boot();
