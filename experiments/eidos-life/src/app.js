import { LifeEngine } from './life-engine.js';
import { EidosMonitor } from './eidos-monitor.js';
import { PatternMemory } from './pattern-memory.js';
import { trackOrganisms } from './organisms.js';
import { TelemetryRecorder } from './telemetry-recorder.js';
import { SCENARIOS, applyScenario } from './scenarios.js';
import { LifeVisualization } from './visualization.js';
import { EidosBackendBridge } from './eidos-backend-bridge.js';

const engine = new LifeEngine({ width: 72, height: 72 });
const monitor = new EidosMonitor();
const memory = new PatternMemory();
const telemetry = new TelemetryRecorder();
const bridge = new EidosBackendBridge({ enabled: false });
let organisms=[]; let paused=false;
applyScenario(engine, 'stable_oscillators');
const viz = new LifeVisualization({ container: document.body, engine });

const ui = id => document.getElementById(id);
const scenarioSel = ui('scenarioSelect');
Object.keys(SCENARIOS).forEach(name => { const o=document.createElement('option'); o.value=name; o.textContent=name; scenarioSel.appendChild(o);});
ui('pauseBtn').onclick=()=>{paused=!paused; ui('pauseBtn').textContent=paused?'Resume':'Pause';};
ui('seedBtn').onclick=()=>applyScenario(engine, scenarioSel.value);
ui('pulseBtn').onclick=()=>engine.pulseAnomaly(36,36,8,0.8);
ui('exportBtn').onclick=()=>{ const blob=new Blob([JSON.stringify(telemetry.exportBundle(),null,2)],{type:'application/json'}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='eidos-life-run-bundle.json'; a.click(); };
for (const id of ['toggleSurprise','toggleMemory','toggleEnergy','toggleOutlines']) ui(id).onchange=(e)=>viz.overlays[id.replace('toggle','').toLowerCase()]=e.target.checked;

function tick(){
  if(!paused){
    const fp = memory.fingerprint(engine.alive,engine.width,engine.height);
    const novelty = memory.novelty(fp); memory.remember(fp);
    const { metrics, rulePreset } = monitor.analyze({ ...engine.snapshot(), novelty });
    if (rulePreset.reseed && metrics.collapseRisk) engine.applyReseed();
    engine.step(rulePreset, { surprise: metrics.surprise });
    organisms = trackOrganisms({ ...engine.snapshot(), previous: organisms });
    const row = { ...metrics, organismCount: organisms.length, largestOrganismMass: organisms.reduce((m,o)=>Math.max(m,o.mass),0) };
    telemetry.record(row, organisms); bridge.sendTelemetry(row);
    ui('regimeLabel').textContent=metrics.regime; ui('generation').textContent=`gen ${engine.generation}`;
    ui('surprise').textContent=metrics.surprise.toFixed(3); ui('entropy').textContent=metrics.entropy.toFixed(3); ui('compression').textContent=`${metrics.compressionRatio.toFixed(2)}x`;
    ui('plasticity').textContent=metrics.plasticity.toFixed(3); ui('aliveRatio').textContent=metrics.aliveRatio.toFixed(3); ui('organisms').textContent=String(organisms.length);
    const timeline=ui('timeline'); timeline.textContent=monitor.timeline.slice(-24).join(' ');
    viz.render({ metrics, organisms });
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
