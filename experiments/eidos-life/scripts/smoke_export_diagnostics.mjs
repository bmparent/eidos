import assert from 'node:assert/strict';
import { LifeEngine } from '../src/life-engine.js';
import { applyScenario } from '../src/scenarios.js';
const e = new LifeEngine({ evolutionEnabled:true }); applyScenario(e,'dead_world');
for(let i=0;i<40;i++) e.step();
const st=e.exportState({scenario:'dead_world'});
for (const k of ['births','deaths','mutations','reseeds','viability_state','diagnosis']) assert(st[k]!==null && st[k]!==undefined);
console.log('ok diag', st.diagnosis.state);
