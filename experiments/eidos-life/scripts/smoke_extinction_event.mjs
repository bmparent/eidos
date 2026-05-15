import assert from 'node:assert/strict';
import { LifeEngine } from '../src/life-engine.js';
import { applyScenario } from '../src/scenarios.js';
const e = new LifeEngine({ evolutionEnabled:true }); applyScenario(e,'extinction_event');
for(let i=0;i<1000;i++){ if(i===300) e.toxicityField.fill(0.95); e.step(); }
const st=e.exportState({scenario:'extinction_event'});
assert(st.collapse_events>=0); assert(st.extinction_events>=0);
console.log('ok extinction', st.extinction_events, st.recovery_events, st.reseeds);
