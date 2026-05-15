import assert from 'node:assert/strict';
import { LifeEngine } from '../src/life-engine.js';
import { applyScenario } from '../src/scenarios.js';
const e = new LifeEngine({ evolutionEnabled:true }); applyScenario(e,'primordial_soup');
let recovered=false;
for(let i=0;i<400;i++){ e.step(); const st=e.exportState({scenario:'primordial_soup'}); if (st.reseeds>0 && st.alive_count>0) recovered=true; }
const st=e.exportState({scenario:'primordial_soup'});
assert(st.reseeds>0); assert(recovered);
console.log('ok recovery', st.alive_count, st.reseeds);
