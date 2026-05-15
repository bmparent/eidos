import assert from 'node:assert/strict';
import { LifeEngine } from '../src/life-engine.js';
import { applyScenario } from '../src/scenarios.js';
const e = new LifeEngine({ width: 36, height: 36, evolutionEnabled:true }); applyScenario(e,'stable_ecology');
for(let i=0;i<5000;i++) e.step();
const st=e.exportState({scenario:'stable_ecology'});
assert(st.alive_count>0); assert(st.births>0); assert(st.deaths>0); assert(!Number.isNaN(st.global_nutrient_mean));
console.log('ok stable', st.alive_count, st.births, st.deaths);
