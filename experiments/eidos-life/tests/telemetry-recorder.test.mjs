import test from 'node:test'; import assert from 'node:assert/strict';
import { TelemetryRecorder } from '../src/telemetry-recorder.js';
import { LifeEngine } from '../src/life-engine.js';
import { SCENARIOS, applyScenario } from '../src/scenarios.js';

test('telemetry export bundle shape', ()=>{
  const t=new TelemetryRecorder();
  t.record({generation:1,regime:'GREEN',surprise:0.1,entropy:0.5,compressionRatio:1.2,novelty:0.2,collapseRisk:0,plasticity:0.3});
  const bundle=t.exportBundle();
  assert.ok(bundle.manifest && bundle.summary && Array.isArray(bundle.telemetry) && Array.isArray(bundle.interestingEvents));
});

test('scenario presets seed valid non-empty worlds', ()=>{
  const e=new LifeEngine({width:72,height:72});
  for (const name of Object.keys(SCENARIOS)) { applyScenario(e,name); const alive=e.alive.reduce((a,b)=>a+b,0); assert.ok(alive>0,`${name} empty`); }
});
