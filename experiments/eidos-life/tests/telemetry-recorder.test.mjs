import test from 'node:test'; import assert from 'node:assert/strict';
import { TelemetryRecorder } from '../src/telemetry-recorder.js';
import { LifeEngine } from '../src/life-engine.js';
import { SCENARIOS, applyScenario } from '../src/scenarios.js';

test('telemetry export bundle shape', ()=>{
  const t=new TelemetryRecorder();
  t.record({generation:1,regime:'GREEN',surprise:0.1,entropy:0.5,compressionRatio:1.2,novelty:0.2,collapseRisk:0,plasticity:0});
  const bundle=t.exportBundle();
  assert.ok(bundle.manifest && bundle.summary && Array.isArray(bundle.telemetry) && Array.isArray(bundle.interestingEvents));
});

test('telemetry summary is compact and bounded', () => {
  const t = new TelemetryRecorder({ maxRowsFull: 50, sampleEveryAfterMax: 5, maxSampledRows: 20 });
  for (let i=1;i<=500;i++) {
    t.record({generation:i,regime:i%2?'GREEN':'RED',surprise:0.1*i,entropy:0.5,compressionRatio:1.2,novelty:0.2,collapseRisk:0,plasticity:0.3,organismCount:2}, [], { events:[{ type:'organism_death', severity:'low', lineageId:1, genomeId:7 }] });
  }
  assert.ok(t.rows.length <= 50);
  assert.ok(t.sampledRows.length <= 20);
  const summary = t.getSummary();
  assert.equal(summary.regimeCounts.GREEN + summary.regimeCounts.RED, 500);
  assert.ok(summary.regimeTransitions.count > 0);
  assert.equal(summary.eventSummary.eventTypeCounts.organism_death, 500);
  const exported = t.exportSummary({ generation: 500 }, { runId: 'x' }, { finalWorldCompact: { generation: 500 } });
  assert.ok(exported.telemetry_sample.length <= 500);
  assert.equal(exported.run_meta.runId, 'x');
  assert.equal(exported.final_world_compact.generation, 500);
  assert.equal(exported.final_world_compact.alive, undefined);
  assert.ok(!('worldState' in exported));
});

test('scenario presets seed valid non-empty worlds', ()=>{
  const e=new LifeEngine({width:72,height:72});
  for (const name of Object.keys(SCENARIOS)) { applyScenario(e,name); const alive=e.alive.reduce((a,b)=>a+b,0); assert.ok(alive>0,`${name} empty`); }
});
