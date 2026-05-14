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
  assert.equal(summary.eventSummary.rawEventCounts.organism_death, 500);
  const exported = t.exportSummary({ generation: 500 }, { runId: 'x' }, { finalWorldCompact: { generation: 500, cellCount:100, aliveCount:25, aliveDensity:0.25, activeGenomeCount:2, activeLineageCount:1, genomeRegistrySize:3, lineageRegistrySize:2 } });
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

test('summary includes raw/confirmed events and regime summary', () => {
  const t = new TelemetryRecorder({ maxRowsFull: 20 });
  for (let i=1;i<=6;i++) t.record({ generation:i, regime:i===2?'RED':'GREEN', confirmedRegime:i>=4&&i<=5?'RED':'GREEN', rawRegime:i===2?'RED':'GREEN', redFlicker:i===2?1:0, surprise:0, entropy:0, compressionRatio:1, novelty:0, collapseRisk:0, plasticity:0 }, [], { organismEventSummary: { rawEventCounts:{organism_birth:10}, confirmedEventCounts:{organism_birth:2}, candidateEventCounts:{organism_birth:3}, eventSuppressionCounts:{cooldown_suppressed:4}, eventRatesPer1kGenerations:{raw_organism_birth_per_1k:1000,confirmed_organism_birth_per_1k:200}, recentRawEvents:Array.from({length:70},(_,k)=>({k})), recentConfirmedEvents:Array.from({length:70},(_,k)=>({k})) } });
  const ex = t.exportSummary({ generation:6 }, {}, { finalWorldCompact: { aliveCount: 1, cellCount: 10, aliveDensity: 0.1, activeGenomeCount:1, activeLineageCount:1, genomeRegistrySize:1, lineageRegistrySize:1 } });
  assert.ok(ex.event_summary.rawEventCounts);
  assert.ok(ex.event_summary.confirmedEventCounts);
  assert.ok(ex.event_summary.eventSuppressionCounts);
  assert.equal(ex.event_summary.eventRatesPer1kGenerations.confirmed_organism_birth_per_1k, 200);
  assert.ok(ex.event_summary.recentConfirmedEvents.length <= 70);
  assert.ok(ex.regime_summary.rawRegimeCounts);
  assert.ok(ex.regime_summary.confirmedRegimeCounts);
  assert.ok((ex.regime_summary.redFlickerCount || 0) >= 1);
});
