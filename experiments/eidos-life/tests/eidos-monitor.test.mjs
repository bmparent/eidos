import test from 'node:test'; import assert from 'node:assert/strict';
import { EidosMonitor } from '../src/eidos-monitor.js';

test('red regime triggers collapse protection', ()=>{
  const m=new EidosMonitor();
  const alive=new Uint8Array(100); alive[0]=1;
  const age=new Uint16Array(100); const energy=new Float32Array(100).fill(0.6); const stress=new Float32Array(100);
  const out=m.analyze({alive,age,energy,stress,novelty:0.2,generation:40});
  assert.equal(out.metrics.regime,'RED'); assert.equal(out.rulePreset.reseed,true);
});
