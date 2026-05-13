import test from 'node:test'; import assert from 'node:assert/strict';
import { PatternMemory } from '../src/pattern-memory.js';

test('novelty lower for repeated fingerprints', ()=>{
  const m=new PatternMemory();
  const a=Uint8Array.from([1,0,1,0,1,0,1,0]);
  m.remember(a);
  const repeated=m.novelty(a);
  const unseen=m.novelty(Uint8Array.from([0,1,0,1,0,1,0,1]));
  assert.ok(repeated < unseen);
});
