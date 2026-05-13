import test from 'node:test'; import assert from 'node:assert/strict';
import { LifeEngine } from '../src/life-engine.js';

test('blinker oscillates under green rule', ()=>{
 const e=new LifeEngine({width:5,height:5}); e.seed([[1,2],[2,2],[3,2]]);
 e.step({birth:[3],survive:[2,3],mutation:0});
 assert.equal(e.alive[e.idx(2,1)],1); assert.equal(e.alive[e.idx(2,2)],1); assert.equal(e.alive[e.idx(2,3)],1);
});

test('wrapped neighbor count works', ()=>{ const e=new LifeEngine({width:5,height:5}); e.seed([[0,0],[4,0],[0,4]]); assert.equal(e.countNeighbors(4,4),3); });
