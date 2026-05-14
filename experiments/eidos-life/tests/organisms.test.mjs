import test from 'node:test'; import assert from 'node:assert/strict';
import { trackOrganisms } from '../src/organisms.js';

test('organism IDs remain unique when matching previous and adding new organisms', ()=>{
  const width = 5, height = 5, size = width * height;
  const alive = new Uint8Array(size);
  const age = new Uint16Array(size);
  const energy = new Float32Array(size).fill(0.7);
  const stress = new Float32Array(size).fill(0.1);
  const idx=(x,y)=>((y+height)%height)*width+((x+width)%width);

  alive[idx(0,0)] = 1; // matches previous id=1
  alive[idx(4,4)] = 1; // new organism
  age[idx(0,0)] = 5;
  age[idx(4,4)] = 2;

  const organisms = trackOrganisms({
    alive, age, energy, stress, width, height,
    previous: [{ id: 1, centroid: { x: 0, y: 0 }, mass: 1 }]
  });

  assert.equal(organisms.length, 2);
  const ids = organisms.map((o) => o.id);
  assert.equal(new Set(ids).size, ids.length);
  assert.ok(ids.includes(1));
});
