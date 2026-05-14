import test from 'node:test';
import assert from 'node:assert/strict';
import { OrganismTracker } from '../src/organism-tracker.js';

function mkSnap(gen, clusters) {
  const width = 8, height = 8, size = width * height;
  const alive = new Uint8Array(size), age = new Uint16Array(size), energy = new Float32Array(size).fill(0.5), stress = new Float32Array(size), genomeId = new Uint16Array(size), lineageId = new Uint16Array(size);
  for (const [x,y,g,l] of clusters) { const i=y*width+x; alive[i]=1; age[i]=gen; genomeId[i]=g; lineageId[i]=l; }
  return { generation: gen, width, height, alive, age, energy, stress, genomeId, lineageId };
}

test('split hysteresis and cooldown', () => {
  const t = new OrganismTracker();
  t.update(mkSnap(1, [[1,1,1,1],[1,2,1,1],[2,1,1,1],[2,2,1,1]]));
  t.update(mkSnap(2, [[1,1,1,1],[1,2,1,1],[5,5,1,1],[5,6,1,1]]));
  assert.equal(t.getEventSummary(2).confirmedEventCounts.organism_split || 0, 0);
  t.update(mkSnap(3, [[1,1,1,1],[1,2,1,1],[5,5,1,1],[5,6,1,1]]));
  t.update(mkSnap(4, [[1,1,1,1],[1,2,1,1],[5,5,1,1],[5,6,1,1]]));
  const s = t.getEventSummary(4);
  assert.equal(s.confirmedEventCounts.organism_split, 1);
  t.update(mkSnap(5, [[1,1,1,1],[1,2,1,1],[5,5,1,1],[5,6,1,1]]));
  assert.ok((t.getEventSummary(5).eventSuppressionCounts.cooldown_suppressed || 0) > 0);
});

test('birth persistence before confirmation', () => {
  const t = new OrganismTracker();
  t.update(mkSnap(1, []));
  t.update(mkSnap(2, [[1,1,2,2]]));
  assert.equal(t.getEventSummary(2).confirmedEventCounts.organism_birth || 0, 0);
  t.update(mkSnap(3, [[1,1,2,2],[1,2,2,2]]));
  t.update(mkSnap(4, [[1,1,2,2],[1,2,2,2]]));
  assert.ok((t.getEventSummary(4).rawEventCounts.organism_birth || 0) >= 1);
});
