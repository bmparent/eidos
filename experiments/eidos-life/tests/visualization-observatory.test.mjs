import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const experimentRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');

test('visualization uses the 3D observatory renderer', async () => {
  const source = await readFile(resolve(experimentRoot, 'src/visualization.js'), 'utf8');

  assert.match(source, /InstancedMesh/);
  assert.match(source, /BoxGeometry|CylinderGeometry/);
  assert.match(source, /DataTexture/);
  assert.match(source, /FogExp2|Fog/);
  assert.match(source, /PerspectiveCamera/);
  assert.match(source, /organisms/);
  assert.match(source, /pulse\(/);
  assert.match(source, /cellMesh\s*=\s*new THREE\.InstancedMesh/);
});
