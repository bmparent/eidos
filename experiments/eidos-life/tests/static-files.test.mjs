import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const experimentRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');

test('static browser entrypoint uses local vendored Three.js', async () => {
  for (const path of [
    'index.html',
    'src/app.js',
    'src/visualization.js',
    'vendor/three.module.js',
  ]) {
    assert.ok(existsSync(resolve(experimentRoot, path)), `${path} missing`);
  }

  const index = await readFile(resolve(experimentRoot, 'index.html'), 'utf8');
  assert.match(index, /"three"\s*:\s*"\.\/vendor\/three\.module\.js"/);
});
