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
  assert.match(index, /id="summaryExportBtn"/);
  assert.match(index, /id="saveCheckpointBtn"/);
  assert.match(index, /id="exportBtn"/);
  assert.match(index, /id="exportWorldBtn"/);

  const app = await readFile(resolve(experimentRoot, 'src/app.js'), 'utf8');
  assert.match(app, /AUTOSAVE_METADATA_INTERVAL\s*=\s*5000/);
  assert.match(app, /CHECKPOINT_MODE_DEFAULT\s*=\s*'metadata_only'/);
});
