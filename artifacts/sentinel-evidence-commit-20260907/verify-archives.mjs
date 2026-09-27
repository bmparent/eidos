import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';

// Run from repo root; read-only, with no credential or provider access.
const inventory = JSON.parse(readFileSync(new URL('./run_manifest.json', import.meta.url)));
const mode = process.argv[2] ?? '--worktree';
assert.ok(['--worktree', '--index', '--head'].includes(mode), 'Use --worktree, --index or --head');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const blobs = new Map();
if (mode !== '--worktree') {
  const prefix = mode === '--index' ? ':' : 'HEAD:';
  const batch = execFileSync('git', ['cat-file', '--batch'], {
    input: inventory.files.map(file => `${prefix}${file.path}\n`).join(''),
    maxBuffer: 32 * 1024 * 1024, windowsHide: true,
  });
  let offset = 0;
  for (const file of inventory.files) {
    const end = batch.indexOf(10, offset);
    assert.ok(end >= offset, `Missing Git response: ${file.path}`);
    const header = batch.subarray(offset, end).toString().split(' ');
    assert.equal(header[1], 'blob', `Missing Git blob: ${file.path}`);
    const size = Number(header[2]);
    assert.ok(Number.isSafeInteger(size) && size >= 0);
    blobs.set(file.path, batch.subarray(end + 1, end + 1 + size));
    offset = end + 1 + size + 1;
  }
  assert.equal(offset, batch.length);
}
let bytes = 0;
for (const file of inventory.files) {
  const actual = mode === '--worktree' ? readFileSync(file.path) : blobs.get(file.path);
  assert.equal(actual.length, file.bytes, `${file.path}: byte count`);
  assert.equal(sha(actual), file.sha256, `${file.path}: SHA-256`);
  if (file.path.endsWith('.json')) JSON.parse(actual);
  if (file.path.endsWith('.jsonl')) actual.toString().trim().split(/\r?\n/).forEach(line => JSON.parse(line));
  bytes += actual.length;
}
console.log(JSON.stringify({ status: 'passed', mode, files: inventory.files.length, bytes, allHashesMatch: true }));
