import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { resolve, join } from 'node:path';
import { canonicalJson } from '../lib/experiments/lock.js';

// Run from repository root. The directory contains authenticated UI downloads;
// this command needs no credential and never requests a new experiment.
const dir = resolve(process.argv[2] || 'artifacts/sentinel-production-live-20260906');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const read = name => readFileSync(join(dir, name));
const json = name => JSON.parse(read(name));
const manifest = json('run_manifest.json');
const source = json('source_receipt.json');
const dataset = json('dataset_receipt.json');
const metrics = json('metrics.json');
const receipt = json('job-receipt.json');
assert.equal(receipt.jobId, manifest.job_id);
assert.equal(source.jobId, manifest.job_id);
assert.equal(sha(canonicalJson(manifest.spec)), manifest.lock_digest);
assert.equal(dataset.dataset.file_sha256, manifest.spec.dataset.expectedSha256);
assert.equal(manifest.engine.execution_profile, 'cpu_engineering');
assert.equal(manifest.spec.engine.seed, 0);
assert.deepEqual(dataset.rows, { calibration: 200, evaluation: 600, sealed_holdout: 200, sent_to_engine: 800, total: 1000 });
assert.equal(dataset.label_isolation.heldout_sent_to_engine, false);
assert.equal(dataset.label_isolation.engine_metadata_contains_labels, false);
assert.equal(dataset.features.source_columns.includes('Label'), false);
assert.equal(metrics.labels_unsealed_after_prediction_freeze, true);
assert.equal(manifest.proof.heldout_evaluated, false);
assert.equal(manifest.proof.gates_advanced, 0);
assert.equal(metrics.prediction_trace_sha256, sha(read('engine_trace.jsonl')));
assert.equal(manifest.metrics_sha256, sha(read('metrics.json')));
assert.equal(manifest.dataset_receipt_sha256, sha(read('dataset_receipt.json')));
const downloaded = [];
for (const [name, expected] of Object.entries(manifest.artifacts)) {
  if (!existsSync(join(dir, name))) continue;
  const bytes = read(name);
  assert.deepEqual({ bytes: bytes.length, sha256: sha(bytes) }, expected, name);
  downloaded.push(name);
}
const sourceHashes = [];
for (const [name, expected] of Object.entries(source.files)) {
  const actual = sha(execFileSync('git', ['show', `${source.commit}:${name}`], { windowsHide: true }));
  assert.equal(actual, expected, name);
  sourceHashes.push({ path: name, sha256: actual });
}
const engine = read('engine_trace.jsonl').toString().trim().split('\n').map(JSON.parse);
const evaluation = read('evaluation_trace.jsonl').toString().trim().split('\n').map(JSON.parse);
const byStep = new Map(engine.map(row => [row.step, row]));
assert.equal(evaluation.length, 600);
assert.equal(metrics.prediction_coverage_complete, true);
for (const [i, row] of evaluation.entries()) {
  assert.equal(row.source_row_index, 200 + i);
  const prediction = byStep.get(row.step);
  assert.ok(prediction);
  assert.equal(row.z, prediction.z);
  assert.equal(row.is_surprise, prediction.is_surprise);
  assert.equal(row.z_threshold, prediction.z_thresh_eff);
}
assert.ok(engine.every(row => row.step < 800));
const alerts = evaluation.filter(row => row.is_surprise).length;
assert.equal(metrics.positive_rows, 0);
assert.deepEqual(metrics.confusion, { tp: 0, fp: alerts, tn: 600 - alerts, fn: 0 });
assert.equal(metrics.false_positive_rate, alerts / 600);
assert.equal(metrics.recall, null);
assert.equal(metrics.roc_auc, null);
const verification = existsSync(join(dir, 'artifact_verification.json')) ? json('artifact_verification.json') : null;
if (verification) {
  assert.equal(verification.jobId, manifest.job_id);
  assert.equal(verification.manifestSha256, sha(read('run_manifest.json')));
  assert.equal(verification.declaredCount, Object.keys(manifest.artifacts).length);
  assert.equal(verification.allMatched, true);
  assert.equal(verification.matchedCount, verification.declaredCount);
  assert.equal(verification.resumedForRetrieval, true);
  assert.equal(verification.providerStatusAfterRetrieval, 'stopped');
  for (const file of verification.files) assert.deepEqual(file.actual, manifest.artifacts[file.path]);
}
const result = { schema: 'eidos.sentinel-lab.live-evidence-verification.v0.1', verifiedAt: new Date().toISOString(), jobId: manifest.job_id, diagnosticId: receipt.diagnosticId, executionCommit: source.commit, lockDigest: manifest.lock_digest, downloadedImmutableCount: downloaded.length, declaredImmutableCount: Object.keys(manifest.artifacts).length, fullSnapshotVerification: verification ? 'passed' : 'pending', sourceHashes, downloaded, evaluationRows: evaluation.length, predictionCorrespondence: '600 matching frozen engine predictions', falsePositiveRate: alerts / 600, confusion: metrics.confusion, recall: null, rocAuc: null, heldoutEvaluated: false, gatesAdvanced: 0, localChecksPassed: true };
writeFileSync(join(dir, 'live-evidence-verification.json'), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result, null, 2));
