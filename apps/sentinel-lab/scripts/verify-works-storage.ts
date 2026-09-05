/** Explicit remote database smoke test; touches only its uniquely named probe records. */
import assert from 'node:assert/strict';
import { createClient } from '@libsql/client';
import { randomUUID } from 'node:crypto';
import { mkdirSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { adaptDatabase } from '../lib/works/database';
import { reserve } from '../lib/works/vendor/functions/_shared/platform/core';

const options = { url: process.env.EIDOS_DATABASE_URL!, authToken: process.env.EIDOS_DATABASE_AUTH_TOKEN! };
assert.ok(options.url && options.authToken, 'Remote database configuration required');
assert.ok(['https:', 'libsql:'].includes(new URL(options.url).protocol), 'Use a remote database');
if (process.argv.includes('--verify-read')) {
  const client = createClient(options);
  try {
    const result = await client.execute({ sql: 'SELECT used FROM eidos_quotas WHERE bucket=?', args: [process.env.EIDOS_STORAGE_PROBE!] });
    console.log(JSON.stringify({ used: result.rows[0]?.used }));
  } finally { client.close(); }
} else {
  const id = 'release-probe-' + randomUUID();
  const clients = Array.from({ length: 3 }, () => createClient(options));
  const databases = clients.map(adaptDatabase);
  const artifact = new URL('../../../artifacts/works-release-20260905/storage-smoke.json', import.meta.url);
  mkdirSync(new URL('.', artifact), { recursive: true });
  const receipt: Record<string, unknown> = { timestamp_utc: new Date().toISOString(), environment: 'dedicated remote Turso database', probe: id, status: 'running' };
  try {
    const attempts = await Promise.all(Array.from({ length: 30 }, (_, i) => reserve(databases[i % 3], id, 100, 750)));
    const accepted = attempts.filter(Boolean).length;
    assert.equal(accepted, 7);
    const used = await databases[0].prepare('SELECT used FROM eidos_quotas WHERE bucket=?').bind(id).first<{ used: number }>();
    assert.equal(used?.used, 700);
    await assert.rejects(databases[0].batch([
      databases[0].prepare('INSERT INTO eidos_stripe_events(id,received_at) VALUES(?,?)').bind(id, new Date().toISOString()),
      databases[0].prepare('INSERT INTO eidos_release_missing_table VALUES(?)').bind('rollback-probe'),
    ]));
    assert.equal(await databases[1].prepare('SELECT id FROM eidos_stripe_events WHERE id=?').bind(id).first(), null);
    const restarted = spawnSync(process.execPath, ['--import', './apps/sentinel-lab/node_modules/tsx/dist/loader.mjs', 'apps/sentinel-lab/scripts/verify-works-storage.ts', '--verify-read'], {
      cwd: process.cwd(), encoding: 'utf8', env: { ...process.env, EIDOS_STORAGE_PROBE: id }, timeout: 30_000,
    });
    assert.equal(restarted.status, 0, 'Fresh process must read the durable reservation');
    assert.equal(JSON.parse(restarted.stdout).used, 700);
    Object.assign(receipt, { status: 'passed', clients: 3, concurrent_attempts: 30, accepted, reserved_tokens: 700, limit: 750, transaction_rollback: 'passed', fresh_process_persistence: 'passed', math: 'floor(750 / 100) = 7 accepted reservations; 7 * 100 = 700 <= 750' });
  } catch (error) {
    Object.assign(receipt, { status: 'failed', reason: error instanceof assert.AssertionError ? error.message : 'Remote storage operation failed; credentials and provider payloads omitted' });
    process.exitCode = 1;
  } finally {
    try {
      await clients[0].batch([
        { sql: 'DELETE FROM eidos_quotas WHERE bucket=?', args: [id] },
        { sql: 'DELETE FROM eidos_stripe_events WHERE id=?', args: [id] },
      ], 'write');
      receipt.probe_cleanup = 'passed';
    } catch { receipt.probe_cleanup = 'failed'; process.exitCode = 1; }
    for (const client of clients) client.close();
    writeFileSync(artifact, JSON.stringify(receipt, null, 2) + '\n');
    console.log(JSON.stringify(receipt));
  }
}
