import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, readdir } from 'node:fs/promises';
import { DatabaseSync } from 'node:sqlite';
import { createClient } from '@libsql/client';

const receipt = {
  marker: 'EIDOS_ISOLATED_SCHEMA_ACCEPTANCE',
  timestamp: new Date().toISOString(),
  sourceRevision: process.env.VERCEL_GIT_COMMIT_SHA,
  environment: process.env.VERCEL_ENV,
  bindingPrefix: process.env.EIDOS_DATABASE_BINDING_PREFIX,
  storeId: 'store_2SAY7236pGjHs3fN',
  mode: 'read-only',
};
let client;
let expected;
try {
  assert.equal(receipt.environment, 'preview', 'Preview environment required');
  assert.match(receipt.sourceRevision || '', /^[a-f0-9]{40}$/, 'Git source revision required');
  assert.equal(process.env.VERCEL_GIT_COMMIT_REF, 'codex/works-paired-integration-20260927', 'Paired preview branch required');
  assert.equal(receipt.bindingPrefix, 'EIDOS_PG_INTEGRATION', 'Dedicated preview binding required');
  const url = process.env.EIDOS_PG_INTEGRATION_TURSO_DATABASE_URL;
  const authToken = process.env.EIDOS_PG_INTEGRATION_TURSO_AUTH_TOKEN;
  assert.ok(url && authToken, 'Dedicated preview credentials required');
  const parsed = new URL(url);
  assert.ok(['https:', 'libsql:'].includes(parsed.protocol) && !parsed.username && !parsed.password, 'Remote database required');
  receipt.databaseIdentityHash = createHash('sha256').update(parsed.hostname + parsed.pathname).digest('hex');
  client = createClient({ url, authToken });
  const directory = new URL('./lib/works/vendor/migrations/', 'file://' + process.cwd() + '/');
  const migrationNames = (await readdir(directory)).filter(name => /^\d+.*\.sql$/.test(name)).sort();
  expected = new DatabaseSync(':memory:');
  for (const name of migrationNames) expected.exec(await readFile(new URL(name, directory), 'utf8'));
  const objects = expected.prepare("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name").all();
  const live = await client.execute("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name");
  const byName = new Map(live.rows.map(row => [row.name, row]));
  const normalize = sql => String(sql).replace(/\bIF\s+NOT\s+EXISTS\b/gi, '').replace(/\s+/g, '').replace(/;$/, '').toLowerCase();
  const missing = objects.filter(object => !byName.has(object.name)).map(object => ({ type: object.type, name: object.name }));
  const different = objects.filter(object => byName.has(object.name) && normalize(byName.get(object.name).sql) !== normalize(object.sql)).map(object => ({ type: object.type, name: object.name }));
  const counts = {};
  for (const object of objects.filter(object => object.type === 'table' && byName.has(object.name))) {
    assert.match(object.name, /^[a-zA-Z_][a-zA-Z0-9_]*$/);
    const result = await client.execute(`SELECT COUNT(*) AS count FROM "${object.name}"`);
    counts[object.name] = Number(result.rows[0].count);
  }
  const integrity = await client.execute('PRAGMA quick_check');
  const foreignKeys = await client.execute('PRAGMA foreign_key_check');
  Object.assign(receipt, {
    migrations: migrationNames,
    expectedObjects: objects.length,
    expectedTables: objects.filter(object => object.type === 'table').length,
    missingObjects: missing,
    differentObjects: different,
    recordCounts: counts,
    quickCheck: integrity.rows.every(row => Object.values(row).includes('ok')) ? 'passed' : 'failed',
    foreignKeyViolations: foreignKeys.rows.length,
    status: missing.length || different.length || foreignKeys.rows.length || !integrity.rows.every(row => Object.values(row).includes('ok')) ? 'failed' : 'passed',
  });
  if (receipt.status !== 'passed') process.exitCode = 1;
} catch (error) {
  receipt.status = 'failed';
  receipt.reason = error instanceof assert.AssertionError ? error.message : 'Database verification failed; provider payloads and credentials omitted';
  process.exitCode = 1;
} finally {
  client?.close();
  expected?.close();
  console.log(JSON.stringify(receipt));
}
