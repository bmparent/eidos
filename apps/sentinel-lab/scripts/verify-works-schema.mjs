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
const applyMissing = process.argv.includes('--apply-isolated-schema');
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
  if (applyMissing) assert.equal(receipt.databaseIdentityHash, '4882b7d6be40a19a97d5dabce311089dc72a984e5a2b3b3f64081216926bb353', 'Previously verified isolated database required');
  client = createClient({ url, authToken });
  const directory = new URL('./lib/works/vendor/migrations/', 'file://' + process.cwd() + '/');
  const migrationNames = (await readdir(directory)).filter(name => /^\d+.*\.sql$/.test(name)).sort();
  expected = new DatabaseSync(':memory:');
  for (const name of migrationNames) expected.exec(await readFile(new URL(name, directory), 'utf8'));
  const objects = expected.prepare("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name").all();
  let live = await client.execute("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name");
  let byName = new Map(live.rows.map(row => [row.name, row]));
  const normalize = sql => String(sql).replace(/\bIF\s+NOT\s+EXISTS\b/gi, '').replace(/\s+/g, '').replace(/;$/, '').toLowerCase();
  let missing = objects.filter(object => !byName.has(object.name)).map(object => ({ type: object.type, name: object.name }));
  let different = objects.filter(object => byName.has(object.name) && normalize(byName.get(object.name).sql) !== normalize(object.sql)).map(object => ({ type: object.type, name: object.name }));
  if (applyMissing) {
    assert.equal(different.length, 0, 'Existing schema must match before migration');
    assert.ok(missing.every(object => /^eidos_(ops_|owner_audit)/.test(object.name)), 'Only the missing operations and owner-audit schema may be added');
    const beforeObjects = live.rows.map(row => ({ name: row.name, sql: normalize(row.sql) }));
    const beforeCounts = {};
    for (const object of live.rows.filter(object => object.type === 'table' && /^eidos_[a-z0-9_]+$/.test(object.name))) {
      beforeCounts[object.name] = Number((await client.execute(`SELECT COUNT(*) AS count FROM "${object.name}"`)).rows[0].count);
    }
    const integrityBefore = await client.execute('PRAGMA quick_check');
    assert.ok(integrityBefore.rows.every(row => Object.values(row).includes('ok')), 'Database integrity required before migration');
    assert.equal((await client.execute('PRAGMA foreign_key_check')).rows.length, 0, 'Foreign-key integrity required before migration');
    const appliedMigrations = [];
    if (missing.length) {
      for (const name of ['0007_operations.sql', '0007_owner_audit.sql']) {
        const schema = await readFile(new URL(name, directory), 'utf8');
        assert.ok(!/\b(?:DROP|ALTER|DELETE|UPDATE|INSERT|REPLACE)\b/i.test(schema), 'Migration must only add schema');
        await client.executeMultiple(schema);
        appliedMigrations.push(name);
      }
    }
    live = await client.execute("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name");
    byName = new Map(live.rows.map(row => [row.name, row]));
    for (const object of beforeObjects) assert.equal(normalize(byName.get(object.name)?.sql), object.sql, 'Existing schema must be preserved');
    for (const [name, count] of Object.entries(beforeCounts)) {
      assert.equal(Number((await client.execute(`SELECT COUNT(*) AS count FROM "${name}"`)).rows[0].count), count, 'Existing record counts must be preserved');
    }
    missing = objects.filter(object => !byName.has(object.name)).map(object => ({ type: object.type, name: object.name }));
    different = objects.filter(object => byName.has(object.name) && normalize(byName.get(object.name).sql) !== normalize(object.sql)).map(object => ({ type: object.type, name: object.name }));
    Object.assign(receipt, { mode: 'explicit additive migration on verified isolated preview', appliedMigrations, existingSchemaPreserved: true, existingRecordCountsPreserved: true });
  }
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
    configuredProviders: {
      accountsEnabled: process.env.EIDOS_ACCOUNTS_ENABLED === 'true',
      passwordEnabled: process.env.EIDOS_PASSWORD_AUTH_ENABLED === 'true',
      googleClientPresent: Boolean(process.env.EIDOS_GOOGLE_CLIENT_ID && process.env.EIDOS_GOOGLE_CLIENT_SECRET),
      googleRedirectConfigured: Boolean(process.env.EIDOS_GOOGLE_REDIRECT_URI),
      mailKeyPresent: Boolean(process.env.RESEND_API_KEY),
      turnstilePresent: Boolean(process.env.TURNSTILE_SECRET_KEY && process.env.TURNSTILE_SITE_KEY),
      kitShopEnabled: process.env.EIDOS_SHOP_ENABLED === 'true',
      kitStripeMode: /^sk_test_/.test(process.env.STRIPE_SECRET_KEY || '') ? 'test' : /^sk_live_/.test(process.env.STRIPE_SECRET_KEY || '') ? 'live' : 'unavailable',
      kitWebhookPresent: Boolean(process.env.EIDOS_KIT_WEBHOOK_SECRET),
      playgroundStripeMode: /^sk_test_/.test(process.env.EIDOS_PLAYGROUND_STRIPE_KEY || '') ? 'test' : /^sk_live_/.test(process.env.EIDOS_PLAYGROUND_STRIPE_KEY || '') ? 'live' : 'unavailable',
      playgroundWebhookPresent: Boolean(process.env.EIDOS_PLAYGROUND_WEBHOOK_SECRET),
    },
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
  // Provider log entries have a per-line size limit. Preserve a complete,
  // reconstructable receipt without logging credentials or customer rows.
  const encoded = JSON.stringify(receipt);
  const chunkSize = 600;
  for (let offset = 0; offset < encoded.length; offset += chunkSize) {
    console.log(JSON.stringify({ marker: receipt.marker, part: 1 + offset / chunkSize,
      parts: Math.ceil(encoded.length / chunkSize), chunk: encoded.slice(offset, offset + chunkSize) }));
  }
}
