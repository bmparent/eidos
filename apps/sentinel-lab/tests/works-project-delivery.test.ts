import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { unlink } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createClient } from '@libsql/client';
import { adaptDatabase } from '../lib/works/database';
import { dispatchWorks, platformEnvironment } from '../lib/works/dispatch';
import { hash } from '../lib/works/vendor/functions/_shared/platform/core';
import { DELIVERY_LIMITS, ensureProjectDelivery, freezeProjectDelivery, readProjectDeliveryArchive, type ProjectDeliveryEnv } from '../lib/works/vendor/functions/_shared/platform/projectDelivery';
import schema from '../lib/works/vendor/functions/_shared/platform/projectDeliverySchema';
import liveSchema from '../lib/works/vendor/functions/_shared/platform/projectDeliveryLiveSchema';
import orderSchema from '../lib/works/vendor/functions/_shared/platform/projectDeliveryOrdersSchema';
import { createProject, upgradeProject } from '../lib/works/project-delivery/src/playground/model';
import { createProject as createLegacyProject } from '../lib/works/vendor/src/playground/model';
import { buildFrozenPackage, createProjectSnapshot } from '../lib/works/project-delivery/src/playground/purchaseSnapshot';

const origin = 'https://controlled-playground.pages.dev', secret = 'controlled-relay-secret-at-least-32-characters';
const source = { EIDOS_PLATFORM_TOKEN: secret, PUBLIC_SITE_URL: origin };
async function fixture(url = ':memory:') {
  const client = createClient({ url }), database = adaptDatabase(client);
  for (const name of ['0001_eidos_platform.sql', '0002_members.sql']) await client.executeMultiple(readFileSync('lib/works/vendor/migrations/' + name, 'utf8'));
  const env: ProjectDeliveryEnv = { EIDOS_RUNTIME: 'sentinel', EIDOS_DB: database, EIDOS_ACCOUNTS_ENABLED: 'true',
    EIDOS_PLAYGROUND_DELIVERY_MODE: 'test', PUBLIC_SITE_URL: origin, RESEND_API_KEY: 'controlled-fixture', EIDOS_MAIL_FROM: 'fixture@example.test',
    EIDOS_RATE_SECRET: 'controlled-rate-secret-at-least-32-characters', TURNSTILE_SECRET_KEY: 'controlled-fixture', TURNSTILE_SITE_KEY: 'controlled-fixture' };
  for (const owner of ['alice', 'bob']) {
    await database.prepare('INSERT INTO eidos_email_members(id,email,username,kind,created_at,newsletter_after) VALUES(?,?,?,?,?,?)').bind(owner, owner + '@example.test', owner, 'person', '2026-10-14', '2026-10-14').run();
    await database.prepare('INSERT INTO eidos_member_sessions(token_hash,member_id,expires) VALUES(?,?,?)').bind(await hash((owner === 'alice' ? 'a' : 'b').repeat(64)), owner, 2_000_000_000).run();
  }
  const project = { ...upgradeProject(createProject()), schemaVersion: 5 as const };
  const snapshot = await createProjectSnapshot(project, 'delivery-document', 3);
  const call = (path: string, input?: unknown, owner = 'alice', extra: Record<string, string> = {}) => dispatchWorks(new Request('https://controlled-backend.vercel.app/api/works/v1' + path, {
    method: input === undefined ? 'GET' : 'POST', headers: { 'x-eidos-platform-token': secret, 'x-eidos-site-origin': origin,
      origin, 'content-type': 'application/json', cookie: '__Host-eidos_session=' + (owner === 'alice' ? 'a' : 'b').repeat(64), ...extra },
    body: input === undefined ? undefined : JSON.stringify(input),
  }), source, env);
  return { client, database, env, snapshot, call, close: () => client.close() };
}

test('isolated renderer graph matches its portable LF manifest and leaves legacy renderer separate', () => {
  const root = 'lib/works/project-delivery/', manifest = JSON.parse(readFileSync(root + 'source-manifest.json', 'utf8'));
  assert.equal(manifest.encoding, 'utf8-lf');
  const digest = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
  const sourceFiles = (folder: string, prefix = 'src'): string[] => readdirSync(folder, { withFileTypes: true })
    .flatMap(entry => entry.isDirectory() ? sourceFiles(join(folder, entry.name), prefix + '/' + entry.name) : [prefix + '/' + entry.name]);
  assert.deepEqual(sourceFiles(root + 'src').sort(), manifest.files.map((file: {path: string}) => file.path).sort(), 'No obsolete or unmanifested source is bundled');
  assert.equal(digest(JSON.stringify(manifest.files)), manifest.inventorySha256);
  for (const file of manifest.files) {
    assert.ok(file.path.startsWith('src/playground/') || file.path === 'src/data/buyerArchivePins.json');
    const bytes = readFileSync(root + file.path);
    assert.equal(bytes.includes(Buffer.from('\r\n')), false, file.path);
    assert.equal(bytes.length, file.bytes, file.path); assert.equal(digest(bytes), file.sha256, file.path);
  }
  assert.equal(digest(readFileSync(manifest.adapter.path)), manifest.adapter.sha256);
  assert.equal(createLegacyProject().schemaVersion, 1);
  assert.equal(schema.replaceAll('\r\n', '\n'), readFileSync('lib/works/vendor/migrations/0006_project_delivery.sql', 'utf8').replaceAll('\r\n', '\n'));
  assert.equal(orderSchema.replaceAll('\r\n', '\n'), readFileSync('lib/works/vendor/migrations/0007_project_delivery_orders.sql', 'utf8').replaceAll('\r\n', '\n'));
});

test('authenticated backend relay freezes exact schema-5 bytes on real libSQL without provider calls', async () => {
  const f = await fixture(), originalFetch = globalThis.fetch;
  globalThis.fetch = async () => { throw Error('Delivery storage must not make outbound requests.'); };
  try {
    const expected = await buildFrozenPackage(f.snapshot);
    const replies = await Promise.all(Array.from({ length: 4 }, () => f.call('/api/playground/delivery', { requestId: 'relay-delivery-request', snapshot: f.snapshot })));
    for (const response of replies) assert.equal(response.status, 200, await response.clone().text());
    const delivered = await Promise.all(replies.map(response => response.json()));
    assert.equal(new Set(delivered.map(result => result.delivery.id)).size, 1);
    const saved = delivered[0].delivery;
    assert.equal(saved.archiveSha256, expected.receipt.archiveSha256); assert.equal(saved.entitlement, 'not-verified');
    assert.deepEqual((await readProjectDeliveryArchive(f.env, 'alice', saved.id)).archive, expected.archive);
    const reopened = await f.call('/api/playground/delivery?id=' + saved.id); assert.equal(reopened.status, 200);
    assert.deepEqual((await reopened.json()).delivery.snapshot, f.snapshot);
    assert.equal((await f.call('/api/playground/delivery?id=' + saved.id, undefined, 'bob')).status, 404);
    assert.equal((await f.call('/api/playground/delivery?id=' + saved.id, undefined, 'alice', { 'x-eidos-platform-token': 'wrong' })).status, 401);
    assert.equal((await f.call('/api/playground/delivery', { requestId: 'forged-owner-request', snapshot: f.snapshot, owner: 'bob' })).status, 400);
    assert.equal((await f.call('/api/playground/projects', { document: f.snapshot.project })).status, 409);
    assert.equal((await f.database.prepare('SELECT COUNT(*) n FROM eidos_pg_delivery_snapshots').first<{n:number}>())?.n, 1);
  } finally { globalThis.fetch = originalFetch; f.close(); }
});

test('delivery mode is opt-in and backend body allowance exceeds legacy ceiling but remains bounded', async () => {
  const f = await fixture();
  try {
    assert.equal((platformEnvironment({}) as ProjectDeliveryEnv).EIDOS_PLAYGROUND_DELIVERY_MODE, undefined);
    assert.equal((platformEnvironment({ EIDOS_PLAYGROUND_DELIVERY_MODE: 'test' }) as ProjectDeliveryEnv).EIDOS_PLAYGROUND_DELIVERY_MODE, 'test');
    f.env.EIDOS_PLAYGROUND_DELIVERY_MODE = undefined;
    assert.equal((await f.call('/api/playground/delivery', { requestId: 'disabled-request', snapshot: f.snapshot })).status, 503);
    assert.equal((await f.call('/api/playground/delivery?capabilities=1')).status, 503);
    assert.equal(await f.database.prepare("SELECT name FROM sqlite_master WHERE name='eidos_pg_delivery_snapshots'").first(), null);
    f.env.EIDOS_PLAYGROUND_DELIVERY_MODE = 'test'; f.env.PUBLIC_SITE_URL = 'https://eidos-works.com';
    assert.equal((await f.call('/api/playground/delivery', { requestId: 'canonical-request', snapshot: f.snapshot })).status, 503);
    f.env.PUBLIC_SITE_URL = 'https://eidosworks.pages.dev';
    assert.equal((await f.call('/api/playground/delivery', { requestId: 'canonical-pages-request', snapshot: f.snapshot })).status, 503);
    f.env.PUBLIC_SITE_URL = origin;
    // A >12KB malformed input reaches the delivery validator (400), proving that
    // the older generic relay ceiling does not reject legitimate larger snapshots.
    assert.equal((await f.call('/api/playground/delivery', { requestId: 'body-size-request', snapshot: f.snapshot, unknown: 'x'.repeat(13000) })).status, 400);
    const tooLarge = await f.call('/api/playground/delivery', { extra: 'x'.repeat(DELIVERY_LIMITS.requestBytes) });
    assert.equal(tooLarge.ok, false);
    assert.equal(await f.database.prepare("SELECT name FROM sqlite_master WHERE name='eidos_pg_delivery_snapshots'").first(), null);
  } finally { f.close(); }
});

test('real libSQL chunk faults roll back and separate worker connections cannot exceed owner quota', async (t) => {
  const path = join(tmpdir(), 'eidos-delivery-fixture-' + crypto.randomUUID() + '.db'), url = 'file:' + path.replaceAll('\\', '/');
  const f = await fixture(url), second = createClient({ url }), otherEnv = { ...f.env, EIDOS_DB: adaptDatabase(second) };
  try {
    await ensureProjectDelivery(f.env);
    await f.client.executeMultiple("CREATE TRIGGER controlled_chunk_failure BEFORE INSERT ON eidos_pg_delivery_chunks WHEN NEW.ordinal=0 BEGIN SELECT RAISE(ABORT,'controlled chunk fault'); END;");
    await assert.rejects(freezeProjectDelivery(f.env, 'alice', { requestId: 'chunk-fault-request', snapshot: f.snapshot }), /controlled chunk fault/);
    assert.equal((await f.database.prepare('SELECT COUNT(*) n FROM eidos_pg_delivery_snapshots').first<{n:number}>())?.n, 0);
    assert.equal((await f.database.prepare('SELECT COUNT(*) n FROM eidos_pg_delivery_chunks').first<{n:number}>())?.n, 0);
    await f.client.execute('DROP TRIGGER controlled_chunk_failure');
    for (let index = 0; index < DELIVERY_LIMITS.ownerCount - 1; index++)
      await freezeProjectDelivery(f.env, 'alice', { requestId: 'quota-seed-' + index, snapshot: f.snapshot });
    const attempts = await Promise.allSettled([
      freezeProjectDelivery(f.env, 'alice', { requestId: 'quota-racer-one', snapshot: f.snapshot }),
      freezeProjectDelivery(otherEnv, 'alice', { requestId: 'quota-racer-two', snapshot: f.snapshot }),
    ]);
    assert.equal(attempts.filter(result => result.status === 'fulfilled').length, 1);
    const rejected = attempts.find(result => result.status === 'rejected');
    assert.match(String(rejected && 'reason' in rejected ? rejected.reason : ''), /storage limit/);
    assert.equal((await f.database.prepare("SELECT COUNT(*) n FROM eidos_pg_delivery_snapshots WHERE owner_id='alice'").first<{n:number}>())?.n, DELIVERY_LIMITS.ownerCount);
    assert.equal((await f.database.prepare('SELECT COUNT(*) n FROM eidos_pg_delivery_chunks c LEFT JOIN eidos_pg_delivery_snapshots s ON s.id=c.snapshot_id WHERE s.id IS NULL').first<{n:number}>())?.n, 0);
    assert.equal((await freezeProjectDelivery(otherEnv, 'bob', { requestId: 'separate-owner-quota', snapshot: f.snapshot })).state, 'frozen');
  } finally {
    second.close(); f.close();
    for (const suffix of ['', '-wal', '-shm']) await unlink(path + suffix).catch(error => {
      if (error.code === 'EBUSY' && process.platform === 'win32') t.diagnostic('Windows libSQL retains the synthetic temp database handle until process exit: ' + path + suffix);
      else if (error.code !== 'ENOENT') throw error;
    });
  }
});

test('LIVE delivery opt-in propagates without defaults and preserves TEST data through isolated libsql storage', async () => {
  const f = await fixture();
  try {
    assert.equal((platformEnvironment({}) as ProjectDeliveryEnv).EIDOS_PLAYGROUND_DELIVERY_LIVE_ENABLED, undefined);
    assert.equal((platformEnvironment({ EIDOS_PLAYGROUND_DELIVERY_LIVE_ENABLED: 'true' }) as ProjectDeliveryEnv).EIDOS_PLAYGROUND_DELIVERY_LIVE_ENABLED, 'true');
    assert.equal(liveSchema, readFileSync('lib/works/vendor/migrations/0008_project_delivery_live.sql', 'utf8'));
    const old = await freezeProjectDelivery(f.env, 'alice', { requestId: 'preserved-test-page', snapshot: f.snapshot });
    const oldRows = await f.database.prepare('SELECT * FROM eidos_pg_delivery_snapshots').all();
    const testEnv = { ...f.env };
    Object.assign(f.env, { EIDOS_PLAYGROUND_DELIVERY_MODE: 'live', PUBLIC_SITE_URL: 'https://eidos-works.com', EIDOS_LOCAL_TEST: 'false',
      STRIPE_SECRET_KEY: 'sk_live_controlled_fixture', EIDOS_TEMPLATE_STRIPE_ACCOUNT_ID: 'acct_controlledfixture',
      EIDOS_TEMPLATE_LIVE_ENABLED: 'true', EIDOS_SHOP_ENABLED: 'true', EIDOS_KIT_WEBHOOK_SECRET: 'whsec_controlled_fixture', RESEND_API_KEY: 're_controlled_fixture',
      EIDOS_TEMPLATE_ARCHIVES: { async get() { return null; } } });
    await assert.rejects(() => freezeProjectDelivery(f.env, 'alice', { requestId: 'gated-live-page', snapshot: f.snapshot }), /not enabled/);
    f.env.EIDOS_PLAYGROUND_DELIVERY_LIVE_ENABLED = 'true';
    const live = await freezeProjectDelivery(f.env, 'alice', { requestId: 'isolated-live-page', snapshot: f.snapshot });
    assert.equal(live.mode, 'live'); assert.notEqual(live.id, old.id);
    assert.deepEqual(await f.database.prepare('SELECT * FROM eidos_pg_delivery_snapshots').all(), oldRows);
    await assert.rejects(() => readProjectDeliveryArchive(f.env, 'alice', old.id), /not found/);
    await assert.rejects(() => readProjectDeliveryArchive(testEnv, 'alice', live.id), /not found/);
    await assert.rejects(() => readProjectDeliveryArchive(f.env, 'bob', live.id), /not found/);
    f.env.EIDOS_PLAYGROUND_DELIVERY_LIVE_ENABLED = 'false'; f.env.EIDOS_SHOP_ENABLED = 'false';
    assert.equal((await readProjectDeliveryArchive(f.env, 'alice', live.id)).archiveSha256, live.archiveSha256);
    assert.equal((await readProjectDeliveryArchive(testEnv, 'alice', old.id)).archiveSha256, old.archiveSha256);
  } finally { f.close(); }
});
