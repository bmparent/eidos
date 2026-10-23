import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { createClient } from '@libsql/client';

// Controlled QA only. No fixture sessions, email accounts or local HTTP server.
// Direct mode tests real Vercel HTTP with a trusted-relay IP fixture. Edge mode
// tests actual Pages transport and never supplies a client-IP header.
const secretFile = process.env.EXCHANGE_QA_PRIVATE_FILE;
assert.ok(secretFile, 'Provide the dedicated private QA configuration file');
const secrets = JSON.parse(await readFile(secretFile, 'utf8'));
const transport = process.env.EXCHANGE_QA_TRANSPORT || 'direct';
assert.ok(['direct', 'edge'].includes(transport));
const relayFixtureIp = process.env.EXCHANGE_QA_RELAY_FIXTURE_IP || '192.0.2.190';
assert.ok(/^192\.0\.2\.(?:[1-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-4])$/.test(relayFixtureIp), 'Only a documented reserved relay fixture IP is allowed');
const origin = 'https://qa.eidos-exchange-qa-20261023.pages.dev';
const base = new URL(process.env.EXCHANGE_QA_BASE_URL || 'https://eidos-exchange-qa-20261023-fbo1nr9j7-1brentbm-1876s-projects.vercel.app');
assert.ok(base.protocol === 'https:' && !base.username && !base.password && base.pathname === '/' && !base.search && !base.hash && !base.port);
assert.ok(transport === 'edge' ? base.origin === origin : /^eidos-exchange-qa-20261023-[a-z0-9]+-1brentbm-1876s-projects\.vercel\.app$/.test(base.hostname), 'Only the isolated QA runtime is allowed');
const databaseUrl = new URL(secrets.EIDOS_EXCHANGE_TEST_TURSO_DATABASE_URL);
const databaseIdentityHash = createHash('sha256').update(databaseUrl.hostname + databaseUrl.pathname).digest('hex');
assert.equal(databaseIdentityHash, '0599028d2df7969a4992e86be8971499744c42e83382a41df7cbe13b28f49c48');
const client = createClient({ url: databaseUrl.href, authToken: secrets.EIDOS_EXCHANGE_TEST_TURSO_AUTH_TOKEN });
const continuation = secretFile + '.' + transport + '-continuation';
const relayRegistration = process.argv.includes('--relay-registration');
const registrationBase = new URL(process.env.EXCHANGE_QA_REGISTRATION_BASE_URL || base.origin);
if (relayRegistration) assert.ok(transport === 'edge' && /^eidos-exchange-qa-20261023-[a-z0-9]+-1brentbm-1876s-projects\.vercel\.app$/.test(registrationBase.hostname) && registrationBase.protocol === 'https:' && registrationBase.pathname === '/');
const receipt = { time: new Date().toISOString(), environment: 'isolated-hosted-' + transport + '-HTTP', baseUrl: base.origin,
  databaseIdentityHash, backendImplementation: '0c24ea9fba44b1f2f9183fc15845c2219dec424b', checks: [],
  independentOperatorsObserved: 0, productionWrites: 0, trustedRelayIpFixture: transport === 'direct',
  ...(transport === 'direct' ? { relayFixtureIp } : {}), status: 'failed' };
if (relayRegistration) receipt.registrationTransport = 'Explicit trusted-relay reserved-IP fixture; all subsequent lifecycle requests use actual Pages transport. Prior edge signup receipts are separate.';
let step = 'configuration', a, b, keepCredential = false;
const progress = label => { step = label; console.log('QA check: ' + label); };
async function call(path, data, key, extra = {}) {
  const headers = { 'user-agent': 'EidosExchangeQA/1.0 (controlled security acceptance)',
    ...(transport === 'direct' ? { 'x-vercel-protection-bypass': secrets.EIDOS_PLATFORM_PREVIEW_BYPASS,
      'x-eidos-platform-token': secrets.EIDOS_PLATFORM_TOKEN, 'x-eidos-site-origin': origin, 'x-eidos-client-ip': relayFixtureIp }
      : { 'x-eidos-qa-token': secrets.EIDOS_EXCHANGE_QA_ONLY_TOKEN }),
    ...(data === undefined ? {} : { 'content-type': 'application/json' }), ...(key ? { authorization: 'Bearer ' + key } : {}), ...extra };
  const response = await fetch(base.origin + (transport === 'direct' ? '/api/works/v1' : '') + path, {
    method: data === undefined ? 'GET' : 'POST', headers, ...(data === undefined ? {} : { body: JSON.stringify(data) }),
    redirect: 'error', signal: AbortSignal.timeout(20000),
  });
  const text = await response.text();
  let value; try { value = JSON.parse(text); } catch { value = undefined; }
  return { status: response.status, headers: response.headers, value, text };
}
const count = async table => Number((await client.execute('SELECT COUNT(*) n FROM ' + table)).rows[0].n);
const moderate = input => call('/api/community/moderate', input, secrets.EIDOS_ADMIN_TOKEN, { origin });
try {
  if (process.argv.includes('--closed')) {
    progress('deployed kill switch, existing status and self revocation');
    a = JSON.parse(await readFile(continuation, 'utf8'));
    assert.equal((await call('/api/community/agent-registration')).value.enabled, false);
    assert.equal((await call('/api/community/agent-registration', {})).status, 503);
    assert.equal((await call('/api/community/agents', { title: 'Denied after closure', body: 'Controlled QA contribution must not persist after closure.' }, a.credential)).status, 503);
    assert.equal((await call('/api/community/agent-account', undefined, a.credential)).status, 200);
    const race = await Promise.all(['rotate-credential', 'revoke-credential'].map(action => call('/api/community/agent-account', { action }, a.credential)));
    assert.deepEqual(race.map(response => response.status).sort(), [200, 401]);
    if (race[0].status === 200) {
      assert.equal(race[0].value.expiresAt, a.expiresAt);
      a.credential = race[0].value.credential;
      assert.equal((await call('/api/community/agent-account', { action: 'revoke-credential' }, a.credential)).status, 200);
    } else assert.equal(race[1].value.revoked, true);
    assert.equal((await call('/api/community/agent-account', undefined, a.credential)).status, 401);
    assert.equal(Number((await client.execute('SELECT COUNT(*) n FROM eidos_exchange_identities WHERE revoked=0')).rows[0].n), 0);
    receipt.checks.push(step, 'competing rotation/revocation has one winner over hosted HTTP');
  } else {
    progress('isolated schema and no unexpected active identities');
    assert.equal(Number((await client.execute('SELECT COUNT(*) n FROM eidos_exchange_identities WHERE revoked=0')).rows[0].n), 0, 'Do not retry an uncertain run');
    for (const table of ['eidos_email_members', 'eidos_member_sessions', 'eidos_member_keys', 'eidos_orders', 'eidos_pg_projects']) assert.equal(await count(table), 0);
    const availability = await call('/api/community/agent-registration');
    assert.equal(availability.status, 200); assert.equal(availability.value.enabled, true); receipt.checks.push(step);
    progress('two real HTTP registrations and no-store credentials');
    const suffix = randomUUID().slice(0, 8);
    const payload = name => ({ name, description: 'Controlled hosted security QA; not independent acquisition.', discoverySource: 'controlled', rulesVersion: availability.value.rulesVersion, operatorAuthorized: true, acceptPublicContributions: true });
    const register = async name => { const response = relayRegistration ? await (async () => {
      const r = await fetch(registrationBase.origin + '/api/works/v1/api/community/agent-registration', { method: 'POST',
        headers: { 'content-type': 'application/json', 'x-vercel-protection-bypass': secrets.EIDOS_PLATFORM_PREVIEW_BYPASS,
          'x-eidos-platform-token': secrets.EIDOS_PLATFORM_TOKEN, 'x-eidos-site-origin': origin, 'x-eidos-client-ip': relayFixtureIp },
        body: JSON.stringify(payload(name)), redirect: 'error', signal: AbortSignal.timeout(20000) });
      return { status: r.status, headers: r.headers, value: await r.json() };
    })() : await call('/api/community/agent-registration', payload(name));
      assert.equal(response.status, 201); assert.equal(response.headers.get('cache-control'), 'no-store');
      assert.ok(/^ew_exchange_[a-f0-9]{64}$/.test(response.value.credential), 'Expected scoped credential format'); return response.value; };
    a = await register('qa-first-' + suffix); b = await register('qa-other-' + suffix);
    if (transport === 'direct') {
      const third = await call('/api/community/agent-registration', payload('qa-third-' + suffix));
      // Capture and revoke an unexpected successful write before asserting.
      if (third.status === 201 && third.value?.id)
        assert.equal((await moderate({ action: 'revoke-exchange-agent', id: third.value.id })).status, 200);
      assert.equal(third.status, 429);
      receipt.checks.push('same reserved relay-IP fixture rejects a third registration');
    } else {
      // The execution service can rotate its network egress between requests.
      // Do not forge an IP or interpret a fresh IP allowance as a quota failure.
      const quotas = await client.execute("SELECT used FROM eidos_quotas WHERE bucket LIKE 'exchange-registration-ip:%'");
      assert.ok(quotas.rows.every(row => Number(row.used) <= 2));
      receipt.perIpBoundary = 'All persisted IP counters are <=2; same-IP third-request rejection is covered separately by trusted-relay fixtures, not asserted for rotating edge egress';
    }
    const stored = await client.execute({ sql: 'SELECT key_hash FROM eidos_exchange_identities WHERE id=?', args: [a.id] });
    assert.equal(stored.rows[0].key_hash, createHash('sha256').update(a.credential).digest('hex'));
    assert.ok(!JSON.stringify((await client.execute({ sql: 'SELECT * FROM eidos_exchange_identities WHERE id=?', args: [a.id] })).rows).includes(a.credential), 'Raw credential must not persist'); receipt.checks.push(step);
    progress('pending persistence, reciprocal status privacy and scoped privileges');
    const finding = { title: 'Controlled hosted QA finding ' + suffix, body: 'Controlled QA finding and proposed fix. This is not a customer contribution.\n<script>alert(1)</script>' };
    const submitted = await call('/api/community/agents', finding, a.credential);
    assert.equal(submitted.status, 201); assert.equal(submitted.value.state, 'pending'); assert.equal(submitted.value.url, undefined);
    const id = submitted.value.id;
    const independentClient = createClient({ url: databaseUrl.href, authToken: secrets.EIDOS_EXCHANGE_TEST_TURSO_AUTH_TOKEN });
    try { assert.equal((await independentClient.execute({ sql: 'SELECT status FROM eidos_threads WHERE id=?', args: [id] })).rows[0].status, 'pending'); } finally { independentClient.close(); }
    assert.ok((await call('/api/community/agent-account', undefined, a.credential)).value.contributions.some(row => row.id === id));
    assert.equal((await call('/api/community/agent-account', undefined, b.credential)).value.contributions.length, 0);
    assert.equal((await call('/api/community/threads?id=' + id)).status, 404);
    assert.equal((await call('/community/thread/' + id)).status, 404);
    assert.equal((await call('/api/community/agent-account')).status, 401);
    assert.equal((await call('/api/community/agent-account?id=' + b.id, undefined, a.credential)).status, 400);
    assert.equal((await call('/api/community/agent-account', { action: 'revoke-credential', id: b.id }, a.credential)).status, 400);
    assert.equal((await call('/api/community/agent-account', undefined, a.credential, { origin: 'https://evil.example' })).status, 403);
    assert.equal((await call('/api/community/agents', { ...finding, author: 'admin' }, a.credential)).status, 400);
    assert.equal((await call('/api/community/moderate', { action: 'publish', kind: 'thread', id }, a.credential, { origin })).status, 401);
    // Member accounts intentionally remain disabled in this hosted QA runtime.
    assert.equal((await call('/api/playground/projects', undefined, a.credential)).status, 503);
    assert.equal((await call('/api/members/account', undefined, a.credential)).value.member, null); receipt.checks.push(step);
    progress('concurrent pending ceiling through real HTTP');
    const overlap = await Promise.all([1, 2].map(n => call('/api/community/agents', { ...finding, title: finding.title + ' ' + n }, a.credential)));
    assert.deepEqual(overlap.map(r => r.status).sort(), [201, 429]);
    assert.equal(Number((await client.execute({ sql: "SELECT COUNT(*) n FROM eidos_threads WHERE owner_id=? AND status='pending'", args: [a.id] })).rows[0].n), 2); receipt.checks.push(step);
    progress('manual publication, escaped public readback and pending reply');
    assert.equal((await moderate({ action: 'publish', kind: 'thread', id })).status, 200);
    assert.equal((await call('/api/community/threads?id=' + id)).value.thread.id, id);
    const page = await call('/community/thread/' + id); assert.equal(page.status, 200);
    assert.ok(page.text.includes('&lt;script&gt;alert(1)&lt;/script&gt;')); assert.ok(!page.text.includes('<script>alert(1)</script>'));
    assert.ok(page.text.includes('[unclaimed]'));
    const reply = await call('/api/community/agents', { threadId: id, body: 'Controlled QA reply remains pending for manual review.' }, b.credential);
    assert.equal(reply.status, 201); assert.equal(reply.value.state, 'pending');
    assert.ok(!(await call('/api/community/agent-account', undefined, a.credential)).value.contributions.some(row => row.id === reply.value.id));
    assert.equal((await call('/api/community/threads?id=' + id)).value.replies.length, 0);
    assert.equal((await moderate({ action: 'unpublish', kind: 'thread', id })).status, 200); receipt.checks.push(step);
    progress('rotation, original expiry, expired-key denial and operator revocation');
    const rotated = await call('/api/community/agent-account', { action: 'rotate-credential' }, a.credential);
    assert.equal(rotated.status, 200); assert.equal(rotated.value.expiresAt, a.expiresAt);
    assert.equal((await call('/api/community/agent-account', undefined, a.credential)).status, 401);
    a.credential = rotated.value.credential;
    await client.execute({ sql: 'UPDATE eidos_exchange_identities SET expires_at=0 WHERE id=?', args: [b.id] });
    assert.equal((await call('/api/community/agent-account', undefined, b.credential)).status, 401);
    assert.equal((await call('/api/community/agents', finding, b.credential)).status, 401);
    assert.equal((await moderate({ action: 'revoke-exchange-agent', id: b.id })).status, 200); receipt.checks.push(step);
    progress('no member, session, project, payment or mention records created');
    for (const table of ['eidos_email_members', 'eidos_member_sessions', 'eidos_member_keys', 'eidos_orders', 'eidos_pg_projects', 'eidos_mentions']) assert.equal(await count(table), 0);
    receipt.checks.push(step);
    await writeFile(continuation, JSON.stringify({ id: a.id, credential: a.credential, expiresAt: a.expiresAt }), { mode: 0o600 });
    keepCredential = true;
    receipt.remainingGate = 'Deployed kill-switch and self-revocation phase; one private QA credential retained only for that check';
  }
  receipt.status = 'passed';
} catch (error) {
  receipt.failedCheck = step; process.exitCode = 1;
  receipt.failedLine = error?.stack?.match(/verify-exchange-remote-http\.mjs:(\d+):/)?.[1];
  let message = String(error?.message || 'Unknown verification failure');
  for (const value of Object.values(secrets)) if (typeof value === 'string' && value.length > 16)
    message = message.replaceAll(value, '[redacted]');
  receipt.failure = message.replace(/(?:https?|libsql|wss?):\/\/\S+/g, '[endpoint]').slice(0, 500);
  if (typeof error?.actual === 'number' && typeof error?.expected === 'number')
    receipt.assertion = { actual: error.actual, expected: error.expected };
} finally {
  if (!keepCredential) for (const identity of [a, b]) if (identity?.id) {
    try { const revoked = await moderate({ action: 'revoke-exchange-agent', id: identity.id }); if (revoked.status !== 200) receipt.cleanupFailure = true; } catch { receipt.cleanupFailure = true; }
  }
  client.close();
  if (process.env.EXCHANGE_QA_RECEIPT) await writeFile(process.env.EXCHANGE_QA_RECEIPT, JSON.stringify(receipt, null, 2) + '\n');
  console.log(JSON.stringify(receipt));
}
