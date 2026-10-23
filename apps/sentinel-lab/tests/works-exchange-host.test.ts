import test from 'node:test';
import assert from 'node:assert/strict';
import { createClient } from '@libsql/client';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { adaptDatabase } from '../lib/works/database';
import { dispatchWorks, platformEnvironment } from '../lib/works/dispatch';
import type { PlatformEnv } from '../lib/works/vendor/functions/_shared/platform/core';

const secret = 'controlled-host-relay-secret-at-least-32-chars';
const admin = 'controlled-host-admin-secret-at-least-32-chars';
const source = { EIDOS_PLATFORM_TOKEN: secret, PUBLIC_SITE_URL: 'https://eidos-works.com' };
test('The scoped host export matches every vendored byte and its flag stays an explicit opt-in', () => {
  const manifest = JSON.parse(readFileSync('lib/works/host-contract.json', 'utf8'));
  assert.equal(manifest.files.length, 5);
  for (const item of manifest.files) assert.equal(createHash('sha256').update(readFileSync('lib/works/vendor/' + item.path)).digest('hex'), item.sha256, item.path);
  assert.equal(platformEnvironment({ EIDOS_EXCHANGE_HOST_ENABLED: 'true' }).EIDOS_EXCHANGE_HOST_ENABLED, 'true');
  assert.equal(platformEnvironment({}).EIDOS_EXCHANGE_HOST_ENABLED, undefined);
});
test('The actual authenticated relay and libSQL enforce host enrollment, private progress, review races, owner authorization and revocation', async () => {
  const client = createClient({ url: ':memory:' });
  for (const file of ['0001_eidos_platform.sql', '0002_members.sql', '0010_agent_exchange_self_registration.sql', '0011_exchange_host.sql']) await client.executeMultiple(readFileSync('lib/works/vendor/migrations/' + file, 'utf8'));
  const env: PlatformEnv = { EIDOS_DB: adaptDatabase(client), EIDOS_RUNTIME: 'sentinel', EIDOS_EXCHANGE_HOST_ENABLED: 'true', EIDOS_EXCHANGE_SELF_REGISTRATION_ENABLED: 'true', EIDOS_RATE_SECRET: secret, EIDOS_ADMIN_TOKEN: admin };
  function call(path: string, data?: unknown, key?: string, headers?: Record<string, string>) {
    return dispatchWorks(new Request('https://controlled-host.vercel.app/api/works/v1' + path, { method: data ? 'POST' : 'GET',
      headers: { 'x-eidos-platform-token': secret, 'x-eidos-site-origin': source.PUBLIC_SITE_URL, 'x-eidos-client-ip': '192.0.2.1', 'content-type': 'application/json', ...(key ? { authorization: 'Bearer ' + key } : {}), ...headers }, body: data ? JSON.stringify(data) : undefined }), source, env);
  }
  const enrollment = (name: string) => ({ action: 'enroll', requestId: crypto.randomUUID(), registration: { name, description: 'Controlled relay host QA, not an outside operator.', rulesVersion: '2026-10-22', operatorAuthorized: true, acceptPublicContributions: true } });
  try {
    assert.equal((await call('/api/community/host', undefined, undefined, { 'x-eidos-platform-token': 'bad' })).status, 401);
    assert.equal((await call('/api/community/host', undefined, undefined, { 'x-eidos-site-origin': 'https://evil.example' })).status, 403);
    const aResponse = await call('/api/community/host', enrollment('qa-relay-host-a')); assert.equal(aResponse.status, 201, await aResponse.clone().text()); const a = (await aResponse.json()).registration;
    const bResponse = await call('/api/community/host', enrollment('qa-relay-host-b'), undefined, { 'x-eidos-client-ip': '192.0.2.2' }); assert.equal(bResponse.status, 201); const b = (await bResponse.json()).registration;
    assert.equal((await call('/api/community/host', { action: 'guide', requestId: crypto.randomUUID(), topic: 'tasks' }, a.credential)).status, 200);
    const other = await (await call('/api/community/host', undefined, b.credential)).json(); assert.ok(other.progress.every((e: { topic: string }) => e.topic !== 'tasks'));
    const review = { action: 'request-review', requestId: crypto.randomUUID(), kind: 'exception', note: 'Controlled non-sensitive owner review of a participation question.' };
    const race = await Promise.all([1, 2, 3, 4].map(() => call('/api/community/host', { ...review, requestId: crypto.randomUUID() }, a.credential)));
    assert.equal(race.filter(r => r.status === 202).length, 1);
    assert.equal((await call('/api/community/host-admin', undefined, a.credential)).status, 401);
    const queue = await (await call('/api/community/host-admin', undefined, admin)).json(); assert.equal(queue.reviews.length, 1); assert.equal(queue.host.modelSpendLimitUsd, 0);
    assert.equal((await call('/api/community/host-admin', { id: queue.reviews[0].id, outcome: 'closed' }, admin, { origin: source.PUBLIC_SITE_URL })).status, 200);
    assert.equal((await (await call('/api/community/agent-account', undefined, a.credential)).json()).identity.verification, 'unclaimed');
    await call('/api/community/moderate', { action: 'revoke-exchange-agent', id: a.id }, admin, { origin: source.PUBLIC_SITE_URL });
    assert.equal((await call('/api/community/host', undefined, a.credential)).status, 401);
    env.EIDOS_EXCHANGE_HOST_ENABLED = undefined;
    assert.equal((await call('/api/community/host', { action: 'guide', requestId: crypto.randomUUID(), topic: 'tasks' }, b.credential)).status, 503);
    assert.equal((await call('/api/community/host', undefined, b.credential)).status, 200);
  } finally { client.close(); }
});
