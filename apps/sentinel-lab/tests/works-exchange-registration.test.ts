import test from 'node:test';
import assert from 'node:assert/strict';
import { createClient } from '@libsql/client';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { adaptDatabase } from '../lib/works/database';
import { dispatchWorks, platformEnvironment } from '../lib/works/dispatch';
import { type PlatformEnv } from '../lib/works/vendor/functions/_shared/platform/core';

const secret = 'local-exchange-relay-secret-at-least-32-characters';
const source = { EIDOS_PLATFORM_TOKEN: secret, PUBLIC_SITE_URL: 'https://eidos-works.com' };
function request(path: string, input?: unknown, key?: string, headers: Record<string, string> = {}) {
  return new Request('https://test-sentinel.vercel.app/api/works/v1' + path, { method: input === undefined ? 'GET' : 'POST',
    headers: { 'x-eidos-platform-token': secret, 'x-eidos-site-origin': source.PUBLIC_SITE_URL,
      'x-eidos-client-ip': '192.0.2.1', 'content-type': 'application/json', ...(key ? { authorization: 'Bearer ' + key } : {}), ...headers },
    body: input === undefined ? undefined : JSON.stringify(input) });
}
const payload = (name: string) => ({ name, description: 'Controlled relay QA, not an independent outside operator.', rulesVersion: '2026-10-22', operatorAuthorized: true, acceptPublicContributions: true });
test('The scoped seven-file Agent Exchange inventory matches the actual vendored bytes', () => {
  const manifest = JSON.parse(readFileSync('lib/works/exchange-contract.json', 'utf8'));
  assert.equal(manifest.files.length, 7);
  for (const item of manifest.files) assert.equal(createHash('sha256').update(readFileSync('lib/works/vendor/' + item.path)).digest('hex'), item.sha256, item.path);
  assert.equal(platformEnvironment({ EIDOS_EXCHANGE_SELF_REGISTRATION_ENABLED: 'true' }).EIDOS_EXCHANGE_SELF_REGISTRATION_ENABLED, 'true');
  assert.equal(platformEnvironment({}).EIDOS_EXCHANGE_SELF_REGISTRATION_ENABLED, undefined);
});
test('Authenticated relay and real libSQL preserve private pending submissions, two identities, scope, rotation and review caps', async () => {
  const client = createClient({ url: ':memory:' });
  for (const name of ['0001_eidos_platform.sql', '0002_members.sql', '0010_agent_exchange_self_registration.sql'])
    await client.executeMultiple(readFileSync('lib/works/vendor/migrations/' + name, 'utf8'));
  const env: PlatformEnv = { EIDOS_DB: adaptDatabase(client), EIDOS_RUNTIME: 'sentinel', EIDOS_EXCHANGE_SELF_REGISTRATION_ENABLED: 'true',
    EIDOS_RATE_SECRET: 'local-test-rate-key-at-least-32-characters', PUBLIC_SITE_URL: source.PUBLIC_SITE_URL };
  const call = (path: string, input?: unknown, key?: string, headers?: Record<string, string>) => dispatchWorks(request(path, input, key, headers), source, env);
  try {
    assert.equal((await call('/api/community/agent-registration', payload('qa-denied'), undefined, { 'x-eidos-platform-token': 'wrong' })).status, 401);
    assert.equal((await call('/api/community/agent-registration', payload('qa-origin'), undefined, { origin: 'https://attacker.example' })).status, 403);
    assert.equal((await call('/api/community/agent-registration', payload('qa-origin'), undefined, { 'sec-fetch-site': 'cross-site' })).status, 403);
    const aResponse = await call('/api/community/agent-registration', payload('qa-relay-one'));
    assert.equal(aResponse.status, 201); const a = await aResponse.json();
    const bResponse = await call('/api/community/agent-registration', payload('qa-relay-two'), undefined, { 'x-eidos-client-ip': '192.0.2.2' });
    assert.equal(bResponse.status, 201); const b = await bResponse.json();
    const finding = { title: 'Controlled libSQL concurrency finding', body: 'Activity: qa\nA reproducible fixture finding and minimal fix. This is controlled test traffic.' };
    const race = await Promise.all([1, 2, 3, 4].map(() => call('/api/community/agents', finding, a.credential)));
    assert.equal(race.filter(r => r.status === 201).length, 2);
    assert.ok(race.every(r => [201, 429].includes(r.status)));
    const mine = await (await call('/api/community/agent-account', undefined, a.credential)).json();
    assert.equal(mine.contributions.length, 2); assert.ok(mine.contributions.every((x: { status: string; url?: string }) => x.status === 'pending' && !x.url));
    const other = await (await call('/api/community/agent-account', undefined, b.credential)).json(); assert.equal(other.contributions.length, 0);
    assert.equal((await (await call('/api/community/threads')).json()).threads.length, 0);
    assert.equal((await call('/api/community/moderate', { action: 'publish', kind: 'thread', id: mine.contributions[0].id }, a.credential, { origin: source.PUBLIC_SITE_URL })).status, 503);
    const rotation = await Promise.all([1, 2].map(() => call('/api/community/agent-account', { action: 'rotate-credential' }, b.credential)));
    assert.deepEqual(rotation.map(r => r.status).sort(), [200, 401]);
    const rotated = await rotation.find(r => r.status === 200)!.json();
    assert.equal((await call('/api/community/agent-account', undefined, b.credential)).status, 401);
    assert.equal((await call('/api/community/agent-account', { action: 'revoke-credential' }, rotated.credential)).status, 200);
    assert.equal((await call('/api/community/agents', finding, rotated.credential)).status, 401);
    env.EIDOS_EXCHANGE_SELF_REGISTRATION_ENABLED = undefined;
    assert.equal((await (await call('/api/community/agent-registration')).json()).enabled, false);
    assert.equal((await call('/api/community/agents', finding, a.credential)).status, 503);
  } finally { client.close(); }
});
