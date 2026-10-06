import test from 'node:test';
import assert from 'node:assert/strict';
import { createClient } from '@libsql/client';
import { readFileSync } from 'node:fs';
import { adaptDatabase, databaseConfiguration, platformDatabase } from '../lib/works/database';
import { dispatchWorks, platformEnvironment } from '../lib/works/dispatch';
import { runStudioDiscussion, studioAgents } from '../lib/works/vendor/functions/_shared/platform/studioAgents';
import type { PlatformEnv } from '../lib/works/vendor/functions/_shared/platform/core';

const source = { EIDOS_PLATFORM_TOKEN: 'test-only-relay-token-at-least-32-characters', PUBLIC_SITE_URL: 'https://eidos-works.com' };
test('Missing isolated bindings cannot fall back to the production database', () => {
  const env = { EIDOS_DATABASE_URL: 'libsql://production.example', EIDOS_DATABASE_AUTH_TOKEN: 'production-test-token', EIDOS_DATABASE_BINDING_PREFIX: 'EIDOS_PG_INTEGRATION' };
  assert.deepEqual(databaseConfiguration(env), { url: undefined, authToken: undefined });
  assert.equal(platformDatabase(env), undefined);
  assert.deepEqual(databaseConfiguration({ ...env, EIDOS_PG_INTEGRATION_TURSO_DATABASE_URL: 'libsql://isolated.example', EIDOS_PG_INTEGRATION_TURSO_AUTH_TOKEN: 'isolated-test-token' }), { url: 'libsql://isolated.example', authToken: 'isolated-test-token' });
  assert.throws(() => databaseConfiguration({ ...env, EIDOS_DATABASE_BINDING_PREFIX: '../unsafe' }));
});
function request(path: string, method = 'GET', extra: Record<string, string> = {}) {
  return new Request('https://eidos-sentinel-lab.vercel.app/api/works/v1' + path, {
    method, headers: {
      'x-eidos-platform-token': source.EIDOS_PLATFORM_TOKEN, 'x-eidos-site-origin': source.PUBLIC_SITE_URL,
      origin: source.PUBLIC_SITE_URL, 'content-type': 'application/json', ...extra,
    }, ...(method === 'POST' ? { body: '{}' } : {}),
  });
}
async function fixture() {
  const client = createClient({ url: ':memory:' });
  for (const name of ['0001_eidos_platform.sql', '0002_members.sql'])
    await client.executeMultiple(readFileSync('lib/works/vendor/migrations/' + name, 'utf8'));
  const env: PlatformEnv = {
    EIDOS_RUNTIME: 'sentinel', EIDOS_DB: adaptDatabase(client), PUBLIC_SITE_URL: source.PUBLIC_SITE_URL,
    EIDOS_COMMUNITY_STUDIO_ENABLED: 'true', EIDOS_COMMUNITY_STUDIO_START_DATE: '2026-10-02',
    EIDOS_MAINTENANCE_TOKEN: 'test-only-maintenance-token-at-least-32-characters',
    EIDOS_ADMIN_TOKEN: 'test-only-admin-token-at-least-32-characters',
  };
  return { client, env };
}
test('Backend forwards both explicit activation settings and keeps publication disabled by default', () => {
  const disabled = platformEnvironment({});
  assert.equal(disabled.EIDOS_COMMUNITY_STUDIO_ENABLED, undefined);
  const configured = platformEnvironment({ EIDOS_COMMUNITY_STUDIO_ENABLED: 'true', EIDOS_COMMUNITY_STUDIO_START_DATE: '2026-10-02' });
  assert.equal(configured.EIDOS_COMMUNITY_STUDIO_ENABLED, 'true');
  assert.equal(configured.EIDOS_COMMUNITY_STUDIO_START_DATE, '2026-10-02');
});
test('Production-shaped maintenance requires both relay and operator credentials before creating studio agents', async () => {
  const { client, env } = await fixture();
  try {
    assert.equal((await dispatchWorks(request('/api/community/maintenance', 'POST', { 'x-eidos-platform-token': 'wrong' }), source, env)).status, 401);
    assert.equal((await dispatchWorks(request('/api/community/maintenance', 'POST'), source, env)).status, 401);
    const disabled = await dispatchWorks(request('/api/community/maintenance', 'POST', { authorization: 'Bearer ' + env.EIDOS_MAINTENANCE_TOKEN }), source, { ...env, EIDOS_COMMUNITY_STUDIO_ENABLED: 'false' });
    assert.equal(disabled.status, 200, await disabled.clone().text());
    assert.equal((await disabled.json()).studio.state, 'disabled');
    assert.equal((await client.execute('SELECT count(*) n FROM eidos_agents')).rows[0].n, 0);
  } finally { client.close(); }
});
test('Real libSQL overlapping studio runs publish exactly one labeled feed entry without external calls', async () => {
  const { client, env } = await fixture();
  const original = globalThis.fetch;
  globalThis.fetch = async () => { throw Error('Unexpected model, paid provider, or network request'); };
  try {
    const results = await Promise.all(Array.from({ length: 20 }, () => runStudioDiscussion(env, new Date('2026-10-02T13:17:00Z'))));
    assert.equal(results.reduce((count, result) => count + result.published, 0), 1);
    assert.equal((await client.execute('SELECT count(*) n FROM eidos_threads')).rows[0].n, 1);
    assert.equal((await client.execute('SELECT count(*) n FROM eidos_agents')).rows[0].n, 3);
    const roster = await (await dispatchWorks(request('/api/community/agents'), source, env)).json();
    assert.ok(roster.agents.every((agent: Record<string, unknown>) => agent.kind === 'agent' && agent.studio && !('key_hash' in agent)));
    const feed = await (await dispatchWorks(request('/community/feed'), source, env)).json();
    assert.match(feed.items[0].content_text, /AI agent · operated by Eidos Works/);
    assert.ok(feed.items[0].tags.includes('agent'));
    assert.equal((await client.execute('SELECT count(*) n FROM eidos_replies')).rows[0].n, 0);
  } finally { globalThis.fetch = original; client.close(); }
});
test('Studio registration preserves a revoked identity on the real libSQL adapter', async () => {
  const { client, env } = await fixture();
  try {
    await runStudioDiscussion(env, new Date('2026-10-02T12:59:00Z'));
    await env.EIDOS_DB!.prepare('UPDATE eidos_agents SET revoked=1 WHERE id=?').bind(studioAgents[1].id).run();
    assert.equal((await runStudioDiscussion(env, new Date('2026-10-02T13:17:00Z'))).state, 'agent-paused');
    assert.equal((await client.execute('SELECT count(*) n FROM eidos_threads')).rows[0].n, 0);
  } finally { client.close(); }
});
