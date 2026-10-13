import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createClient } from '@libsql/client';
import { adaptDatabase, databaseConfiguration } from '../lib/works/database';
import { dispatchWorks, platformEnvironment } from '../lib/works/dispatch';
import { runStudioDiscussion, studioAgents } from '../lib/works/vendor/functions/_shared/platform/studioAgents';

// Run only in the explicitly selected, previously verified isolated preview.
// No migration, customer row read, credential output, or production write.
const receipt: Record<string, unknown> = {
  marker: 'EIDOS_COMMUNITY_HOSTED_ACCEPTANCE', time: new Date().toISOString(),
  source: process.env.VERCEL_GIT_COMMIT_SHA, environment: process.env.VERCEL_ENV,
  checks: [], status: 'failed',
};
let client: ReturnType<typeof createClient> | undefined;
try {
  assert.equal(process.env.VERCEL_ENV, 'preview');
  assert.equal(process.env.VERCEL_GIT_COMMIT_REF, 'codex/community-studio-backend-20261001');
  assert.equal(process.env.EIDOS_DATABASE_BINDING_PREFIX, 'EIDOS_PG_INTEGRATION');
  const selected = databaseConfiguration(process.env);
  assert.ok(selected.url && selected.authToken, 'Dedicated preview bindings required');
  const url = new URL(selected.url);
  const identity = createHash('sha256').update(url.hostname + url.pathname).digest('hex');
  assert.equal(identity, '4882b7d6be40a19a97d5dabce311089dc72a984e5a2b3b3f64081216926bb353', 'Previously verified isolated database required');
  receipt.databaseIdentityHash = identity;
  client = createClient({ url: selected.url, authToken: selected.authToken });
  for (const [table, columns] of Object.entries({
    eidos_agents: ['id', 'name', 'key_hash', 'profile_url', 'revoked'],
    eidos_threads: ['id', 'owner_id', 'author_type', 'status', 'published_at'],
    eidos_replies: ['id', 'thread_id', 'owner_id', 'status'],
  })) {
    const result = await client.execute(`PRAGMA table_info(${table})`);
    for (const column of columns) assert.ok(result.rows.some(row => row.name === column), `${table}.${column} required`);
  }
  const env = { ...platformEnvironment(process.env), EIDOS_DB: adaptDatabase(client),
    EIDOS_COMMUNITY_STUDIO_ENABLED: 'true', EIDOS_COMMUNITY_STUDIO_START_DATE: '2026-10-07' };
  assert.ok(process.env.EIDOS_PLATFORM_TOKEN && process.env.EIDOS_PLATFORM_TOKEN.length >= 32, 'Preview relay credential required');
  assert.ok(env.EIDOS_ADMIN_TOKEN && env.EIDOS_ADMIN_TOKEN.length >= 32, 'Preview moderation credential required');
  const source: Record<string, string | undefined> = { ...process.env, PUBLIC_SITE_URL: 'https://eidosworks-test-20260923.pages.dev' };
  const req = (path: string, method = 'GET', body?: unknown, token?: string) => new Request('https://preview.vercel.app/api/works/v1' + path, {
    method, headers: { 'x-eidos-platform-token': source.EIDOS_PLATFORM_TOKEN!,
      'x-eidos-site-origin': source.PUBLIC_SITE_URL!, origin: source.PUBLIC_SITE_URL!,
      'content-type': 'application/json', ...(token ? { authorization: 'Bearer ' + token } : {}) },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  // Simulated schedule dates are confined to this build process and isolated DB;
  // the deployed API has no clock override or local-test bypass.
  const now = new Date('2026-10-07T13:17:00Z');
  const results = await Promise.all(Array.from({ length: 20 }, () => runStudioDiscussion(env, now)));
  assert.ok(results.reduce((n, r) => n + r.published, 0) <= 1);
  const rosterResponse = await dispatchWorks(req('/api/community/agents'), source, env);
  assert.equal(rosterResponse.status, 200);
  const roster = await rosterResponse.json();
  assert.equal(roster.agents.filter((a: { studio?: boolean }) => a.studio).length, 3);
  assert.ok(roster.agents.every((a: Record<string, unknown>) => !('key_hash' in a)));
  const id = '9ba4b87a-4fa8-48fd-a2a4-27c15d000001';
  const topic = await client.execute({ sql: 'SELECT status,author_type,body FROM eidos_threads WHERE id=?', args: [id] });
  assert.equal(topic.rows.length, 1);
  assert.equal(topic.rows[0].status, 'published');
  assert.equal(topic.rows[0].author_type, 'agent');
  assert.match(String(topic.rows[0].body), /AI agent · operated by Eidos Works/);
  // Preserve existing preview data. Add one clearly identified QA reply and
  // approve it through the unchanged authenticated dispatch/moderation handler.
  const replyId = crypto.randomUUID();
  await client.execute({ sql: "INSERT INTO eidos_replies(id,thread_id,body,author,author_type,status,created_at) VALUES(?,?,?,'Community preview QA','guest','pending',?)",
    args: [replyId, id, 'Isolated QA reply: this checks stored discussion replies and moderation; it is not a customer contribution.', new Date().toISOString()] });
  const denied = await dispatchWorks(req('/api/community/moderate', 'POST', { action: 'publish', kind: 'reply', id: replyId }), source, env);
  assert.equal(denied.status, 401);
  const approved = await dispatchWorks(req('/api/community/moderate', 'POST', { action: 'publish', kind: 'reply', id: replyId }, env.EIDOS_ADMIN_TOKEN), source, env);
  assert.equal(approved.status, 200);
  assert.equal((await client.execute({ sql: 'SELECT status FROM eidos_replies WHERE id=?', args: [replyId] })).rows[0].status, 'published');
  const repeated = await runStudioDiscussion(env, now);
  assert.equal(repeated.published, 0);
  const feed = await dispatchWorks(req('/community/feed'), source, env);
  assert.equal(feed.status, 200);
  assert.ok((await feed.json()).items.some((item: { id: string }) => item.id.endsWith(id)));
  receipt.checks = ['known isolated DB identity', 'existing community schema', 'twenty overlapping scheduler calls', 'three real registered AI identities', 'disclosed stored prompt', 'unauthorized moderation denied', 'stored QA reply moderation', 'repeat does not publish', 'actual JSON feed'];
  receipt.registered = studioAgents.map(a => a.name);
  receipt.topicId = id;
  receipt.runtimeClockOverride = false;
  receipt.localTestBypass = false;
  receipt.newPublished = results.reduce((n, r) => n + r.published, 0);
  receipt.status = 'passed';
} catch (error) {
  receipt.reason = error instanceof assert.AssertionError ? error.message : 'Hosted verification failed; provider payloads and credentials omitted';
  process.exitCode = 1;
} finally {
  client?.close();
  console.log(JSON.stringify(receipt));
}
