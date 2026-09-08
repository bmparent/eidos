import test from 'node:test';
import assert from 'node:assert/strict';
import { createClient } from '@libsql/client';
import { readFileSync } from 'node:fs';
import { adaptDatabase } from '../lib/works/database';
import { dispatchWorks } from '../lib/works/dispatch';
import { hash, reserve, type PlatformEnv } from '../lib/works/vendor/functions/_shared/platform/core';

const secret = 'local-test-relay-key-at-least-32-characters';
const source = { EIDOS_PLATFORM_TOKEN: secret, PUBLIC_SITE_URL: 'https://eidos-works.com' };
function request(path: string, input?: unknown, extra: Record<string,string> = {}) {
  return new Request('https://eidos-sentinel-lab.vercel.app/api/works/v1' + path, {
    method: input === undefined ? 'GET' : 'POST',
    headers: { 'x-eidos-platform-token': secret, 'x-eidos-site-origin': source.PUBLIC_SITE_URL,
      origin: source.PUBLIC_SITE_URL, 'content-type': 'application/json', ...extra },
    body: input === undefined ? undefined : JSON.stringify(input),
  });
}
async function setup() {
  const client = createClient({ url: ':memory:' });
  await client.executeMultiple(readFileSync('lib/works/vendor/migrations/0001_eidos_platform.sql', 'utf8'));
  await client.executeMultiple(readFileSync('lib/works/vendor/migrations/0002_members.sql', 'utf8'));
  return { client, db: adaptDatabase(client) };
}
await test('Relay authentication, exact origins, routes and method gates', async () => {
  assert.equal((await dispatchWorks(request('/health'), {})).status, 200);
  assert.equal((await dispatchWorks(request('/api/assistant', {question:'test'}, {'x-eidos-platform-token':'wrong'}), source)).status, 401);
  assert.equal((await dispatchWorks(request('/api/assistant', {question:'test'}, {'x-eidos-site-origin':'https://other.pages.dev'}), source)).status, 403);
  assert.equal((await dispatchWorks(request('/api/experiments'), source)).status, 404);
  assert.equal((await dispatchWorks(request('/api/assistant'), source)).status, 405);
  assert.equal((await dispatchWorks(request('/api/assistant', {question:'test'}, {origin:'https://attacker.example'}), source, {})).status, 403);
});
await test('Member sign-in survives the authenticated relay and libSQL consumes links exactly once', async () => {
  const {client,db}=await setup();
  const env:PlatformEnv={EIDOS_RUNTIME:'sentinel',EIDOS_DB:db,EIDOS_ACCOUNTS_ENABLED:'true',RESEND_API_KEY:'test-only',EIDOS_MAIL_FROM:'Eidos <papers@example.test>',TURNSTILE_SITE_KEY:'test',TURNSTILE_SECRET_KEY:'test',EIDOS_RATE_SECRET:'test-only-rate-secret-longer-than-32-characters',PUBLIC_SITE_URL:source.PUBLIC_SITE_URL};
  const token='a'.repeat(64);
  try {
    await db.prepare('INSERT INTO eidos_signin_links(token_hash,email,username,kind,newsletter,expires) VALUES(?,?,?,?,?,?)').bind(await hash(token),'relay@example.test','relay_member','person',1,Math.floor(Date.now()/1000)+900).run();
    const verified=await dispatchWorks(request('/api/members/auth',{action:'verify',token}),source,env);
    assert.equal(verified.status,200);
    const cookie=verified.headers.get('set-cookie')!;assert.match(cookie,/^__Host-eidos_session=/);assert.match(cookie,/; Secure$/);
    assert.equal((await dispatchWorks(request('/api/members/auth',{action:'verify',token}),source,env)).status,400);
    const own=await (await dispatchWorks(request('/api/members/account',undefined,{cookie}),source,env)).json();assert.equal(own.member.username,'relay_member');
    assert.equal((await dispatchWorks(request('/api/members/account',{action:'bookmark',slug:'useful-paper',saved:true},{cookie}),source,env)).status,200);
    const saved=await (await dispatchWorks(request('/api/members/account',undefined,{cookie}),source,env)).json();assert.equal(saved.saved[0].slug,'useful-paper');
    const anonymous=await (await dispatchWorks(request('/api/members/account'),source,env)).json();assert.equal(anonymous.member,null);
    const logout=await dispatchWorks(request('/api/members/auth',{action:'logout'},{cookie}),source,env);assert.match(logout.headers.get('set-cookie')!,/Max-Age=0; Secure$/);
    assert.equal((await (await dispatchWorks(request('/api/members/account',undefined,{cookie}),source,env)).json()).member,null);
    assert.equal((await dispatchWorks(request('/api/members/unsubscribe'),source,env)).status,405);
  }finally{client.close();}
});
await test('Published answers work with no database, model key or outbound fetch', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => { throw Error('Unexpected external request'); };
  try {
    const response = await dispatchWorks(request('/api/assistant', {question:'What can Eidos build?'}), source, {});
    assert.equal(response.status, 200);
    assert.equal((await response.json()).mode, 'sources');
    const config = await (await dispatchWorks(request('/api/public-config'), source, {})).json();
    assert.equal(config.aiReady, false);
    assert.equal(config.shopReady, false);
  } finally { globalThis.fetch = original; }
});
await test('Real libSQL adapter reserves atomically and rolls back failed batches', async () => {
  const { client, db } = await setup();
  try {
    const results = await Promise.all(Array.from({length:30}, () => reserve(db, 'test', 100, 750)));
    assert.equal(results.filter(Boolean).length, 7);
    assert.equal((await db.prepare('SELECT used FROM eidos_quotas WHERE bucket=?').bind('test').first<{used:number}>())?.used, 700);
    await assert.rejects(db.batch([
      db.prepare('INSERT INTO eidos_stripe_events VALUES(?,?)').bind('event-test','now'),
      db.prepare('INSERT INTO does_not_exist VALUES(?)').bind('fail'),
    ]));
    assert.equal(await db.prepare('SELECT * FROM eidos_stripe_events WHERE id=?').bind('event-test').first(), null);
  } finally { client.close(); }
});
await test('Sentinel quota and idempotency prevent repeated model calls', async () => {
  const { client, db } = await setup();
  const env: PlatformEnv = { EIDOS_RUNTIME:'sentinel', EIDOS_DB:db, EIDOS_AI_ENABLED:'true',
    OPENAI_API_KEY:'test-provider-key', EIDOS_ASSISTANT_MODEL:'test-model',
    EIDOS_AI_DAILY_TOKENS:'20000', EIDOS_RATE_SECRET:'test-rate-secret-at-least-32-characters' };
  const original = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async (_url, init) => {
    calls++;
    const payload = JSON.parse(String(init?.body));
    assert.equal(payload.max_output_tokens,320); assert.equal(payload.store,false);
    assert.equal(payload.tools, undefined);
    return Response.json({output:[{type:'message',content:[{type:'output_text',text:'A brief studio answer.'}]}]});
  };
  try {
    const input = { question:'What storefronts do you build?', enhanced:true, requestId:'test-idempotency-12345' };
    const first = await (await dispatchWorks(request('/api/assistant',input),source,env)).json();
    assert.equal(first.mode,'ai');
    const duplicate = await (await dispatchWorks(request('/api/assistant',input),source,env)).json();
    assert.deepEqual(duplicate,first); assert.equal(calls,1);
    const disabled = await (await dispatchWorks(request('/api/assistant',{...input,requestId:'different-request-1234'}),source,{...env,EIDOS_AI_DAILY_TOKENS:'0'})).json();
    assert.equal(disabled.mode,'sources'); assert.equal(calls,1);
    const cloudflare = await (await dispatchWorks(request('/api/assistant',{...input,requestId:'different-request-9876'}),source,{...env,EIDOS_RUNTIME:undefined})).json();
    assert.equal(cloudflare.mode,'sources'); assert.equal(calls,1);
  } finally { globalThis.fetch=original; client.close(); }
});
