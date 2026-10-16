import test from 'node:test';
import assert from 'node:assert/strict';
import { checkPlaygroundProvider } from '../lib/works/playground-provider-check';

const token = 'owner-diagnostic-fixture-32-characters', key = 'provider-secret-fixture-do-not-expose';
const env = { VERCEL_ENV: 'preview', EIDOS_PLAYGROUND_PROVIDER_DIAGNOSTICS: 'test-auth', EIDOS_PLATFORM_TOKEN: token, EIDOS_PLAYGROUND_OPENAI_API_KEY: key };
const request = (changes: Record<string, string> = {}) => new Request('https://preview.test/api/playground-provider-check', { headers: { 'x-eidos-platform-token': token, 'x-eidos-site-origin': 'https://eidosworks-playground-20261014.pages.dev', ...changes } });

test('owner binding inspection is private and returns presence only without any provider call', async()=>{
  let calls=0;const transport:typeof fetch=async()=>{calls++;throw Error('No network');};
  const source={...env,EIDOS_ACCOUNTS_ENABLED:'true',RESEND_API_KEY:'re_private_fixture',EIDOS_MAIL_FROM:'private-fixture@example.test',EIDOS_RATE_SECRET:'r'.repeat(32),EIDOS_DATABASE_BINDING_PREFIX:'EIDOS_TEMPLATES_TEST'};
  const response=await checkPlaygroundProvider(request({'x-eidos-bindings-only':'true'}),source,transport),body=await response.text();
  assert.equal(response.status,200);assert.equal(calls,0);
  assert.equal(JSON.parse(body).bindings.mailKeyPresent,true);assert.equal(JSON.parse(body).bindings.accountsEnabled,true);
  assert.equal(body.includes(source.RESEND_API_KEY),false);assert.equal(body.includes(source.EIDOS_MAIL_FROM),false);assert.equal(body.includes(key),false);
  assert.equal((await checkPlaygroundProvider(request({'x-eidos-bindings-only':'true','x-eidos-platform-token':'wrong'}),source,transport)).status,401);
});

test('diagnostic fails closed outside Preview, without explicit flag, token or exact origin', async () => {
  const forbidden: typeof fetch = async () => { throw Error('network must not be reached'); };
  for (const [input, req, status] of [
    [{ ...env, VERCEL_ENV: 'production' }, request(), 404],
    [{ ...env, EIDOS_PLAYGROUND_PROVIDER_DIAGNOSTICS: '' }, request(), 404],
    [env, request({ 'x-eidos-platform-token': 'wrong' }), 401],
    [env, request({ 'x-eidos-site-origin': 'https://attacker.test' }), 403],
    [{ ...env, EIDOS_PLAYGROUND_OPENAI_API_KEY: '' }, request(), 503],
  ] as const) assert.equal((await checkPlaygroundProvider(req, input, forbidden)).status, status);
});

test('diagnostic uses dedicated key only for one fixed metadata request and never returns secrets', async () => {
  let calls = 0;
  const transport: typeof fetch = async (url, init) => {
    calls++; assert.equal(url, 'https://api.openai.com/v1/models'); assert.equal(init?.redirect, 'error');
    assert.equal(new Headers(init?.headers).get('authorization'), 'Bearer ' + key);
    return Response.json({ data: [{ id: 'gpt-6-astra' }, { id: 'gpt-6.1-sol' }, { id: 'unrelated-private-model' }] }, { headers: { 'x-request-id': 'req_provider_auth' } });
  };
  const response = await checkPlaygroundProvider(request(), env, transport), body = await response.text();
  assert.equal(response.status, 200); assert.equal(calls, 1); assert.equal(response.headers.get('cache-control'), 'no-store');
  assert.equal(body.includes(key), false); assert.equal(body.includes(token), false); assert.equal(body.includes('unrelated-private-model'), false);
  assert.deepEqual(JSON.parse(body).models, { 'gpt-6-astra': true, 'gpt-6.1-sol': true });
});

test('provider errors and exceptions cannot leak provider response or credential details', async () => {
  for (const transport of [async () => new Response(key, { status: 401 }), async () => { throw Error(key); }]) {
    const response = await checkPlaygroundProvider(request(), env, transport), body = await response.text();
    assert.equal(response.status, 502); assert.equal(body.includes(key), false);
  }
});
