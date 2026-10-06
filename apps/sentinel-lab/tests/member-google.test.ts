import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createClient } from '@libsql/client';
import { generateKeyPair, exportJWK, createLocalJWKSet, SignJWT } from 'jose';
import { googleVerifier } from '../lib/works/googleIdentity';
import { passwordService } from '../lib/works/passwords';
import { adaptDatabase } from '../lib/works/database';
import { onRequestPost as start, onRequestGet as callback } from '../lib/works/vendor/functions/api/members/google';
import { onRequestPost as credentials } from '../lib/works/vendor/functions/api/members/credentials';
import { onRequestGet as account } from '../lib/works/vendor/functions/api/members/account';
import type { Context, PlatformEnv } from '../lib/works/vendor/functions/_shared/platform/core';

await test('OIDC cryptographic verification rejects wrong issuer/audience/nonce, expiry, unverified email and bad signatures', async () => {
  const { privateKey, publicKey } = await generateKeyPair('RS256');
  const key = await exportJWK(publicKey); key.kid = 'fixture';
  const verifier = googleVerifier('client.test', createLocalJWKSet({ keys: [key] }))!;
  const sign = (claims: Record<string, unknown> = {}) => new SignJWT({ email: 'owner@gmail.com', email_verified: true, nonce: 'nonce.test', ...claims }).setProtectedHeader({ alg: 'RS256', kid: 'fixture' }).setSubject('subject.test').setIssuedAt().setIssuer('https://accounts.google.com').setAudience('client.test').setExpirationTime('5m').sign(privateKey);
  const valid = await sign(); assert.deepEqual(await verifier(valid, 'nonce.test'), { subject: 'subject.test', email: 'owner@gmail.com', authoritativeEmail: true });
  await assert.rejects(verifier(valid, 'wrong'));
  await assert.rejects(verifier(await sign({ email_verified: false }), 'nonce.test'));
  await assert.rejects(verifier(await sign({ azp: 'other-client' }), 'nonce.test'));
  const wrongIssuer = await new SignJWT({ email: 'owner@gmail.com', email_verified: true, nonce: 'nonce.test' }).setProtectedHeader({ alg: 'RS256', kid: 'fixture' }).setSubject('subject.test').setIssuedAt().setIssuer('https://attacker.example').setAudience('client.test').setExpirationTime('5m').sign(privateKey);
  await assert.rejects(verifier(wrongIssuer, 'nonce.test'));
  const expired = await new SignJWT({ email: 'owner@gmail.com', email_verified: true, nonce: 'nonce.test' }).setProtectedHeader({ alg: 'RS256', kid: 'fixture' }).setSubject('subject.test').setIssuedAt(1).setIssuer('https://accounts.google.com').setAudience('client.test').setExpirationTime(2).sign(privateKey);
  await assert.rejects(verifier(expired, 'nonce.test'));
  await assert.rejects(googleVerifier('other-client', createLocalJWKSet({ keys: [key] }))!(valid, 'nonce.test'));
  const chunks = valid.split('.'); chunks[1] = Buffer.from(JSON.stringify({ sub: 'attacker' })).toString('base64url'); await assert.rejects(verifier(chunks.join('.'), 'nonce.test'));
  assert.equal((await verifier(await sign({ email: 'owner@thirdparty.test' }), 'nonce.test')).authoritativeEmail, false);
});

function context(env: PlatformEnv, input?: unknown, cookie = '', query = ''): Context { return { env, request: new Request('http://localhost:8788/api/members/google' + query, { method: input === undefined ? 'GET' : 'POST', headers: { origin: 'http://localhost:8788', 'content-type': 'application/json', cookie }, ...(input === undefined ? {} : { body: JSON.stringify(input) }) }) }; }
async function setup() {
  const client = createClient({ url: ':memory:' });
  for (const name of ['0001_eidos_platform.sql', '0002_members.sql', '0006_member_credentials.sql']) await client.executeMultiple(readFileSync(new URL('../lib/works/vendor/migrations/' + name, import.meta.url), 'utf8'));
  const identity = { subject: 'google-owner', email: 'owner@gmail.com', authoritativeEmail: true };
  const env: PlatformEnv = { EIDOS_DB: adaptDatabase(client), EIDOS_LOCAL_TEST: 'true', EIDOS_PASSWORD_SERVICE: passwordService, PUBLIC_SITE_URL: 'http://localhost:8788', EIDOS_GOOGLE_CLIENT_ID: 'client.test', EIDOS_GOOGLE_CLIENT_SECRET: 'test-secret', EIDOS_GOOGLE_REDIRECT_URI: 'http://localhost:8788/api/members/google', EIDOS_GOOGLE_VERIFY: async () => identity };
  return { client, env, identity };
}
async function begin(env: PlatformEnv, cookie = '', action = 'start', currentPassword?: string) {
  const response = await start(context(env, { action, currentPassword }, cookie));
  assert.equal(response.status, 200, await response.clone().text());
  const url = new URL((await response.json()).authorizeUrl);
  assert.equal(url.origin, 'https://accounts.google.com'); assert.equal(url.searchParams.get('code_challenge_method'), 'S256');
  assert.equal(url.searchParams.get('client_secret'), null);
  return { state: url.searchParams.get('state')!, cookie: response.headers.get('set-cookie')!.split(';')[0] };
}
const sessionCookie = (r: Response) => r.headers.getSetCookie().find(c => c.startsWith('eidos_session_local='))?.split(';')[0] || '';

await test('Google state is bound to browser, single use and PKCE; new member deliberately chooses username', async () => {
  const { client, env } = await setup(); const original = globalThis.fetch; let exchanges = 0;
  globalThis.fetch = async (url, init) => { assert.equal(String(url), 'https://oauth2.googleapis.com/token'); exchanges++; assert.match(String(init?.body), /code_verifier=/); return Response.json({ id_token: 'verified-test-token' }); };
  try {
    const flow = await begin(env);
    assert.match((await callback(context(env, undefined, '', '?state=' + flow.state + '&code=test'))).headers.get('location')!, /oauth=error$/); assert.equal(exchanges, 0);
    const response = await callback(context(env, undefined, flow.cookie, '?state=' + flow.state + '&code=test'));
    assert.match(response.headers.get('location')!, /oauth=choose-username$/); assert.equal(exchanges, 1);
    assert.equal((await client.execute('SELECT COUNT(*) n FROM eidos_email_members')).rows[0].n, 0);
    const ticket = response.headers.getSetCookie().find(c => c.startsWith('eidos_onboard_local='))!.split(';')[0];
    const completed = await start(context(env, { action: 'complete-signup', username: 'google_owner' }, ticket));
    assert.equal(completed.status, 200, await completed.clone().text()); assert.ok(sessionCookie(completed));
    assert.equal((await start(context(env, { action: 'complete-signup', username: 'duplicate_owner' }, ticket))).status, 401);
    assert.match((await callback(context(env, undefined, flow.cookie, '?state=' + flow.state + '&code=test'))).headers.get('location')!, /oauth=error$/); assert.equal(exchanges, 1);
    assert.equal((await client.execute('SELECT COUNT(*) n FROM eidos_email_members')).rows[0].n, 1);
  } finally { globalThis.fetch = original; client.close(); }
});

await test('Google links authoritative email to the existing verified member and preserves username; third-party collision requires deliberate linking', async () => {
  const { client, env, identity } = await setup(); const original = globalThis.fetch;
  globalThis.fetch = async () => Response.json({ id_token: 'verified-test-token' });
  try {
    const pass = 'A long existing-member passphrase!';
    const pending = await credentials(context(env, { action: 'signup', email: identity.email, username: 'existing_member', password: pass }));
    const token = new URLSearchParams(new URL((await pending.json()).localVerificationUrl).hash.slice(1)).get('token');
    const signedIn = await credentials(context(env, { action: 'verify-signup', token })); assert.equal(signedIn.status, 200);
    const before = (await client.execute('SELECT id,username FROM eidos_email_members')).rows[0];
    let flow = await begin(env);
    const response = await callback(context(env, undefined, flow.cookie, '?state=' + flow.state + '&code=test'));
    assert.match(response.headers.get('location')!, /oauth=success$/);
    const member = (await (await account(context(env, undefined, sessionCookie(response)))).json()).member; assert.equal(member.username, before.username);
    assert.equal((await client.execute('SELECT member_id FROM eidos_member_google')).rows[0].member_id, before.id);
    assert.equal((await client.execute('SELECT COUNT(*) n FROM eidos_email_members')).rows[0].n, 1);
    identity.subject = 'another-subject'; identity.authoritativeEmail = false;
    flow = await begin(env);
    assert.match((await callback(context(env, undefined, flow.cookie, '?state=' + flow.state + '&code=test'))).headers.get('location')!, /oauth=link-required$/);
    flow = await begin(env, sessionCookie(response), 'link', pass);
    assert.match((await callback(context(env, undefined, flow.cookie + '; ' + sessionCookie(response), '?state=' + flow.state + '&code=test'))).headers.get('location')!, /oauth=collision$/);
    assert.equal((await client.execute('SELECT subject FROM eidos_member_google')).rows[0].subject, 'google-owner');
    identity.email = 'new@thirdparty.test';
    flow = await begin(env);
    assert.match((await callback(context(env, undefined, flow.cookie, '?state=' + flow.state + '&code=test'))).headers.get('location')!, /oauth=signup-required$/);
    assert.equal((await client.execute('SELECT COUNT(*) n FROM eidos_member_google_signup')).rows[0].n, 0);
    assert.equal((await client.execute('SELECT COUNT(*) n FROM eidos_email_members')).rows[0].n, 1);
    assert.equal((await client.execute("SELECT COUNT(*) n FROM eidos_member_oauth_flows WHERE consumed=2 AND (nonce<>'' OR verifier<>'')")).rows[0].n, 0);
  } finally { globalThis.fetch = original; client.close(); }
});
