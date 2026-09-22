import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createClient } from '@libsql/client';
import { adaptDatabase, databaseConfiguration, platformDatabase } from '../lib/works/database';
import { passwordService } from '../lib/works/passwords';
import { validatePlaygroundImage } from '../lib/works/playgroundImage';
import sharp from 'sharp';
import { hash, type Context, type PlatformEnv } from '../lib/works/vendor/functions/_shared/platform/core';
import { onRequestPost as credentials } from '../lib/works/vendor/functions/api/members/credentials';
import { onRequestPost as magic } from '../lib/works/vendor/functions/api/members/auth';
import { onRequestGet as account, onRequestPost as accountChange } from '../lib/works/vendor/functions/api/members/account';
import { onRequestGet as projects, onRequestPost as save } from '../lib/works/vendor/functions/api/playground/projects';
import { createProject } from '../lib/works/vendor/src/playground/model';

const pass = 'A long unique test passphrase 42!';
const nextPass = 'A different test passphrase 43!';
test('Isolated preview database selection never falls back to unrelated bindings', () => {
  const env = { EIDOS_DATABASE_URL: 'libsql://shared.example', EIDOS_DATABASE_AUTH_TOKEN: 'shared-token', EIDOS_DATABASE_BINDING_PREFIX: 'EIDOS_PG_INTEGRATION' };
  assert.deepEqual(databaseConfiguration(env), { url: undefined, authToken: undefined });
  assert.deepEqual(databaseConfiguration({ ...env, EIDOS_PG_INTEGRATION_TURSO_DATABASE_URL: 'libsql://isolated.example', EIDOS_PG_INTEGRATION_TURSO_AUTH_TOKEN: 'isolated-token' }), { url: 'libsql://isolated.example', authToken: 'isolated-token' });
  assert.throws(() => databaseConfiguration({ ...env, EIDOS_DATABASE_BINDING_PREFIX: '../unsafe' }));
});
async function setup() {
  const client = createClient({ url: ':memory:' });
  for (const file of ['0001_eidos_platform.sql', '0002_members.sql', '0006_member_credentials.sql']) await client.executeMultiple(readFileSync(new URL('../lib/works/vendor/migrations/' + file, import.meta.url), 'utf8'));
  await client.executeMultiple(readFileSync(new URL('../lib/works/vendor/migrations/0006_member_credentials.sql', import.meta.url), 'utf8'));
  const env: PlatformEnv = { EIDOS_DB: adaptDatabase(client), EIDOS_LOCAL_TEST: 'true', EIDOS_RUNTIME: 'sentinel', EIDOS_PASSWORD_SERVICE: passwordService, PUBLIC_SITE_URL: 'http://localhost:8788', EIDOS_RATE_SECRET: 'test-only-at-least-thirty-two-characters' };
  return { client, env };
}
function context(env: PlatformEnv, input?: unknown, cookie = '', path = '/api/members/credentials'): Context {
  return { env, request: new Request('http://localhost:8788' + path, { method: input === undefined ? 'GET' : 'POST', headers: { origin: 'http://localhost:8788', 'content-type': 'application/json', cookie }, ...(input === undefined ? {} : { body: JSON.stringify(input) }) }) };
}
async function payload(response: Response, status = 200) { const result = await response.json(); assert.equal(response.status, status, result.error || 'Unexpected application response status'); return result; }
const cookieOf = (r: Response) => r.headers.get('set-cookie')!.split(';')[0];
async function signup(env: PlatformEnv, name = 'account_a') {
  const pending = await payload(await credentials(context(env, { action: 'signup', email: name + '@example.test', username: name, password: pass })));
  const token = new URLSearchParams(new URL(pending.localVerificationUrl).hash.slice(1)).get('token');
  const verified = await credentials(context(env, { action: 'verify-signup', token }));
  await payload(verified); return { cookie: cookieOf(verified), token, email: name + '@example.test', name };
}
async function resetToken(env: PlatformEnv, email: string) {
  const pending = await payload(await credentials(context(env, { action: 'request-reset', email })));
  return new URLSearchParams(new URL(pending.localVerificationUrl).hash.slice(1)).get('token');
}

await test('Password KDF uses independent salts, preserves spaces and rejects malformed encodings', async () => {
  const a = await passwordService.hash(pass), b = await passwordService.hash(pass);
  assert.match(a, /^scrypt\$131072\$8\$1\$/); assert.notEqual(a, b); assert.ok(!a.includes(pass));
  assert.equal(await passwordService.verify(pass, a), true);
  assert.equal(await passwordService.verify(' ' + pass, a), false);
  assert.equal(await passwordService.verify(pass, null), false);
  assert.equal(await passwordService.verify(pass, 'scrypt$1$1$1$bad$bad'), false);
});

await test('Verified signup, username/email login, logout and exact project persistence use application flows', async () => {
  const { client, env } = await setup();
  try {
    const user = await signup(env);
    assert.equal((await credentials(context(env, { action: 'verify-signup', token: user.token }))).status, 400);
    const document = createProject(); document.name = 'Persistent account project';
    const saved = await payload(await save(context(env, { document }, user.cookie)));
    const memberBefore = (await client.execute('SELECT id FROM eidos_email_members')).rows[0].id;
    await payload(await magic(context(env, { action: 'logout' }, user.cookie)));
    assert.equal((await payload(await account(context(env, undefined, user.cookie)))).member, null);
    for (const identifier of [user.name, user.email]) {
      const signedIn = await credentials(context(env, { action: 'login', identifier, password: pass }));
      await payload(signedIn);
      const reopened = await payload(await projects(context(env, undefined, cookieOf(signedIn), '/api/playground/projects?id=' + saved.id)));
      assert.deepEqual(reopened.document, document); assert.equal(reopened.ownerId, memberBefore);
    }
    const stored = (await client.execute('SELECT password_hash FROM eidos_member_passwords')).rows[0].password_hash;
    assert.match(String(stored), /^scrypt\$/);
    assert.equal((await client.execute('SELECT password_hash FROM eidos_member_auth_tokens')).rows[0].password_hash, null);
  } finally { client.close(); }
});

await test('Two legitimate accounts cannot read projects or revisions owned by each other', async () => {
  const { client, env } = await setup();
  try {
    const a = await signup(env), b = await signup(env, 'account_b'), saved = await payload(await save(context(env, { document: createProject() }, a.cookie)));
    for (const suffix of ['', '&revisions=1', '&revision=' + saved.revision]) {
      const response = await projects(context(env, undefined, b.cookie, '/api/playground/projects?id=' + saved.id + suffix));
      assert.ok([403, 404].includes(response.status), await response.text());
    }
    assert.deepEqual((await payload(await projects(context(env, undefined, b.cookie)))).projects, []);
    assert.equal((await save(context(env, { id: saved.id, expectedRevision: saved.revision, document: createProject() }, b.cookie))).status, 404);
  } finally { client.close(); }
});

await test('Password change rotates current session and reset invalidates all sessions, old links and old passwords', async () => {
  const { client, env } = await setup();
  try {
    const user = await signup(env);
    const second = await credentials(context(env, { action: 'login', identifier: user.name, password: pass })); await payload(second);
    const changed = await credentials(context(env, { action: 'change-password', currentPassword: pass, password: nextPass }, user.cookie)); await payload(changed);
    assert.notEqual(cookieOf(changed), user.cookie);
    for (const cookie of [user.cookie, cookieOf(second)]) assert.equal((await payload(await account(context(env, undefined, cookie)))).member, null);
    const pendingMagic = await payload(await magic(context(env, { action: 'signin', email: user.email })));
    const oldMagicToken = new URLSearchParams(new URL(pendingMagic.localVerificationUrl).hash.slice(1)).get('token');
    const token = await resetToken(env, user.email);
    await payload(await credentials(context(env, { action: 'reset', token, password: pass })));
    assert.equal((await credentials(context(env, { action: 'reset', token, password: nextPass }))).status, 400);
    assert.equal((await payload(await account(context(env, undefined, cookieOf(changed))))).member, null);
    assert.equal((await magic(context(env, { action: 'verify', token: oldMagicToken }))).status, 400);
    assert.equal((await credentials(context(env, { action: 'login', identifier: user.email, password: nextPass }))).status, 401);
    await payload(await credentials(context(env, { action: 'login', identifier: user.email, password: pass })));
  } finally { client.close(); }
});

await test('Existing magic-link member adds a password without changing member, agent key or project identity', async () => {
  const { client, env } = await setup();
  try {
    const pending = await payload(await magic(context(env, { action: 'signup', username: 'legacy_agent', email: 'legacy@example.test', kind: 'agent' })));
    const token = new URLSearchParams(new URL(pending.localVerificationUrl).hash.slice(1)).get('token');
    const signedIn = await magic(context(env, { action: 'verify', token })); await payload(signedIn);
    const owner = (await client.execute('SELECT id FROM eidos_email_members')).rows[0].id;
    const agentKey = await payload(await accountChange(context(env, { action: 'create-key' }, cookieOf(signedIn))), 201);
    const saved = await payload(await save(context(env, { document: createProject() }, cookieOf(signedIn))));
    const reset = await resetToken(env, 'legacy@example.test');
    await payload(await credentials(context(env, { action: 'reset', token: reset, password: pass })));
    const login = await credentials(context(env, { action: 'login', identifier: 'legacy_agent', password: pass })); await payload(login);
    const info = await payload(await projects(context(env, undefined, cookieOf(login), '/api/playground/projects?id=' + saved.id)));
    assert.equal(info.ownerId, owner); assert.equal((await client.execute('SELECT COUNT(*) n FROM eidos_email_members')).rows[0].n, 1);
    assert.equal((await payload(await account(context(env, undefined, cookieOf(login))))).member.kind, 'agent');
    assert.equal((await client.execute({ sql: 'SELECT member_id FROM eidos_member_keys WHERE key_hash=? AND revoked=0', args: [await hash(agentKey.key)] })).rows[0].member_id, owner);
  } finally { client.close(); }
});

await test('Unknown and incorrect credentials have identical errors; reserved usernames, weak passwords and missing challenges fail', async () => {
  const { client, env } = await setup();
  try {
    const user = await signup(env);
    const wrong = await credentials(context(env, { action: 'login', identifier: user.email, password: 'incorrect' }));
    const absent = await credentials(context(env, { action: 'login', identifier: 'missing', password: 'incorrect' }));
    assert.equal(wrong.status, 401); assert.equal(absent.status, 401); assert.equal(await wrong.text(), await absent.text());
    assert.equal((await credentials(context(env, { action: 'signup', username: 'admin', email: 'admin@example.test', password: pass }))).status, 400);
    assert.equal((await credentials(context(env, { action: 'signup', username: 'weak_user', email: 'weak@example.test', password: 'short' }))).status, 400);
    const productionEnv = { ...env, EIDOS_LOCAL_TEST: undefined, EIDOS_PASSWORD_AUTH_ENABLED: 'true', EIDOS_ACCOUNTS_ENABLED: 'true', RESEND_API_KEY: 'test-only', EIDOS_MAIL_FROM: 'test@example.test', TURNSTILE_SITE_KEY: 'test-only', TURNSTILE_SECRET_KEY: 'test-only' };
    assert.equal((await credentials(context(productionEnv, { action: 'login', identifier: user.email, password: pass }))).status, 400);
  } finally { client.close(); }
});

await test('Concurrent reset defeats a password login already inside its KDF', async () => {
  const { client, env } = await setup();
  try {
    const user = await signup(env), token = await resetToken(env, user.email);
    let entered!: () => void, release!: () => void;
    const started = new Promise<void>(r => { entered = r; }), blocked = new Promise<void>(r => { release = r; });
    const slowEnv = { ...env, EIDOS_PASSWORD_SERVICE: { ...passwordService, verify: async (p: string, h: string | null) => { const valid = await passwordService.verify(p, h); entered(); await blocked; return valid; } } };
    const login = credentials(context(slowEnv, { action: 'login', identifier: user.email, password: pass }));
    await started;
    await payload(await credentials(context(env, { action: 'reset', token, password: nextPass })));
    release(); assert.equal((await login).status, 401);
    assert.equal((await client.execute('SELECT COUNT(*) n FROM eidos_member_sessions')).rows[0].n, 0);
  } finally { client.close(); }
});

await test('Reset expiry and single-use admission fail closed, and repeated login attempts are bounded', async () => {
  const { client, env } = await setup();
  try {
    const user = await signup(env), token = await resetToken(env, user.email);
    // Controlled clock-expiry fixture; no identity or ownership rows are changed.
    await client.execute({ sql: 'UPDATE eidos_member_auth_tokens SET expires=0 WHERE token_hash=?', args: [await hash(token!)] });
    assert.equal((await credentials(context(env, { action: 'reset', token, password: nextPass }))).status, 400);
    const statuses = [];
    for (let i = 0; i < 11; i++) statuses.push((await credentials(context(env, { action: 'login', identifier: user.name, password: 'bad' }))).status);
    assert.deepEqual(statuses, [...Array(10).fill(401), 429]);
  } finally { client.close(); }
});

await test('Concurrent application saves cannot exceed the account project cap', async () => {
  const { client, env } = await setup();
  try {
    const user = await signup(env);
    for (let i = 0; i < 19; i++) await payload(await save(context(env, { document: createProject() }, user.cookie)));
    // Both requests observe the last available slot before either writes, as two
    // remote requests can. Only the schedule is controlled; database writes are real.
    let arrivals = 0, release!: () => void;
    const ready = new Promise<void>(resolve => { release = resolve; });
    const original = env.EIDOS_DB!;
    const raced = { ...env, EIDOS_DB: { ...original, prepare(sql: string) {
      const statement = original.prepare(sql);
      if (sql !== 'SELECT COUNT(*) AS n FROM eidos_pg_projects WHERE owner_id=?') return statement;
      const originalBind = statement.bind.bind(statement);
      statement.bind = (...args: unknown[]) => {
        const bound = originalBind(...args), first = bound.first.bind(bound);
        bound.first = async <T>() => { const result = await first<T>(); if (++arrivals === 2) release(); await ready; return result; };
        return bound;
      };
      return statement;
    } } };
    const documents = await Promise.all([0, 255].map(async red => {
      const document = createProject();
      const image = await sharp({ create: { width: 1, height: 1, channels: 4, background: { r: red, g: 100, b: 50, alpha: 1 } } }).png().toBuffer();
      document.sections.find(s => s.id === 'hero')!.image = 'data:image/png;base64,' + image.toString('base64');
      return document;
    }));
    const results = await Promise.all(documents.map(document => save(context({ ...raced, EIDOS_VALIDATE_PLAYGROUND_IMAGE: validatePlaygroundImage }, { document }, user.cookie))));
    assert.deepEqual(results.map(r => r.status).sort(), [200, 409]);
    assert.equal((await payload(await projects(context(env, undefined, user.cookie)))).projects.length, 20);
    assert.equal((await client.execute('SELECT COUNT(*) n FROM eidos_pg_projects')).rows[0].n, 20);
    assert.equal((await client.execute('SELECT COUNT(*) n FROM eidos_pg_assets')).rows[0].n, 1, 'The rejected save must not leave an orphan image');
  } finally { client.close(); }
});

await test('Expired sessions stop reading saved work and a fresh password login preserves ownership', async () => {
  const { client, env } = await setup(); const originalNow = Date.now;
  try {
    const user = await signup(env), saved = await payload(await save(context(env, { document: createProject() }, user.cookie)));
    Date.now = () => originalNow() + 31 * 86400000;
    assert.equal((await payload(await account(context(env, undefined, user.cookie)))).member, null);
    assert.equal((await projects(context(env, undefined, user.cookie, '/api/playground/projects?id=' + saved.id))).status, 401);
    Date.now = originalNow;
    const login = await credentials(context(env, { action: 'login', identifier: user.email, password: pass })); await payload(login);
    assert.equal((await payload(await projects(context(env, undefined, cookieOf(login), '/api/playground/projects?id=' + saved.id)))).id, saved.id);
  } finally { Date.now = originalNow; client.close(); }
});

test('database cache cannot reuse a previous binding after configuration changes',()=>{const first={EIDOS_DATABASE_URL:'libsql://first.example',EIDOS_DATABASE_AUTH_TOKEN:'fixture-one'};const second={EIDOS_DATABASE_URL:'libsql://second.example',EIDOS_DATABASE_AUTH_TOKEN:'fixture-two'};const a=platformDatabase(first);assert.equal(platformDatabase(first),a);assert.notEqual(platformDatabase(second),a);assert.equal(platformDatabase({...second,EIDOS_DATABASE_BINDING_PREFIX:'EIDOS_ABSENT'}),undefined);});
