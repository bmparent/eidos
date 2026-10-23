import test from 'node:test';
import assert from 'node:assert/strict';
import { createClient } from '@libsql/client';
import { readFileSync } from 'node:fs';
import { adaptDatabase } from '../lib/works/database';
import { onRequestPost as maintenance } from '../lib/works/vendor/functions/api/community/maintenance';
import type { PlatformEnv } from '../lib/works/vendor/functions/_shared/platform/core';

async function maintenanceFixture() {
  const client = createClient({ url: ':memory:' });
  await client.execute('PRAGMA foreign_keys=ON');
  for (const name of ['0001_eidos_platform.sql', '0002_members.sql', '0006_member_credentials.sql'])
    await client.executeMultiple(readFileSync(new URL('../lib/works/vendor/migrations/' + name, import.meta.url), 'utf8'));
  const env: PlatformEnv = {
    EIDOS_DB: adaptDatabase(client), EIDOS_LOCAL_TEST: 'true', EIDOS_ACCOUNTS_ENABLED: 'true',
    EIDOS_MAINTENANCE_TOKEN: 'test-only-maintenance-token-at-least-32-characters',
    PUBLIC_SITE_URL: 'http://localhost:8788',
  };
  return { env, close: () => client.close() };
}
test('Account maintenance expires only transient credentials, preserves live claims and persistent records, and repeats safely', async () => {
  const { env, close } = await maintenanceFixture();
  const database = env.EIDOS_DB!, originalNow = Date.now;
  Date.now = () => 1_800_000_000_000;
  const cutoff = Math.floor(Date.now() / 1000);
  const tables = ['eidos_member_auth_tokens', 'eidos_member_oauth_flows', 'eidos_member_google_signup'];
  const persistent = ['eidos_email_members', 'eidos_member_google', 'eidos_member_passwords', 'eidos_member_keys', 'eidos_bookmarks', 'eidos_orders', 'eidos_stripe_events', 'eidos_revoked_payments'];
  const rows = async (table: string) => (await database.prepare('SELECT * FROM ' + table + ' ORDER BY 1').all()).results;
  async function run(token = env.EIDOS_MAINTENANCE_TOKEN!, settings = env) {
    return maintenance({ env: settings, request: new Request('http://localhost:8788/api/community/maintenance', { method: 'POST', headers: { authorization: 'Bearer ' + token } }) });
  }
  try {
    await database.prepare("INSERT INTO eidos_email_members(id,email,username,kind,created_at,newsletter_after) VALUES('member','member@example.test','member','person','2020-01-01','2020-01-01')").run();
    await database.prepare("INSERT INTO eidos_member_google VALUES('linked-subject','member','2020-01-01')").run();
    await database.prepare("INSERT INTO eidos_member_passwords VALUES('member','persistent-password-hash','2020-01-01')").run();
    await database.prepare("INSERT INTO eidos_member_keys(id,member_id,key_hash,last_four,created_at) VALUES('key','member','key-hash','1234','2020-01-01')").run();
    await database.prepare("INSERT INTO eidos_bookmarks VALUES('member','saved-article','2020-01-01')").run();
    await database.prepare("INSERT INTO eidos_orders(id,receipt_hash,status,created_at,payment_intent) VALUES('order','receipt-hash','paid','2020-01-01','payment')").run();
    await database.prepare("INSERT INTO eidos_stripe_events VALUES('event','2020-01-01')").run();
    await database.prepare("INSERT INTO eidos_revoked_payments VALUES('refund','2020-01-01')").run();
    // consumed=1 is an in-flight claim; consumed=2 is completed. Neither is
    // sufficient for deletion while unexpired. Include abandoned secret-bearing rows.
    for (const delta of [-1, 0, 1]) for (const consumed of [0, 1, 2]) {
      const key = `${delta}:${consumed}`, expires = cutoff + delta;
      for (const purpose of ['signup', 'reset']) await database.prepare('INSERT INTO eidos_member_auth_tokens(token_hash,purpose,member_id,email,password_hash,expires,consumed) VALUES(?,?,?,?,?,?,?)')
        .bind(key + purpose, purpose, 'member', 'member@example.test', purpose === 'signup' ? 'pending-password-hash' : null, expires, consumed).run();
      await database.prepare('INSERT INTO eidos_member_oauth_flows(state_hash,browser_hash,nonce,verifier,member_id,expires,consumed) VALUES(?,?,?,?,?,?,?)')
        .bind(key, 'browser', 'nonce', 'pkce-secret', 'member', expires, consumed).run();
      await database.prepare('INSERT INTO eidos_member_google_signup(token_hash,subject,email,expires,consumed) VALUES(?,?,?,?,?)')
        .bind(key, 'pending-subject', 'pending@example.test', expires, consumed).run();
    }
    const saved = await Promise.all(persistent.map(rows));
    const transient = await Promise.all(tables.map(rows));
    assert.equal((await run('wrong')).status, 401);
    assert.deepEqual(await Promise.all(tables.map(rows)), transient, 'Unauthorized maintenance must not delete anything');
    // Accounts-disabled installations can still be on the pre-0006 schema.
    assert.equal((await run(env.EIDOS_MAINTENANCE_TOKEN!, { ...env, EIDOS_LOCAL_TEST: undefined, EIDOS_ACCOUNTS_ENABLED: undefined })).status, 200);
    assert.deepEqual(await Promise.all(tables.map(rows)), transient, 'Keep the existing account feature gate');
    const live = transient.map(records => records.filter(record => Number((record as { expires: number }).expires) > cutoff));
    for (let repeat = 0; repeat < 2; repeat++) {
      const response = await run();
      assert.equal(response.status, 200, await response.clone().text());
      assert.deepEqual(await Promise.all(tables.map(rows)), live, 'Delete expired/boundary rows only, regardless of consumed state');
      assert.deepEqual(await Promise.all(persistent.map(rows)), saved, 'Linked identities, credentials and payment history must remain byte-for-byte unchanged');
    }
  } finally { Date.now = originalNow; close(); }
});
