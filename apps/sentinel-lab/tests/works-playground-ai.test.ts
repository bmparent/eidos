import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { unlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import aiSchema from '../lib/works/vendor/functions/_shared/platform/playgroundAiSchema';
import { createClient } from '@libsql/client';
import { adaptDatabase } from '../lib/works/database';
import { hash, type PlatformEnv } from '../lib/works/vendor/functions/_shared/platform/core';
import { onRequestGet, onRequestPost } from '../lib/works/vendor/functions/api/playground/ai';
import { USD, aiBalance, aiPolicyStatus, cancelAiAction, claimAiDispatch, configureAiPolicy, ensureAiLedger, grantPaidAiBalance, markAiUnknown,
  meterAiUsage, monthlyPeriod, ownedAiAction, providerMaximum, quoteAiAction, reconcileAiDeadlines, reconcileAiReversals,
  reserveAiAction, retailCharge, setAiLimit, settleAiAction, validateRateCard, recordVerifiedAiPayment, type AiBrief, type AiRateCard } from '../lib/works/vendor/functions/_shared/platform/playgroundAi';

// CONTROLLED FIXTURES ONLY: these prices are deliberately synthetic, not accepted provider/customer prices.
const at = new Date('2026-10-14T12:00:00.000Z');
const card: AiRateCard = { version: 'fixture-v1', model: 'gpt-6.1-sol', evidence: 'controlled-fixture',
  inputMicrosPerMillion: 50 * USD, outputMicrosPerMillion: 100 * USD, retailNumerator: 1, retailDenominator: 1,
  maxInputTokens: 8000, maxOutputTokens: 2000, maxCalls: 1, maxRetries: 0, maxAgentTurns: 1, maxSubagents: 0,
  paidTools: 0, maxToolReturnBytes: 0, processing: 'standard', reasoning: 'medium', maxRuntimeMs: 30000, validUntil: '2027-01-01T00:00:00.000Z' };
const brief: AiBrief = { documentId: 'document-one', revision: 'revision-1', scope: 'element', targetId: 'headline-1', instruction: 'Clarify this offer', context: 'A local service business. Current headline: Reliable cleaning.' };
const usage = { inputTokens: 1000, outputTokens: 1000, calls: 1, paidTools: 0 };
const fullUsage = { inputTokens: card.maxInputTokens, outputTokens: card.maxOutputTokens, calls: 1, paidTools: 0 };
async function fixture(limit = USD, balance = USD, budget = 10 * USD, databaseUrl = ':memory:') {
  const client = createClient({ url: databaseUrl }), database = adaptDatabase(client);
  for (const name of ['0001_eidos_platform.sql', '0002_members.sql', '0004_paid_templates.sql'])
    await client.executeMultiple(readFileSync(new URL('../lib/works/vendor/migrations/' + name, import.meta.url), 'utf8'));
  const env: PlatformEnv = { EIDOS_DB: database, EIDOS_LOCAL_TEST: 'true', EIDOS_PLAYGROUND_AI_MODE: 'test', EIDOS_TEMPLATE_STRIPE_ACCOUNT_ID: 'acct_fixturemerchant' };
  await ensureAiLedger(env); await ensureAiLedger(env);
  for (const owner of ['alice', 'bob']) {
    await database.prepare('INSERT INTO eidos_email_members(id,email,username,kind,created_at,newsletter_after) VALUES(?,?,?,?,?,?)').bind(owner, owner + '@example.test', owner, 'person', at.toISOString(), at.toISOString()).run();
    await setAiLimit(env, owner, 'test', limit, at);
  }
  await configureAiPolicy(env, 'test', card, budget, true);
  let serial = 0;
  async function paid(owner: string, amount: number, kind: 'included' | 'purchased' = 'purchased', status = 'paid', mode = 'test') {
    const id = 'order-' + ++serial;
    await database.prepare('INSERT INTO eidos_orders(id,receipt_hash,session_id,status,created_at,payment_intent) VALUES(?,?,?,?,?,?)').bind(id, id, 'cs_' + mode + '_' + id, status, at.toISOString(), 'pi_' + mode + '_' + id).run();
    await database.prepare('INSERT INTO eidos_template_orders(order_id,product_id,edition_id,version,price_cents,archive_key,archive_sha256,download_name,guide_path,member_id,receipt_expires,checkout_expires,fulfillment_status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)')
      .bind(id, 'switchboard-developer-v1', 'developer', '1.0.0', 500, 'immutable/archive.zip', 'a'.repeat(64), 'archive.zip', '/setup', owner, 2000000000, 2000000000, status === 'paid' ? 'ready' : 'awaiting_payment', at.toISOString()).run();
    await database.prepare('INSERT INTO eidos_stripe_events(id,received_at) VALUES(?,?)').bind('fixture-event-' + id, at.toISOString()).run();
    await database.prepare('INSERT INTO eidos_pg_ai_checkout_merchants(order_id,mode,merchant_id,created_at) VALUES(?,?,?,?)').bind(id, mode, 'acct_fixturemerchant', at.toISOString()).run();
    if (mode === 'test') await recordVerifiedAiPayment(env, id, 'fixture-event-' + id, 'test', 'acct_fixturemerchant');
    const rule = { productId: 'switchboard-developer-v1', editionId: 'developer', version: '1.0.0', priceCents: 500, grantVersion: 'fixture-allowance-v1', kind, amount };
    return { id, rule };
  }
  async function credit(owner: string, amount: number, kind: 'included' | 'purchased' = 'purchased') {
    const row = await paid(owner, amount, kind); await grantPaidAiBalance(env, owner, 'test', row.id, row.rule); return row;
  }
  if (balance) await credit('alice', balance);
  let requests = 0, calls = 0;
  async function quote(owner = 'alice', change: Partial<AiBrief> = {}, date = at, key = 'request-' + ++requests) {
    return quoteAiAction(env, owner, 'test', key, { ...brief, ...change }, date);
  }
  async function execute(id: string, owner = 'alice', metered = usage, usable = true) {
    await reserveAiAction(env, owner, 'test', id, at);
    const token = await claimAiDispatch(env, owner, 'test', id, at);
    if (token) { calls++; return settleAiAction(env, owner, 'test', id, { dispatchToken: token, providerId: 'response-' + id, providerRequestId: 'req-' + id, usage: metered, usable, result: { operations: [] } }, at); }
    return ownedAiAction(env, owner, 'test', id);
  }
  return { client, database, env, quote, execute, paid, credit, calls: () => calls, close: () => client.close() };
}

test('01: concurrent $0.60 actions with $1 remaining admit exactly one and only one dispatch', async () => {
  const f = await fixture();
  try {
    const actions = await Promise.all([f.quote(), f.quote()]); assert.equal(actions[0].maximum, 600000);
    const results = await Promise.allSettled(actions.map(a => f.execute(a.id, 'alice', fullUsage)));
    assert.equal(results.filter(x => x.status === 'fulfilled').length, 1); assert.equal(f.calls(), 1);
    const b = await aiBalance(f.env, 'alice', 'test', at); assert.equal(b.settled, 600000); assert.equal(b.availablePurchased, 400000);
  } finally { f.close(); }
});
test('02: concurrent action replay yields one reservation, dispatch and settlement; conflicting key fails', async () => {
  const f = await fixture();
  try {
    const quotes = await Promise.all(Array.from({ length: 12 }, () => f.quote('alice', {}, at, 'same-request')));
    assert.equal(new Set(quotes.map(a => a.id)).size, 1);
    await Promise.all(quotes.map(a => f.execute(a.id))); assert.equal(f.calls(), 1);
    assert.equal((await aiBalance(f.env, 'alice', 'test', at)).settled, 150000);
    await assert.rejects(f.quote('alice', { revision: 'newer' }, at, 'same-request'), /different scope|another request/);
  } finally { f.close(); }
});
test('03: $50 balance/$10 cap, included consumption and top-ups never lift the cap', async () => {
  const f = await fixture(10 * USD, 50 * USD, 100 * USD);
  try {
    await f.credit('alice', 5 * USD, 'included');
    for (let i = 0; i < 16; i++) await f.execute((await f.quote()).id, 'alice', fullUsage);
    await assert.rejects(f.execute((await f.quote()).id, 'alice', fullUsage), /allowance, monthly limit/);
    await f.credit('alice', 10 * USD);
    await assert.rejects(f.execute((await f.quote()).id, 'alice', fullUsage), /allowance, monthly limit/);
    const b = await aiBalance(f.env, 'alice', 'test', at);
    assert.equal(b.requestedLimit, 10 * USD); assert.equal(b.settled, 9600000); assert.equal(b.settledIncluded, 5 * USD); assert.equal(b.settledPurchased, 4600000);
  } finally { f.close(); }
});
test('04: projects/devices share owner cap and authenticated endpoints reject cross-account IDs/forged owners', async () => {
  const f = await fixture();
  try {
    const a = await f.quote(), b = await f.quote('alice', { documentId: 'another-project' });
    await reserveAiAction(f.env, 'alice', 'test', a.id, at);
    await assert.rejects(reserveAiAction(f.env, 'alice', 'test', b.id, at), /allowance, monthly limit/);
    await assert.rejects(ownedAiAction(f.env, 'bob', 'test', a.id), /not found/);
    const token = 'b'.repeat(64);
    await f.database.prepare('INSERT INTO eidos_member_sessions(token_hash,member_id,expires) VALUES(?,?,?)').bind(await hash(token), 'bob', 2000000000).run();
    const headers = { 'content-type': 'application/json', origin: 'http://localhost', cookie: 'eidos_session_local=' + token };
    const denied = await onRequestGet({ env: f.env, request: new Request('http://localhost/api/playground/ai?id=' + a.id, { headers }) });
    assert.equal(denied.status, 404);
    const forged = await onRequestPost({ env: f.env, request: new Request('http://localhost/api/playground/ai', { method: 'POST', headers, body: JSON.stringify({ action: 'limit', limitMicros: 99 * USD, owner: 'alice' }) }) });
    assert.equal(forged.status, 400); assert.equal((await aiBalance(f.env, 'alice', 'test', at)).requestedLimit, USD);
    const unauth = await onRequestGet({ env: f.env, request: new Request('http://localhost/api/playground/ai') }); assert.equal(unauth.status, 401);
  } finally { f.close(); }
});
test('05: unknown pricing, forged totals/model/owner, large context and unbounded tools/retries fail closed', async () => {
  const f = await fixture();
  try {
    for (const field of ['model', 'owner', 'maximum', 'tools']) await assert.rejects(quoteAiAction(f.env, 'alice', 'test', 'forged-' + field, { ...brief, [field]: 1 }, at), /server-owned/);
    await assert.rejects(f.quote('alice', { context: 'x'.repeat(9000) }), /context exceeds/);
    for (const change of [{ maxRetries: 1 }, { maxCalls: 2 }, { paidTools: 1 }, { processing: 'flex' }, { inputMicrosPerMillion: 0 }, { inputMicrosPerMillion: 0.1 }])
      assert.throws(() => validateRateCard({ ...card, ...change }));
    await f.database.prepare('DELETE FROM eidos_pg_ai_policies').run(); await assert.rejects(f.quote(), /paused/);
    await assert.rejects(configureAiPolicy(f.env, 'live', card, USD, true), /LIVE/);
    assert.throws(() => retailCharge(0.1, card));
  } finally { f.close(); }
});
test('06: actual usage releases difference; above-bound usage never overcharges and disables route', async () => {
  const f = await fixture(10 * USD, 10 * USD);
  try {
    await f.execute((await f.quote()).id);
    assert.equal((await aiBalance(f.env, 'alice', 'test', at)).availablePurchased, 9850000);
    const a = await f.quote(); const settled = await f.execute(a.id, 'alice', { ...fullUsage, outputTokens: 5000 });
    assert.equal(settled.charged_purchased, a.maximum); assert.equal(settled.provider_cost, 900000);
    await assert.rejects(f.quote(), /paused/);
    assert.equal((await aiBalance(f.env, 'alice', 'test', at)).pending, 0);
  } finally { f.close(); }
});
test('07: unusable technical output restores allowance while real provider expense remains counted', async () => {
  const f = await fixture(USD, USD, 700000);
  try {
    const a = await f.quote(); const settled = await f.execute(a.id, 'alice', usage, false);
    assert.equal(settled.state, 'technical_waiver'); assert.equal(settled.provider_cost, 150000);
    assert.equal((await aiBalance(f.env, 'alice', 'test', at)).availablePurchased, USD);
    await assert.rejects(f.execute((await f.quote()).id), /allowance, monthly limit/);
  } finally { f.close(); }
});
test('08: submitted timeout/crash/lease expiry retain provider cost; 24h waiver never retroactively debits', async () => {
  const f = await fixture(USD, USD, 700000);
  try {
    const a = await f.quote(); await reserveAiAction(f.env, 'alice', 'test', a.id, at);
    const token = await claimAiDispatch(f.env, 'alice', 'test', a.id, at); assert.ok(token);
    await markAiUnknown(f.env, 'alice', 'test', a.id, 'timeout-or-worker-crash');
    assert.equal(await claimAiDispatch(f.env, 'alice', 'test', a.id, at), null);
    await cancelAiAction(f.env, 'alice', 'test', a.id);
    await reconcileAiDeadlines(f.env, new Date(at.getTime() + 3600000));
    assert.equal((await ownedAiAction(f.env, 'alice', 'test', a.id)).provider_status, 'held');
    await reconcileAiDeadlines(f.env, new Date(at.getTime() + 25 * 3600000));
    const waived = await ownedAiAction(f.env, 'alice', 'test', a.id); assert.equal(waived.provider_status, 'pessimistic'); assert.equal(waived.provider_cost, a.provider_maximum);
    assert.equal((await aiBalance(f.env, 'alice', 'test', at)).availablePurchased, USD);
    await assert.rejects(f.execute((await f.quote()).id), /allowance, monthly limit/);
    await settleAiAction(f.env, 'alice', 'test', a.id, { dispatchToken: token, providerId: 'response-late', providerRequestId: 'req-late', usage, usable: true, result: {} }, at);
    const late = await ownedAiAction(f.env, 'alice', 'test', a.id); assert.equal(late.charged_purchased, 0); assert.equal(late.customer_status, 'waived'); assert.equal(late.provider_cost, 150000);
  } finally { f.close(); }
});
test('09/13: duplicate/unpaid/out-of-order grants, refunds, owner mismatch and TEST/LIVE isolation', async () => {
  const f = await fixture();
  try {
    const row = await f.paid('alice', USD, 'included');
    const grants = await Promise.all(Array.from({ length: 10 }, () => grantPaidAiBalance(f.env, 'alice', 'test', row.id, row.rule)));
    assert.equal(grants.filter(g => g.granted).length, 1);
    await assert.rejects(grantPaidAiBalance(f.env, 'bob', 'test', row.id, row.rule), /matching verified/);
    await assert.rejects(grantPaidAiBalance(f.env, 'alice', 'test', row.id, { ...row.rule, grantVersion: 'attempted-regrant' }), /matching verified/);
    await assert.rejects(grantPaidAiBalance(f.env, 'alice', 'live', row.id, row.rule), /LIVE/);
    const unpaid = await f.paid('alice', USD, 'included', 'pending'); await assert.rejects(grantPaidAiBalance(f.env, 'alice', 'test', unpaid.id, unpaid.rule), /matching verified/);
    const wrongMode = await f.paid('alice', USD, 'included', 'paid', 'live'); await assert.rejects(grantPaidAiBalance(f.env, 'alice', 'test', wrongMode.id, wrongMode.rule), /matching verified/);
    await f.database.prepare("UPDATE eidos_orders SET status='refunded' WHERE id=?").bind(row.id).run();
    await assert.rejects(f.execute((await f.quote()).id), /allowance, monthly limit/);
    await reconcileAiReversals(f.env); await reconcileAiReversals(f.env);
    const b = await aiBalance(f.env, 'alice', 'test', at); assert.equal(b.frozen, true); assert.equal(b.availableIncluded, 0); assert.equal(b.availablePurchased, USD);
    const replay = await grantPaidAiBalance(f.env, 'alice', 'test', row.id, row.rule); assert.equal(replay.granted, false); assert.equal(replay.reversed, true);
  } finally { f.close(); }
});
test('10: cap edits racing dispatch/settlement preserve the commitment floor and block new work', async () => {
  const f = await fixture();
  try {
    const a = await f.quote(); await reserveAiAction(f.env, 'alice', 'test', a.id, at);
    await setAiLimit(f.env, 'alice', 'test', 200000, at);
    const before = await aiBalance(f.env, 'alice', 'test', at); assert.equal(before.effectiveLimit, 600000); assert.equal(before.loweringPending, true); assert.equal(before.remainingCap, 0);
    await Promise.all([f.execute(a.id), setAiLimit(f.env, 'alice', 'test', 100000, at)]);
    const b = await aiBalance(f.env, 'alice', 'test', at); assert.equal(b.effectiveLimit, 150000); assert.equal(b.requestedLimit, 100000); assert.equal(b.pending, 0);
    await assert.rejects(f.execute((await f.quote()).id), /allowance, monthly limit/);
  } finally { f.close(); }
});
test('11: monthly period is server UTC; pending budget survives, and late settlement stays original month', async () => {
  const f = await fixture(USD, 2 * USD, USD);
  try {
    const end = new Date('2026-10-31T23:59:30.000Z'), next = new Date('2026-11-01T00:00:10.000Z');
    assert.deepEqual(monthlyPeriod(end), { period: '2026-10', resetsAt: '2026-11-01T00:00:00.000Z' });
    const a = await f.quote('alice', {}, end); await reserveAiAction(f.env, 'alice', 'test', a.id, end);
    const token = await claimAiDispatch(f.env, 'alice', 'test', a.id, end); assert.ok(token);
    const newAction = await f.quote('alice', {}, next); await assert.rejects(reserveAiAction(f.env, 'alice', 'test', newAction.id, next), /allowance, monthly limit/);
    await settleAiAction(f.env, 'alice', 'test', a.id, { dispatchToken: token, providerId: 'r', providerRequestId: 'q', usage, usable: true, result: {} }, next);
    assert.equal((await aiBalance(f.env, 'alice', 'test', next)).settled, 0); assert.equal((await aiBalance(f.env, 'alice', 'test', end)).settled, 150000);
    await reserveAiAction(f.env, 'alice', 'test', newAction.id, next);
    assert.equal((await aiBalance(f.env, 'alice', 'test', next)).pending, 600000);
  } finally { f.close(); }
});
test('12/14: owner aggregate budget admits only one customer; paused route has no provider fallback', async () => {
  const f = await fixture(USD, USD, USD);
  try {
    await f.credit('bob', USD);
    const a = await f.quote(), b = await f.quote('bob');
    const results = await Promise.allSettled([reserveAiAction(f.env, 'alice', 'test', a.id, at), reserveAiAction(f.env, 'bob', 'test', b.id, at)]);
    assert.equal(results.filter(r => r.status === 'fulfilled').length, 1);
    await configureAiPolicy(f.env, 'test', card, USD, false);
    const admitted = results[0].status === 'fulfilled' ? a : b, owner = admitted.owner_id;
    await assert.rejects(claimAiDispatch(f.env, owner, 'test', admitted.id, at), /paused/);
    await cancelAiAction(f.env, owner, 'test', admitted.id);
    assert.equal((await ownedAiAction(f.env, owner, 'test', admitted.id)).provider_status, 'none'); assert.equal(f.calls(), 0);
  } finally { f.close(); }
});
test('migration/schema parity and conservative integer rounding cover sub-cent rates', () => {
  assert.equal(aiSchema.replaceAll('\r\n', '\n'), readFileSync(new URL('../lib/works/vendor/migrations/0005_playground_ai.sql', import.meta.url), 'utf8').replaceAll('\r\n', '\n'));
  assert.equal(providerMaximum(card), 600000);
  assert.equal(retailCharge(1, card), 10000);
  assert.equal(meterAiUsage({ inputTokens: 1, outputTokens: 1, calls: 1, paidTools: 0 }, { ...card, inputMicrosPerMillion: 1, outputMicrosPerMillion: 1 }), 2);
});
test('independent database connections serialize shared provider/customer admission', async (t) => {
  const path = join(tmpdir(), 'eidos-ai-fixture-' + crypto.randomUUID() + '.db'), url = 'file:' + path.replaceAll('\\', '/');
  const f = await fixture(USD, USD, USD, url), otherClient = createClient({ url });
  const otherEnv = { ...f.env, EIDOS_DB: adaptDatabase(otherClient) };
  try {
    await f.credit('bob', USD);
    const a = await f.quote(), b = await f.quote('bob');
    const results = await Promise.allSettled([reserveAiAction(f.env, 'alice', 'test', a.id, at), reserveAiAction(otherEnv, 'bob', 'test', b.id, at)]);
    assert.equal(results.filter(result => result.status === 'fulfilled').length, 1);
    assert.equal((await f.database.prepare("SELECT SUM(provider_maximum) n FROM eidos_pg_ai_actions WHERE provider_status='held'").first<{ n: number }>())?.n, 600000);
  } finally {
    otherClient.close(); f.close();
    for (const suffix of ['', '-wal', '-shm']) await unlink(path + suffix).catch(error => {
      if (error.code === 'EBUSY' && process.platform === 'win32') t.diagnostic('Windows libSQL retains the synthetic temp database handle until process exit: ' + path + suffix);
      else if (error.code !== 'ENOENT') throw error;
    });
  }
});

test('owner budget reductions block reserved dispatch atomically across independent database connections', async (t) => {
  const path = join(tmpdir(), 'eidos-ai-budget-fixture-' + crypto.randomUUID() + '.db'), url = 'file:' + path.replaceAll('\\', '/');
  const f = await fixture(USD, USD, USD, url), otherClient = createClient({ url });
  const otherEnv = { ...f.env, EIDOS_DB: adaptDatabase(otherClient) };
  try {
    const a = await f.quote();
    await reserveAiAction(f.env, 'alice', 'test', a.id, at);
    const reduced = await configureAiPolicy(otherEnv, 'test', card, 0, true);
    assert.equal(reduced.configuredBudget, 0); assert.equal(reduced.reserved, 600000);
    assert.equal(reduced.settled, 0); assert.equal(reduced.submitted, 0); assert.equal(reduced.overBudget, true);
    assert.equal(reduced.dispatchBlocked, true);
    assert.equal(await claimAiDispatch(f.env, 'alice', 'test', a.id, at), null);
    assert.equal((await ownedAiAction(f.env, 'alice', 'test', a.id)).state, 'reserved');
    await cancelAiAction(f.env, 'alice', 'test', a.id);
    assert.equal((await aiPolicyStatus(otherEnv, 'test')).committed, 0);
    // Either admission wins before the reduction or the reduction rejects admission.
    // After both commits there can be no new dispatch under the zero budget.
    await configureAiPolicy(otherEnv, 'test', card, USD, true);
    const b = await f.quote();
    const interleaved = await Promise.allSettled([
      reserveAiAction(f.env, 'alice', 'test', b.id, at),
      configureAiPolicy(otherEnv, 'test', card, 0, true),
    ]);
    assert.equal(interleaved[1].status, 'fulfilled');
    assert.equal(await claimAiDispatch(f.env, 'alice', 'test', b.id, at), null);
    assert.equal((await aiPolicyStatus(otherEnv, 'test')).configuredBudget, 0);
    await cancelAiAction(f.env, 'alice', 'test', b.id);
    await configureAiPolicy(otherEnv, 'test', card, USD, true);
    const c = await f.quote(); await reserveAiAction(f.env, 'alice', 'test', c.id, at);
    assert.ok(await claimAiDispatch(f.env, 'alice', 'test', c.id, at));
    const inFlight = await configureAiPolicy(otherEnv, 'test', card, 0, true);
    assert.equal(inFlight.reserved, 0); assert.equal(inFlight.submitted, 600000); assert.equal(inFlight.settled, 0);
    await cancelAiAction(f.env, 'alice', 'test', c.id);
    assert.equal((await aiPolicyStatus(otherEnv, 'test')).submitted, 600000);
    assert.equal(f.calls(), 0);
  } finally {
    otherClient.close(); f.close();
    for (const suffix of ['', '-wal', '-shm']) await unlink(path + suffix).catch(error => {
      if (error.code === 'EBUSY' && process.platform === 'win32') t.diagnostic('Windows libSQL retains the synthetic temp database handle until process exit: ' + path + suffix);
      else if (error.code !== 'ENOENT') throw error;
    });
  }
});
