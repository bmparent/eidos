import { db, hash, HttpError, record, type Database, type PlatformEnv } from './core';
import schema from './playgroundAiSchema';

/** Accounting only. No provider transport is enabled by this module. Never accept a rate card from a browser. */
export type AiMode = 'test' | 'live';
export const USD = 1_000_000;
export const REAL_PROVIDER_TEST_BUDGET = 5 * USD;
export const AI_COUNT_LIMITS = { perOwnerHour: 20, globalHour: 60, lifetime: 200 } as const;
export const CUSTOMER_HOLD_WINDOW_MS = 24 * 60 * 60 * 1000;
const MAX_MONEY = 1_000_000_000_000;
const initialized = new WeakMap<Database, Promise<unknown>>();
export async function ensureAiLedger(env: PlatformEnv) {
  const database = db(env);
  let ready = initialized.get(database);
  if (!ready) {
    ready = database.batch(schema.split(';').map(s => s.trim()).filter(Boolean).map(s => database.prepare(s)));
    initialized.set(database, ready);
  }
  try { await ready; } catch (error) { initialized.delete(database); throw error; }
}
export function money(value: unknown, positive = false): number {
  if (!Number.isSafeInteger(value) || Number(value) < (positive ? 1 : 0) || Number(value) > MAX_MONEY)
    throw new HttpError(400, 'USD amounts must be bounded integer microdollars.');
  return Number(value);
}
function boundedInteger(value: unknown, min: number, max: number) {
  if (!Number.isSafeInteger(value) || Number(value) < min || Number(value) > max) throw new HttpError(503, 'AI pricing or limits are invalid.');
  return Number(value);
}
function ceilingRatio(a: number, b: number, denominator: number) {
  const result = (BigInt(a) * BigInt(b) + BigInt(denominator) - 1n) / BigInt(denominator);
  if (result > BigInt(MAX_MONEY)) throw new HttpError(503, 'AI bound exceeds the supported amount.');
  return Number(result);
}
export function monthlyPeriod(at = new Date()) {
  if (!Number.isFinite(at.getTime())) throw new HttpError(503, 'Server clock is unavailable.');
  return { period: at.toISOString().slice(0, 7), resetsAt: new Date(Date.UTC(at.getUTCFullYear(), at.getUTCMonth() + 1, 1)).toISOString() };
}
export interface AiRateCard {
  version: string; model: 'gpt-6.1-sol' | 'gpt-6-astra'; evidence: 'controlled-fixture' | 'owner-approved';
  inputMicrosPerMillion: number; outputMicrosPerMillion: number;
  /** Optional for historical fixtures. Real provider cards must price every input category. */
  cachedInputMicrosPerMillion?: number; cacheWriteMicrosPerMillion?: number;
  retailNumerator: number; retailDenominator: number;
  maxInputTokens: number; maxOutputTokens: number; maxCalls: 1; maxRetries: 0;
  maxAgentTurns: 1; maxSubagents: 0; paidTools: 0; maxToolReturnBytes: 0;
  processing: 'standard'; reasoning: 'low' | 'medium' | 'high'; maxRuntimeMs: number;
  validUntil: string;
}
export function validateRateCard(input: unknown): AiRateCard {
  if (!record(input) || !/^[a-zA-Z0-9._-]{4,100}$/.test(String(input.version)) ||
      !['gpt-6.1-sol', 'gpt-6-astra'].includes(String(input.model)) ||
      !['controlled-fixture', 'owner-approved'].includes(String(input.evidence)) ||
      input.maxCalls !== 1 || input.maxRetries !== 0 || input.maxAgentTurns !== 1 || input.maxSubagents !== 0 ||
      input.paidTools !== 0 || input.maxToolReturnBytes !== 0 || input.processing !== 'standard' ||
      !['low', 'medium', 'high'].includes(String(input.reasoning)) || !Number.isFinite(Date.parse(String(input.validUntil))))
    throw new HttpError(503, 'AI routing is not bounded by an accepted rate card.');
  for (const key of ['inputMicrosPerMillion', 'outputMicrosPerMillion', 'retailNumerator', 'retailDenominator']) money(input[key], true);
  for (const key of ['cachedInputMicrosPerMillion', 'cacheWriteMicrosPerMillion']) if (input[key] !== undefined) money(input[key], true);
  if ((input.cachedInputMicrosPerMillion === undefined) !== (input.cacheWriteMicrosPerMillion === undefined)) throw new HttpError(503, 'All input categories require accepted pricing.');
  boundedInteger(input.maxInputTokens, 2048, 128000);
  boundedInteger(input.maxOutputTokens, 128, 16000);
  boundedInteger(input.maxRuntimeMs, 1000, 120000);
  return structuredClone(input) as unknown as AiRateCard;
}
export function providerMaximum(card: AiRateCard) {
  // Every input token reserves the most expensive input category, including cache writes.
  // Reasoning is already included in the provider's total output-token bound.
  return money(ceilingRatio(card.maxInputTokens, Math.max(card.inputMicrosPerMillion, card.cachedInputMicrosPerMillion || 0, card.cacheWriteMicrosPerMillion || 0), 1_000_000) +
    ceilingRatio(card.maxOutputTokens, card.outputMicrosPerMillion, 1_000_000), true);
}
export function retailCharge(providerCost: number, card: AiRateCard) {
  // Retail rounds UP to a whole USD cent, using BigInt rather than accumulated binary floating point.
  const micros = ceilingRatio(money(providerCost), card.retailNumerator, card.retailDenominator);
  return money(ceilingRatio(micros, 1, 10_000) * 10_000);
}
interface Policy { mode: AiMode; enabled: number; provider_budget: number; rate_json: string; paused_reason: string | null }
async function policy(env: PlatformEnv, mode: AiMode, at: Date, mustEnable = true) {
  const row = await db(env).prepare('SELECT * FROM eidos_pg_ai_policies WHERE mode=?').bind(mode).first<Policy>();
  if (!row || (mustEnable && (!row.enabled || row.paused_reason))) throw new HttpError(503, 'Eidos AI is paused. Manual editing and downloads remain available.');
  money(row.provider_budget);
  const card = validateRateCard(JSON.parse(row.rate_json));
  if (mustEnable && Date.parse(card.validUntil) <= at.getTime()) throw new HttpError(503, 'The AI rate card needs review.');
  if (mode === 'live') throw new HttpError(503, 'LIVE AI funding has not been approved.');
  return { row, card };
}
/** Server/operator configuration only. No HTTP endpoint accepts this input. Budget is lifetime, across month boundaries. */
export async function configureAiPolicy(env: PlatformEnv, mode: AiMode, card: AiRateCard, providerBudget: number, enabled = false) {
  await ensureAiLedger(env);
  validateRateCard(card); money(providerBudget);
  if (card.evidence === 'owner-approved' && providerBudget > REAL_PROVIDER_TEST_BUDGET)
    throw new HttpError(503, 'The real provider test budget cannot exceed $5.');
  if (mode === 'live' && enabled) throw new HttpError(503, 'LIVE AI funding has not been approved.');
  await db(env).batch([
    db(env).prepare('INSERT INTO eidos_pg_ai_policies(mode,enabled,provider_budget,rate_json,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(mode) DO UPDATE SET enabled=excluded.enabled,provider_budget=excluded.provider_budget,rate_json=excluded.rate_json,updated_at=excluded.updated_at')
      .bind(mode, enabled ? 1 : 0, providerBudget, JSON.stringify(card), new Date().toISOString()),
    journal(env, null, mode, null, 'policy_configured', { version: card.version, providerBudget, enabled }),
  ]);
  return aiPolicyStatus(env, mode);
}
function journal(env: PlatformEnv, owner: string | null, mode: AiMode, action: string | null, event: string, detail: unknown) {
  return db(env).prepare('INSERT INTO eidos_pg_ai_journal(id,owner_id,mode,action_id,event,detail_json,created_at) VALUES(?,?,?,?,?,?,?)')
    .bind(crypto.randomUUID(), owner, mode, action, event, JSON.stringify(detail), new Date().toISOString());
}
export async function setAiLimit(env: PlatformEnv, owner: string, mode: AiMode, requestedLimit: unknown, at = new Date()) {
  await ensureAiLedger(env);
  money(requestedLimit);
  await db(env).batch([
    db(env).prepare('INSERT INTO eidos_pg_ai_accounts(owner_id,mode,requested_limit,updated_at) VALUES(?,?,?,?) ON CONFLICT(owner_id,mode) DO UPDATE SET requested_limit=excluded.requested_limit,updated_at=excluded.updated_at')
      .bind(owner, mode, requestedLimit, at.toISOString()),
    journal(env, owner, mode, null, 'limit_requested', { requestedLimit }),
  ]);
  return aiBalance(env, owner, mode, at);
}
interface Balance { owner_id: string; mode: AiMode; requested_limit: number; frozen: number; available_included: number; available_purchased: number }
const customerCommitted = `CASE WHEN customer_status='held' THEN maximum WHEN customer_status='settled' THEN charged_included+charged_purchased ELSE 0 END`;
const providerCommitted = `CASE WHEN provider_status='held' THEN provider_maximum WHEN provider_status IN ('settled','pessimistic') THEN provider_cost ELSE 0 END`;
// This ceiling is checked inside each atomic write, including after a concurrent
// operator update. Synthetic accounting fixtures retain their existing budgets.
const realBudgetBound = `(json_extract(p.rate_json,'$.evidence')!='owner-approved' OR p.provider_budget<=${REAL_PROVIDER_TEST_BUDGET})`;
/** Operator snapshot. A lower budget blocks new dispatch immediately; existing submitted costs remain obligations. */
export async function aiPolicyStatus(env: PlatformEnv, mode: AiMode) {
  const snapshot = await db(env).prepare(`SELECT p.provider_budget configuredBudget,p.enabled,p.paused_reason pausedReason,
    COALESCE(SUM(CASE WHEN a.provider_status='held' AND a.state='reserved' THEN a.provider_maximum ELSE 0 END),0) reserved,
    COALESCE(SUM(CASE WHEN a.provider_status='held' AND a.state!='reserved' THEN a.provider_maximum ELSE 0 END),0) submitted,
    COALESCE(SUM(CASE WHEN a.provider_status='settled' THEN a.provider_cost ELSE 0 END),0) settled,
    COALESCE(SUM(CASE WHEN a.provider_status='pessimistic' THEN a.provider_cost ELSE 0 END),0) unresolved
    FROM eidos_pg_ai_policies p LEFT JOIN eidos_pg_ai_actions a ON a.mode=p.mode WHERE p.mode=? GROUP BY p.mode`).bind(mode)
    .first<{configuredBudget:number;enabled:number;pausedReason:string|null;reserved:number;submitted:number;settled:number;unresolved:number}>();
  if (!snapshot) throw new HttpError(503, 'AI policy is not configured.');
  const probe = await db(env).prepare("SELECT COALESCE(SUM(json_extract(detail_json,'$.providerHeldMicros')),0) held FROM eidos_pg_ai_journal WHERE mode=? AND event='diagnostic_count_hold'").bind(mode).first<{held:number}>();
  const diagnosticHeld = money(probe?.held || 0);
  const committed = money(snapshot.reserved + snapshot.submitted + snapshot.settled + snapshot.unresolved + diagnosticHeld);
  return { ...snapshot, diagnosticHeld, enabled: Boolean(snapshot.enabled), committed, remaining: Math.max(0, snapshot.configuredBudget - committed),
    overBudget: committed > snapshot.configuredBudget, dispatchBlocked: !snapshot.enabled || Boolean(snapshot.pausedReason) || committed > snapshot.configuredBudget };
}
/** Server-only pricing lookup; browsers cannot provide model or rate parameters. */
export async function activeAiRateCard(env: PlatformEnv, mode: AiMode, at = new Date()) {
  await ensureAiLedger(env);
  return (await policy(env, mode, at)).card;
}
export async function aiBalance(env: PlatformEnv, owner: string, mode: AiMode, at = new Date()) {
  await ensureAiLedger(env);
  const { period, resetsAt } = monthlyPeriod(at);
  const balance = await db(env).prepare('SELECT * FROM eidos_pg_ai_balances WHERE owner_id=? AND mode=?').bind(owner, mode).first<Balance>();
  if (!balance) throw new HttpError(409, 'Set a Monthly AI usage limit before using AI. No default funding is enabled.');
  const usage = await db(env).prepare(`SELECT COALESCE(SUM(CASE WHEN customer_status='held' THEN maximum ELSE 0 END),0) pending, COALESCE(SUM(charged_included),0) included, COALESCE(SUM(charged_purchased),0) purchased FROM eidos_pg_ai_actions WHERE owner_id=? AND mode=? AND period=?`).bind(owner, mode, period).first<{ pending: number; included: number; purchased: number }>();
  const settled = money((usage?.included || 0) + (usage?.purchased || 0)), pending = money(usage?.pending || 0);
  return { currency: 'USD', unitsPerUsd: USD, mode, period, resetsAt, requestedLimit: balance.requested_limit,
    effectiveLimit: Math.max(balance.requested_limit, settled + pending), settled, pending,
    settledIncluded: usage?.included || 0, settledPurchased: usage?.purchased || 0,
    remainingCap: Math.max(0, balance.requested_limit - settled - pending),
    availableIncluded: balance.available_included, availablePurchased: balance.available_purchased, frozen: Boolean(balance.frozen),
    loweringPending: balance.requested_limit < settled + pending,
    notice: 'Counts all AI usage, including included allowance. Package purchases and explicit balance top-ups are separate transactions. Top-ups never change this limit.' };
}
export interface AiBrief { documentId: string; revision: string; scope: 'element' | 'section' | 'page' | 'brand'; targetId: string; instruction: string; context: string }
function validateBrief(input: unknown): AiBrief {
  const fields = ['documentId', 'revision', 'scope', 'targetId', 'instruction', 'context'];
  if (!record(input) || Object.keys(input).some(key => !fields.includes(key)) || fields.some(key => typeof input[key] !== 'string') ||
      !/^[a-zA-Z0-9_-]{8,100}$/.test(String(input.documentId)) || !/^[a-zA-Z0-9_-]{1,100}$/.test(String(input.revision)) ||
      !['element', 'section', 'page', 'brand'].includes(String(input.scope)) || !/^[a-zA-Z0-9_-]{1,100}$/.test(String(input.targetId)) ||
      String(input.instruction).length < 3 || String(input.instruction).length > 2000 || String(input.context).length > 100000)
    throw new HttpError(400, 'Provide a bounded project scope and revision; billing owner, model and totals are server-owned.');
  return structuredClone(input) as unknown as AiBrief;
}
export { validateBrief as validateAiBrief };
export interface AiAction {
  id: string; owner_id: string; mode: AiMode; idempotency_key: string; fingerprint: string;
  document_id: string; revision: string; scope: string; payload_json: string; period: string; resets_at: string;
  rate_json: string; estimate: number; maximum: number; provider_maximum: number; state: string;
  customer_status: string; provider_status: string; held_included: number; held_purchased: number;
  charged_included: number; charged_purchased: number; provider_cost: number; dispatch_token: string | null;
  provider_id: string | null; provider_request_id: string | null; usage_json: string | null; result_json: string | null;
  expires_at: string; submitted_at: string | null; customer_deadline: string | null; settled_at: string | null;
}
export async function ownedAiAction(env: PlatformEnv, owner: string, mode: AiMode, id: string) {
  const action = await db(env).prepare('SELECT * FROM eidos_pg_ai_actions WHERE id=? AND owner_id=? AND mode=?').bind(id, owner, mode).first<AiAction>();
  if (!action) throw new HttpError(404, 'AI action not found.');
  return action;
}
export function publicAiAction(action: AiAction) {
  const rate = validateRateCard(JSON.parse(action.rate_json));
  const brief = validateBrief(JSON.parse(action.payload_json));
  return { id: action.id, state: action.state, documentId: action.document_id, revision: action.revision,
    scope: action.scope, targetId: brief.targetId, estimate: action.estimate, maximum: action.maximum, currency: 'USD', unitsPerUsd: USD,
    model: rate.model, rateCardVersion: rate.version, reasoning: rate.reasoning, processing: rate.processing,
    expiresAt: action.expires_at, resetsAt: action.resets_at, chargedIncluded: action.charged_included,
    chargedPurchased: action.charged_purchased, customerStatus: action.customer_status,
    result: action.result_json ? JSON.parse(action.result_json) : null,
    notice: 'A usable validated proposal is charged when generated, even if discarded. Apply is free. Manual edits are free.' };
}
export async function quoteAiAction(env: PlatformEnv, owner: string, mode: AiMode, idempotencyKey: string, input: unknown, at = new Date()) {
  await ensureAiLedger(env);
  if (!/^[a-zA-Z0-9_-]{8,100}$/.test(idempotencyKey)) throw new HttpError(400, 'Use a unique action reference.');
  const brief = validateBrief(input), payload = JSON.stringify(brief), fingerprint = await hash(payload);
  const previous = await db(env).prepare('SELECT * FROM eidos_pg_ai_actions WHERE owner_id=? AND mode=? AND idempotency_key=?').bind(owner, mode, idempotencyKey).first<AiAction>();
  if (previous) {
    if (previous.fingerprint !== fingerprint) throw new HttpError(409, 'This action reference belongs to a different scope or revision.');
    return previous;
  }
  const { card } = await policy(env, mode, at);
  await aiBalance(env, owner, mode, at);
  // UTF-8 bytes are a conservative text token ceiling. The adapter must include framing/schema in this fixed allowance.
  const inputBound = new TextEncoder().encode(payload).length + 2048;
  if (inputBound > card.maxInputTokens) throw new HttpError(413, 'The selected context exceeds this action limit.');
  const providerBound = providerMaximum(card), maximum = retailCharge(providerBound, card);
  const estimate = retailCharge(ceilingRatio(inputBound, card.inputMicrosPerMillion, 1_000_000) + ceilingRatio(Math.ceil(card.maxOutputTokens / 2), card.outputMicrosPerMillion, 1_000_000), card);
  const { period, resetsAt } = monthlyPeriod(at), id = crypto.randomUUID();
  await db(env).prepare('INSERT OR IGNORE INTO eidos_pg_ai_actions(id,owner_id,mode,idempotency_key,fingerprint,document_id,revision,scope,payload_json,period,resets_at,rate_json,estimate,maximum,provider_maximum,created_at,expires_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)')
    .bind(id, owner, mode, idempotencyKey, fingerprint, brief.documentId, brief.revision, brief.scope, payload, period, resetsAt, JSON.stringify(card), estimate, maximum, providerBound, at.toISOString(), new Date(at.getTime() + 5 * 60 * 1000).toISOString()).run();
  const saved = await db(env).prepare('SELECT * FROM eidos_pg_ai_actions WHERE owner_id=? AND mode=? AND idempotency_key=?').bind(owner, mode, idempotencyKey).first<AiAction>();
  if (!saved || saved.fingerprint !== fingerprint) throw new HttpError(409, 'This action reference belongs to another request.');
  return saved;
}
/** ONE SQLite write statement claims BOTH ledgers. There are no read-then-write budget decisions. */
export async function reserveAiAction(env: PlatformEnv, owner: string, mode: AiMode, id: string, at = new Date()) {
  const action = await ownedAiAction(env, owner, mode, id);
  if (action.state !== 'quoted') return action;
  const { row } = await policy(env, mode, at);
  if (action.rate_json !== row.rate_json) throw new HttpError(409, 'Pricing changed. Request a fresh quote.');
  const period = monthlyPeriod(at).period;
  const result = await db(env).prepare(`UPDATE eidos_pg_ai_actions SET state='reserved',customer_status='held',provider_status='held',
    held_included=MIN(maximum,(SELECT available_included FROM eidos_pg_ai_balances WHERE owner_id=? AND mode=?)),
    held_purchased=maximum-MIN(maximum,(SELECT available_included FROM eidos_pg_ai_balances WHERE owner_id=? AND mode=?))
    WHERE id=? AND owner_id=? AND mode=? AND state='quoted' AND expires_at>? AND period=?
    AND EXISTS(SELECT 1 FROM eidos_pg_ai_balances b WHERE b.owner_id=? AND b.mode=? AND b.frozen=0 AND b.available_included>=0 AND b.available_purchased>=0
      AND b.available_included+b.available_purchased>=maximum
      AND b.requested_limit >= maximum+(SELECT COALESCE(SUM(${customerCommitted}),0) FROM eidos_pg_ai_actions WHERE owner_id=? AND mode=? AND period=?))
    AND NOT EXISTS(SELECT 1 FROM eidos_pg_ai_grants g JOIN eidos_orders o ON o.id=g.order_id WHERE g.owner_id=eidos_pg_ai_actions.owner_id AND g.mode=eidos_pg_ai_actions.mode AND (o.status!='paid' OR g.reversed=1 OR EXISTS(SELECT 1 FROM eidos_revoked_payments r WHERE r.payment_intent=g.payment_intent)))
    AND EXISTS(SELECT 1 FROM eidos_pg_ai_policies p WHERE p.mode=? AND p.enabled=1 AND p.paused_reason IS NULL AND p.rate_json=?
      AND ${realBudgetBound} AND p.provider_budget >= provider_maximum+(SELECT COALESCE(SUM(${providerCommitted}),0) FROM eidos_pg_ai_actions WHERE mode=?)) RETURNING id`)
    .bind(owner, mode, owner, mode, id, owner, mode, at.toISOString(), period, owner, mode, owner, mode, period, mode, action.rate_json, mode).first();
  if (!result) {
    const current = await ownedAiAction(env, owner, mode, id);
    if (current.state !== 'quoted') return current;
    throw new HttpError(409, 'AI allowance, monthly limit, owner budget or quote changed. Manual editing remains available.');
  }
  return ownedAiAction(env, owner, mode, id);
}
/** The caller may submit only when it receives this one-time token. A crash after the claim is UNKNOWN, never retryable. */
export async function claimAiDispatch(env: PlatformEnv, owner: string, mode: AiMode, id: string, at = new Date()) {
  await ownedAiAction(env, owner, mode, id);
  await policy(env, mode, at);
  const token = crypto.randomUUID();
  const claimed = await db(env).prepare(`UPDATE eidos_pg_ai_actions SET state='submitted',dispatch_token=?,submitted_at=?,customer_deadline=?
    WHERE id=? AND owner_id=? AND mode=? AND state='reserved' AND expires_at>? AND period=?
    AND EXISTS(SELECT 1 FROM eidos_pg_ai_accounts a WHERE a.owner_id=? AND a.mode=? AND a.frozen=0)
    AND NOT EXISTS(SELECT 1 FROM eidos_pg_ai_grants g JOIN eidos_orders o ON o.id=g.order_id WHERE g.owner_id=eidos_pg_ai_actions.owner_id AND g.mode=eidos_pg_ai_actions.mode AND (o.status!='paid' OR g.reversed=1 OR EXISTS(SELECT 1 FROM eidos_revoked_payments r WHERE r.payment_intent=g.payment_intent)))
    AND EXISTS(SELECT 1 FROM eidos_pg_ai_policies p WHERE p.mode=? AND p.enabled=1 AND p.paused_reason IS NULL AND p.rate_json=eidos_pg_ai_actions.rate_json
      AND ${realBudgetBound} AND p.provider_budget >= (SELECT COALESCE(SUM(${providerCommitted}),0) FROM eidos_pg_ai_actions WHERE mode=?)) RETURNING id`)
    .bind(token, at.toISOString(), new Date(at.getTime() + CUSTOMER_HOLD_WINDOW_MS).toISOString(), id, owner, mode, at.toISOString(), monthlyPeriod(at).period, owner, mode, mode, mode).first();
  return claimed ? token : null;
}
/** Zero-priced input counting is bounded and idempotent even though no money is reserved yet. */
export interface AiCountClaim { action_id: string; client_request_id: string; request_hash: string; state: 'submitted'|'counted'|'failed' }
export async function claimAiInputCount(env: PlatformEnv, owner: string, mode: AiMode, id: string, requestJson: string, at = new Date()) {
  const action = await ownedAiAction(env, owner, mode, id);
  if (typeof requestJson !== 'string' || new TextEncoder().encode(requestJson).length > 100000) throw new HttpError(413, 'The provider input is too large.');
  const requestHash = await hash(requestJson), clientId = crypto.randomUUID(), hourStart = new Date(at.getTime()-60*60*1000).toISOString();
  const claimed = await db(env).prepare(`INSERT INTO eidos_pg_ai_count_claims(action_id,owner_id,mode,request_hash,request_json,client_request_id,state,created_at)
    SELECT a.id,a.owner_id,a.mode,?,?,?,'submitted',? FROM eidos_pg_ai_actions a
    WHERE a.id=? AND a.owner_id=? AND a.mode=? AND a.state='quoted' AND a.dispatch_token IS NULL AND a.expires_at>? AND a.period=?
      AND EXISTS(SELECT 1 FROM eidos_pg_ai_balances b WHERE b.owner_id=a.owner_id AND b.mode=a.mode AND b.frozen=0 AND b.available_included>=0 AND b.available_purchased>=0
        AND b.available_included+b.available_purchased>=a.maximum AND b.requested_limit>=a.maximum+
          (SELECT COALESCE(SUM(${customerCommitted}),0) FROM eidos_pg_ai_actions WHERE owner_id=a.owner_id AND mode=a.mode AND period=a.period))
      AND NOT EXISTS(SELECT 1 FROM eidos_pg_ai_grants g JOIN eidos_orders o ON o.id=g.order_id WHERE g.owner_id=a.owner_id AND g.mode=a.mode
        AND (o.status!='paid' OR g.reversed=1 OR EXISTS(SELECT 1 FROM eidos_revoked_payments r WHERE r.payment_intent=g.payment_intent)))
      AND EXISTS(SELECT 1 FROM eidos_pg_ai_policies p WHERE p.mode=a.mode AND p.enabled=1 AND p.paused_reason IS NULL AND p.rate_json=a.rate_json
        AND ${realBudgetBound} AND p.provider_budget>=a.provider_maximum+(SELECT COALESCE(SUM(${providerCommitted}),0) FROM eidos_pg_ai_actions WHERE mode=a.mode))
      AND (SELECT COUNT(*) FROM eidos_pg_ai_count_claims WHERE mode=a.mode)<${AI_COUNT_LIMITS.lifetime}
      AND (SELECT COUNT(*) FROM eidos_pg_ai_count_claims WHERE mode=a.mode AND created_at>?)<${AI_COUNT_LIMITS.globalHour}
      AND (SELECT COUNT(*) FROM eidos_pg_ai_count_claims WHERE mode=a.mode AND owner_id=a.owner_id AND created_at>?)<${AI_COUNT_LIMITS.perOwnerHour}
      AND NOT EXISTS(SELECT 1 FROM eidos_pg_ai_count_claims WHERE action_id=a.id)
    ON CONFLICT(action_id) DO NOTHING RETURNING action_id`)
    .bind(requestHash, requestJson, clientId, at.toISOString(), id, owner, mode, at.toISOString(), monthlyPeriod(at).period, hourStart, hourStart).first();
  if (claimed) return { action_id: id, client_request_id: clientId, request_hash: requestHash, state: 'submitted' } satisfies AiCountClaim;
  const existing = await db(env).prepare('SELECT * FROM eidos_pg_ai_count_claims WHERE action_id=? AND owner_id=? AND mode=?').bind(id, owner, mode).first<AiCountClaim>();
  if (existing && existing.request_hash !== requestHash) throw new HttpError(409, 'This counting reference belongs to a different frozen request.');
  if (existing) throw new HttpError(409, existing.state === 'submitted' ? 'This full-input quote is still being prepared. Do not submit another count.' : 'This counting attempt is already recorded. Use a fresh action reference.');
  // Neither funds nor quota can be bypassed by opening another project/device.
  if (action.state !== 'quoted') throw new HttpError(409, 'This action cannot request another input count.');
  throw new HttpError(409, 'AI allowance, monthly limit, owner budget or counting quota blocks this quote. Manual editing remains available.');
}
export async function finishAiInputCount(env: PlatformEnv, owner: string, mode: AiMode, id: string, clientId: string,
  evidence: { success: boolean; providerRequestId?: string; inputTokens?: number; reason?: string }, at = new Date()) {
  if (evidence.providerRequestId !== undefined && !/^[!-~]{1,200}$/.test(evidence.providerRequestId)) throw new HttpError(409, 'Invalid counting identity.');
  if (evidence.inputTokens !== undefined) boundedInteger(evidence.inputTokens,0,10_000_000);
  await db(env).prepare(`UPDATE eidos_pg_ai_count_claims SET state=?,provider_request_id=?,input_tokens=?,failure_reason=?,completed_at=?
    WHERE action_id=? AND owner_id=? AND mode=? AND client_request_id=? AND state='submitted'`)
    .bind(evidence.success?'counted':'failed', evidence.providerRequestId || null, evidence.inputTokens ?? null, evidence.reason?.slice(0,100) || null,
      at.toISOString(),id,owner,mode,clientId).run();
}
export interface AiUsage {
  inputTokens: number; outputTokens: number; calls: number; paidTools: number;
  cachedInputTokens?: number; cacheWriteTokens?: number; reasoningTokens?: number;
}
export function meterAiUsage(usage: AiUsage, card: AiRateCard) {
  boundedInteger(usage.inputTokens, 0, 10_000_000); boundedInteger(usage.outputTokens, 0, 10_000_000);
  boundedInteger(usage.calls, 1, 100); boundedInteger(usage.paidTools, 0, 100);
  const cached = boundedInteger(usage.cachedInputTokens ?? 0, 0, usage.inputTokens), written = boundedInteger(usage.cacheWriteTokens ?? 0, 0, usage.inputTokens);
  boundedInteger(usage.reasoningTokens ?? 0, 0, usage.outputTokens);
  if (cached + written > usage.inputTokens || ((cached || written) && (card.cachedInputMicrosPerMillion === undefined || card.cacheWriteMicrosPerMillion === undefined)))
    throw new HttpError(503, 'Provider input categories cannot be metered with this rate card.');
  return money(ceilingRatio(usage.inputTokens - cached - written, card.inputMicrosPerMillion, 1_000_000) +
    ceilingRatio(cached, card.cachedInputMicrosPerMillion || 0, 1_000_000) + ceilingRatio(written, card.cacheWriteMicrosPerMillion || 0, 1_000_000) +
    ceilingRatio(usage.outputTokens, card.outputMicrosPerMillion, 1_000_000));
}
/** Persist identity on receipt even when the body or usage is incomplete. Never release either ledger hold. */
export async function recordAiProviderReceipt(env: PlatformEnv, owner: string, mode: AiMode, id: string, dispatchToken: string,
  receipt: { providerId?: string; providerRequestId?: string }) {
  const action = await ownedAiAction(env, owner, mode, id);
  const valid = (value: unknown) => typeof value === 'string' && /^[!-~]{1,200}$/.test(value);
  if (action.dispatch_token !== dispatchToken || (!receipt.providerId && !receipt.providerRequestId) ||
      (receipt.providerId !== undefined && !valid(receipt.providerId)) || (receipt.providerRequestId !== undefined && !valid(receipt.providerRequestId)))
    throw new HttpError(409, 'Provider receipt does not match this dispatched action.');
  const providerId = receipt.providerId ?? null, requestId = receipt.providerRequestId ?? null;
  const claimed = await db(env).prepare(`UPDATE eidos_pg_ai_actions SET provider_id=COALESCE(provider_id,?),provider_request_id=COALESCE(provider_request_id,?)
    WHERE id=? AND owner_id=? AND mode=? AND dispatch_token=? AND provider_status IN ('held','pessimistic')
      AND (? IS NULL OR provider_id IS NULL OR provider_id=?) AND (? IS NULL OR provider_request_id IS NULL OR provider_request_id=?) RETURNING id`)
    .bind(providerId, requestId, id, owner, mode, dispatchToken, providerId, providerId, requestId, requestId).first();
  if (!claimed) throw new HttpError(409, 'Provider receipt does not match this dispatched action.');
  return ownedAiAction(env, owner, mode, id);
}
export async function settleAiAction(env: PlatformEnv, owner: string, mode: AiMode, id: string, evidence: {
  dispatchToken: string; providerId: string; providerRequestId: string; usage: AiUsage; usable: boolean; result?: unknown;
}, at = new Date()) {
  await reconcileAiDeadlines(env, at);
  const action = await ownedAiAction(env, owner, mode, id);
  if (action.dispatch_token !== evidence.dispatchToken || typeof evidence.providerId !== 'string' || !evidence.providerId || evidence.providerId.length > 200 || typeof evidence.providerRequestId !== 'string' || !evidence.providerRequestId || evidence.providerRequestId.length > 200 ||
      (action.provider_id !== null && action.provider_id !== evidence.providerId) || (action.provider_request_id !== null && action.provider_request_id !== evidence.providerRequestId))
    throw new HttpError(409, 'Provider evidence does not match this dispatched action.');
  if (action.provider_status === 'settled') return action;
  // Bind the first provider identity atomically before accepting usage. A later
  // reconciliation may resolve its cost, but cannot substitute another request.
  // This separate claim never releases either hold, including after a crash.
  const identity = await db(env).prepare(`UPDATE eidos_pg_ai_actions SET provider_id=COALESCE(provider_id,?),provider_request_id=COALESCE(provider_request_id,?)
    WHERE id=? AND owner_id=? AND mode=? AND dispatch_token=? AND provider_status IN ('held','pessimistic')
      AND (provider_id IS NULL OR provider_id=?) AND (provider_request_id IS NULL OR provider_request_id=?) RETURNING id`)
    .bind(evidence.providerId, evidence.providerRequestId, id, owner, mode, evidence.dispatchToken, evidence.providerId, evidence.providerRequestId).first();
  if (!identity) {
    const current = await ownedAiAction(env, owner, mode, id);
    if (current.dispatch_token === evidence.dispatchToken && current.provider_id === evidence.providerId &&
        current.provider_request_id === evidence.providerRequestId && current.provider_status === 'settled') return current;
    throw new HttpError(409, 'Provider evidence does not match this dispatched action.');
  }
  const card = validateRateCard(JSON.parse(action.rate_json));
  let unboundedExpansion: boolean, providerCost: number, exceeded: boolean;
  try {
    unboundedExpansion = evidence.usage.calls > card.maxCalls || evidence.usage.paidTools > card.paidTools;
    providerCost = Math.max(meterAiUsage(evidence.usage, card), unboundedExpansion ? action.provider_maximum : 0);
    exceeded = providerCost > action.provider_maximum || evidence.usage.inputTokens > card.maxInputTokens || evidence.usage.outputTokens > card.maxOutputTokens || unboundedExpansion;
  } catch {
    // Invalid or unsupported usage is unknown liability, never zero expense.
    // Retain the existing hold/pessimistic amount and customer waiver, and stop
    // new dispatch before rejecting the evidence. Only matching, metered later
    // evidence may resolve this liability; the policy remains paused for review.
    await db(env).batch([
      db(env).prepare('UPDATE eidos_pg_ai_policies SET enabled=0,paused_reason=?,updated_at=? WHERE mode=?')
        .bind('Unmeterable provider usage; reconciliation required.', at.toISOString(), mode),
      db(env).prepare(`UPDATE eidos_pg_ai_actions SET state=CASE WHEN customer_status='waived' THEN 'technical_waiver' ELSE 'reconciliation_pending' END,failure_reason='unmeterable_provider_usage'
        WHERE id=? AND owner_id=? AND mode=? AND dispatch_token=? AND provider_id=? AND provider_request_id=? AND provider_status IN ('held','pessimistic')`)
        .bind(id, owner, mode, evidence.dispatchToken, evidence.providerId, evidence.providerRequestId),
      journal(env, owner, mode, id, 'unmeterable_provider_usage', { liability: 'retained', policyPaused: true }),
    ]);
    throw new HttpError(503, 'Provider usage could not be bounded. Eidos AI is paused; this action requires reconciliation.');
  }
  // A trusted adapter validates proposal operations. The ledger independently
  // enforces serializable, bounded storage before allowing any customer debit.
  let usable = evidence.usable && !unboundedExpansion, invalidResult = false;
  let resultJson: string | null = null;
  if (usable) {
    try {
      const serialized = JSON.stringify(evidence.result ?? null);
      if (typeof serialized !== 'string' || serialized.length > 100000) invalidResult = true;
      else resultJson = serialized;
    } catch { invalidResult = true; }
    if (invalidResult) usable = false;
  }
  // Price only up to the approved provider bound. Actual provider expense stays
  // uncapped, while even an extreme retail ratio cannot overflow before the cap.
  const charge = usable && action.customer_status === 'held' ? Math.min(action.maximum, retailCharge(Math.min(providerCost, action.provider_maximum), card)) : 0;
  const included = Math.min(action.held_included, charge), purchased = charge - included;
  await db(env).batch([
    db(env).prepare(`UPDATE eidos_pg_ai_actions SET state=CASE WHEN customer_status='waived' OR ?=0 THEN 'technical_waiver' ELSE 'settled' END,
      customer_status=CASE WHEN customer_status='waived' OR ?=0 THEN 'waived' ELSE 'settled' END,
      provider_status=?,charged_included=CASE WHEN customer_status='held' THEN ? ELSE 0 END,charged_purchased=CASE WHEN customer_status='held' THEN ? ELSE 0 END,provider_cost=?,provider_id=?,provider_request_id=?,usage_json=?,result_json=?,settled_at=?,failure_reason=?
      WHERE id=? AND owner_id=? AND mode=? AND dispatch_token=? AND provider_status IN ('held','pessimistic')`)
      .bind(usable ? 1 : 0, usable ? 1 : 0, unboundedExpansion ? 'pessimistic' : 'settled', included, purchased, providerCost, evidence.providerId, evidence.providerRequestId, JSON.stringify(evidence.usage), resultJson, at.toISOString(), unboundedExpansion ? 'unbounded_expansion_requires_provider_reconciliation' : invalidResult ? 'invalid_proposal_result' : exceeded ? 'provider_bound_exceeded' : usable ? null : 'unusable_proposal', id, owner, mode, evidence.dispatchToken),
    ...(exceeded ? [db(env).prepare('UPDATE eidos_pg_ai_policies SET enabled=0,paused_reason=?,updated_at=? WHERE mode=?').bind('Provider bound exceeded; Eidos absorbs excess. Review required.', at.toISOString(), mode)] : []),
    journal(env, owner, mode, id, 'settlement_evidence', { providerCost, charge, usable, invalidResult, exceeded }),
  ]);
  return ownedAiAction(env, owner, mode, id);
}
export async function markAiUnknown(env: PlatformEnv, owner: string, mode: AiMode, id: string, reason: string) {
  await ownedAiAction(env, owner, mode, id);
  await db(env).prepare("UPDATE eidos_pg_ai_actions SET state='reconciliation_pending',failure_reason=? WHERE id=? AND owner_id=? AND mode=? AND state='submitted' AND provider_status='held'")
    .bind(reason.slice(0, 200), id, owner, mode).run();
  return ownedAiAction(env, owner, mode, id);
}
/** Reserved/quoted state proves no dispatch claim occurred; submitted/unknown must retain its provider hold. */
export async function cancelAiAction(env: PlatformEnv, owner: string, mode: AiMode, id: string) {
  await ownedAiAction(env, owner, mode, id);
  await db(env).prepare("UPDATE eidos_pg_ai_actions SET state='cancelled',customer_status='waived',provider_status='none',failure_reason='cancelled_before_dispatch' WHERE id=? AND owner_id=? AND mode=? AND state IN ('quoted','reserved') AND dispatch_token IS NULL")
    .bind(id, owner, mode).run();
  return ownedAiAction(env, owner, mode, id);
}
/** Run by maintenance and account reads. No lease expiry releases upstream budget. */
export async function reconcileAiDeadlines(env: PlatformEnv, at = new Date()) {
  await ensureAiLedger(env);
  await db(env).batch([
    db(env).prepare("UPDATE eidos_pg_ai_actions SET state='cancelled',customer_status='waived',provider_status='none',failure_reason='expired_before_dispatch' WHERE state IN ('quoted','reserved') AND dispatch_token IS NULL AND expires_at<=?").bind(at.toISOString()),
    db(env).prepare("UPDATE eidos_pg_ai_actions SET state='technical_waiver',customer_status='waived',provider_status='pessimistic',provider_cost=provider_maximum,settled_at=?,failure_reason='customer_deadline_waiver_provider_maximum_retained' WHERE provider_status='held' AND dispatch_token IS NOT NULL AND customer_deadline<=?").bind(at.toISOString(), at.toISOString()),
  ]);
}

/** Trusted commerce policy: only verified paid immutable orders can mint credits, once per order/version/kind. */
export interface AiGrantRule { productId: string; editionId: string; version: string; priceCents: number; grantVersion: string; kind: 'included' | 'purchased'; amount: number }
/** Only called by the signed existing commerce webhook after its exact order/mode/amount checks and event journal commit. */
export async function recordVerifiedAiPayment(env: PlatformEnv, orderId: string, eventId: string, mode: AiMode, verifiedMerchantId: string) {
  if (env.EIDOS_PLAYGROUND_AI_MODE !== 'test') return;
  if (mode !== 'test' || !/^acct_[a-zA-Z0-9]+$/.test(verifiedMerchantId) || verifiedMerchantId !== env.EIDOS_TEMPLATE_STRIPE_ACCOUNT_ID)
    throw new HttpError(503, 'The isolated AI payment merchant is not configured.');
  await ensureAiLedger(env);
  await db(env).prepare(`INSERT OR IGNORE INTO eidos_pg_ai_paid_receipts(order_id,mode,merchant_id,event_id,currency,amount_cents,payment_intent,created_at)
    SELECT o.id,?,?,?,'usd',t.price_cents,o.payment_intent,? FROM eidos_orders o JOIN eidos_template_orders t ON t.order_id=o.id
    JOIN eidos_pg_ai_checkout_merchants m ON m.order_id=o.id AND m.mode=? AND m.merchant_id=?
    WHERE o.id=? AND o.status='paid' AND t.fulfillment_status='ready' AND t.member_id IS NOT NULL AND substr(o.session_id,1,8)='cs_test_'
      AND EXISTS(SELECT 1 FROM eidos_stripe_events WHERE id=?) AND NOT EXISTS(SELECT 1 FROM eidos_revoked_payments WHERE payment_intent=o.payment_intent)`)
    .bind(mode, verifiedMerchantId, eventId, new Date().toISOString(), mode, verifiedMerchantId, orderId, eventId).run();
}
export async function grantPaidAiBalance(env: PlatformEnv, owner: string, mode: AiMode, orderId: string, rule: AiGrantRule) {
  await ensureAiLedger(env); money(rule.amount, true); boundedInteger(rule.priceCents, 1, 10_000_000);
  if (mode !== 'test') throw new HttpError(503, 'LIVE AI funding has not been approved.');
  // The existing signed webhook owns paid status. Caller cannot grant by passing a claimed payment event.
  const result = await db(env).prepare(`INSERT OR IGNORE INTO eidos_pg_ai_grants(id,owner_id,mode,order_id,grant_version,kind,amount,payment_intent,created_at)
    SELECT ?,t.member_id,?,o.id,?,?,?,o.payment_intent,? FROM eidos_orders o JOIN eidos_template_orders t ON t.order_id=o.id
    JOIN eidos_pg_ai_accounts a ON a.owner_id=t.member_id AND a.mode=?
    JOIN eidos_pg_ai_paid_receipts p ON p.order_id=o.id AND p.mode=a.mode AND p.payment_intent=o.payment_intent AND p.currency='usd' AND p.amount_cents=t.price_cents AND p.merchant_id=?
    WHERE o.id=? AND t.member_id=? AND o.status='paid' AND t.fulfillment_status='ready'
      AND substr(o.session_id,1,8)=? AND a.frozen=0
      AND t.product_id=? AND t.edition_id=? AND t.version=? AND t.price_cents=?
      AND NOT EXISTS(SELECT 1 FROM eidos_revoked_payments r WHERE r.payment_intent=o.payment_intent)`)
    .bind(crypto.randomUUID(), mode, rule.grantVersion, rule.kind, rule.amount, new Date().toISOString(), mode, env.EIDOS_TEMPLATE_STRIPE_ACCOUNT_ID || '', orderId, owner,
      mode === 'test' ? 'cs_test_' : 'cs_live_', rule.productId, rule.editionId, rule.version, rule.priceCents).run();
  const existing = await db(env).prepare('SELECT * FROM eidos_pg_ai_grants WHERE order_id=? AND grant_version=? AND kind=? AND owner_id=? AND mode=?').bind(orderId, rule.grantVersion, rule.kind, owner, mode).first<{ amount: number; reversed: number }>();
  if (!existing || existing.amount !== rule.amount) throw new HttpError(409, 'A matching verified paid order is required for this allowance.');
  return { granted: Boolean(result.meta.changes), reversed: Boolean(existing.reversed) };
}
/** Invoked after the existing commerce revocation transaction; admission also checks source status atomically. */
export async function reconcileAiReversals(env: PlatformEnv) {
  await ensureAiLedger(env);
  const database = db(env);
  // One transaction freezes affected accounts before removing currently unspent funds. Held/consumed credits never create negative balances.
  const invalid = `EXISTS(SELECT 1 FROM eidos_orders o WHERE o.id=g.order_id AND o.status='refunded') OR EXISTS(SELECT 1 FROM eidos_revoked_payments r WHERE r.payment_intent=g.payment_intent)`;
  await database.batch([
    database.prepare(`UPDATE eidos_pg_ai_accounts SET frozen=1 WHERE EXISTS(SELECT 1 FROM eidos_pg_ai_grants g WHERE g.owner_id=eidos_pg_ai_accounts.owner_id AND g.mode=eidos_pg_ai_accounts.mode AND (${invalid}))`),
    // Iteration per grant prevents multiple reversed grants from subtracting the same available amount.
  ]);
  const grants = await database.prepare(`SELECT id FROM eidos_pg_ai_grants g WHERE reversed=0 AND (${invalid})`).all<{ id: string }>();
  for (const grant of grants.results) await database.prepare(`UPDATE eidos_pg_ai_grants SET reversed=1,
    revoked_amount=MIN(amount,MAX(0,(SELECT CASE WHEN kind='included' THEN available_included ELSE available_purchased END FROM eidos_pg_ai_balances b WHERE b.owner_id=eidos_pg_ai_grants.owner_id AND b.mode=eidos_pg_ai_grants.mode))) WHERE id=? AND reversed=0`).bind(grant.id).run();
}
