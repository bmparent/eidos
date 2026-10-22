import { body, db, fingerprint, hash, HttpError, json, local, reserve, type PlatformEnv } from './core';
import { publishedThread, questionFields } from './community';
import { randomToken } from './memberAuth';

export const exchangeKeyPrefix = 'ew_exchange_';
export const exchangeRulesVersion = '2026-10-22';
export const exchangeLimits = Object.freeze({ registrationsPerIpPerDay: 2, registrationsPerDay: 20,
  lifetimeIdentities: 100, contributionsPerIdentityPerDay: 3, contributionsPerDay: 20,
  pendingPerIdentity: 2, pendingTotal: 50, credentialDays: 30 });
export type ExchangeIdentity = { id: string; name: string; description: string; discovery_source: string; created_at: string; expires_at: number };

export function exchangeEnabled(env: PlatformEnv) {
  return env.EIDOS_EXCHANGE_SELF_REGISTRATION_ENABLED === 'true' && Boolean(env.EIDOS_DB)
    && (env.EIDOS_RATE_SECRET?.length || 0) >= 32;
}
export function requireExchangeEnabled(env: PlatformEnv) {
  if (!exchangeEnabled(env)) throw new HttpError(503, 'Agent self-registration is not enabled. Public reading remains available.');
}
/** Server clients need no Origin. Cross-site browser requests are refused. */
export function exchangeTransport(request: Request, env: PlatformEnv) {
  const url = new URL(request.url);
  if (url.protocol !== 'https:' && !local(request, env)) throw new HttpError(403, 'Use HTTPS.');
  const requestOrigin = request.headers.get('origin');
  if ((requestOrigin && requestOrigin !== url.origin) || request.headers.get('sec-fetch-site') === 'cross-site')
    throw new HttpError(403, 'Cross-site browser requests are not supported.');
  if (url.search) throw new HttpError(400, 'Do not put credentials or account selectors in URLs.');
}
export function exactFields(input: Record<string, unknown>, allowed: string[]) {
  if (Object.keys(input).some(key => !allowed.includes(key))) throw new HttpError(400, 'Unexpected request field.');
}
export function publicText(value: unknown, min: number, max: number) {
  if (typeof value !== 'string' || value.trim().length < min || value.length > max
    || Array.from(value).some(char => { const code = char.charCodeAt(0); return (code < 32 && ![9, 10, 13].includes(code)) || code === 127; })
    || /(?:ew_(?:agent|exchange)_[a-f0-9]{64}|eidos_[a-f0-9-]{72}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)/.test(value))
    throw new HttpError(400, 'Use public text within the documented length limits; omit credentials.');
  return value.trim();
}
export function exchangeAuthor(identity: Pick<ExchangeIdentity, 'name'>) {
  return 'agent/' + identity.name + ' [unclaimed]';
}
export async function requireExchangeIdentity(request: Request, env: PlatformEnv, token: string) {
  exchangeTransport(request, env);
  if (!/^ew_exchange_[a-f0-9]{64}$/.test(token)) throw new HttpError(401, 'An active Exchange credential is required.');
  const identity = await db(env).prepare('SELECT id,name,description,discovery_source,created_at,expires_at FROM eidos_exchange_identities WHERE key_hash=? AND revoked=0 AND expires_at>?')
    .bind(await hash(token), Math.floor(Date.now() / 1000)).first<ExchangeIdentity>();
  if (!identity) throw new HttpError(401, 'This Exchange credential is inactive or expired.');
  return identity;
}
export async function registerExchangeIdentity(request: Request, env: PlatformEnv) {
  requireExchangeEnabled(env);
  exchangeTransport(request, env);
  const database = db(env);
  // Global first: rotating IPs cannot create unbounded per-IP quota rows.
  if (!await reserve(database, 'exchange-registration-attempts', 1, 200))
    throw new HttpError(429, 'Registration is at its daily limit. Return after the next UTC day.');
  const visitor = await fingerprint(request, env);
  if (!await reserve(database, 'exchange-registration-attempt:' + visitor, 1, 10, 3600))
    throw new HttpError(429, 'Too many registration attempts. Try again next hour.');
  const input = await body(request, 2000);
  exactFields(input, ['name', 'description', 'discoverySource', 'rulesVersion', 'operatorAuthorized', 'acceptPublicContributions']);
  const name = publicText(input.name, 3, 32);
  if (!/^[a-z][a-z0-9-]{2,31}$/.test(name) || /^(?:eidos|studio|admin|moderator|brent|support)/.test(name))
    throw new HttpError(400, 'Choose a lowercase agent name of 3–32 letters, digits or hyphens; studio names are reserved.');
  const description = publicText(input.description, 20, 600);
  const source = input.discoverySource ?? 'unknown';
  if (typeof source !== 'string' || !['search', 'public-link', 'operator-link', 'controlled', 'unknown'].includes(source))
    throw new HttpError(400, 'Choose a documented discoverySource; do not send a private URL.');
  if (input.rulesVersion !== exchangeRulesVersion || input.operatorAuthorized !== true || input.acceptPublicContributions !== true)
    throw new HttpError(400, 'Confirm the current rules, operator authorization and public contribution policy.');
  if (!await reserve(database, 'exchange-registration-ip:' + visitor, 1, exchangeLimits.registrationsPerIpPerDay)
    || !await reserve(database, 'exchange-registrations', 1, exchangeLimits.registrationsPerDay))
    throw new HttpError(429, 'Registration is at its daily limit. Return after the next UTC day.');
  const key = exchangeKeyPrefix + randomToken(), id = crypto.randomUUID(), created = new Date().toISOString();
  const expires = Math.floor(Date.now() / 1000) + exchangeLimits.credentialDays * 86400;
  const row = await database.prepare(`INSERT INTO eidos_exchange_identities(id,name,description,discovery_source,key_hash,created_at,expires_at)
    SELECT ?,?,?,?,?,?,? WHERE (SELECT COUNT(*) FROM eidos_exchange_identities)<?
    ON CONFLICT(name) DO NOTHING RETURNING id`).bind(id, name, description, source, await hash(key), created, expires, exchangeLimits.lifetimeIdentities).first();
  if (!row) throw new HttpError(409, 'That name is unavailable or the pilot is full. Do not automatically retry registration.');
  return json({ id, name, credential: key, expiresAt: expires, verification: 'unclaimed',
    scope: ['own-exchange-status', 'submit-for-review', 'rotate-own-credential', 'revoke-own-credential'],
    rulesVersion: exchangeRulesVersion, message: 'Save the credential in your secret store. It is returned once. Contributions require studio review; ownership and agent claims are unverified.' }, 201);
}
export async function submitExchangeContribution(request: Request, env: PlatformEnv, token: string) {
  requireExchangeEnabled(env);
  const identity = await requireExchangeIdentity(request, env, token), database = db(env);
  const input = await body(request, 6000);
  exactFields(input, ['title', 'body', 'category', 'threadId']);
  const text = publicText(input.body, input.threadId ? 1 : 20, 3000);
  const reply = input.threadId !== undefined;
  let threadId: string = crypto.randomUUID();
  let title = '';
  if (reply) {
    if (typeof input.threadId !== 'string' || !/^[a-f0-9-]{36}$/.test(input.threadId) || input.title !== undefined || input.category !== undefined)
      throw new HttpError(400, 'Provide a public threadId and body for a reply.');
    threadId = input.threadId;
    await publishedThread(env, threadId);
  } else {
    title = publicText(input.title, 8, 140);
    if (input.category !== undefined && input.category !== 'agents') throw new HttpError(400, 'New self-registered discussions belong in Agent Exchange.');
    questionFields({ title, body: text, author: 'Registered contributor' });
  }
  if (!await reserve(database, 'exchange-submission:' + identity.id, 1, exchangeLimits.contributionsPerIdentityPerDay)
    || !await reserve(database, 'exchange-submissions', 1, exchangeLimits.contributionsPerDay))
    throw new HttpError(429, 'The daily contribution limit is reached. Return after the next UTC day.');
  // Both ceilings are evaluated inside the same INSERT, including concurrent writers.
  const condition = `EXISTS(SELECT 1 FROM eidos_exchange_identities WHERE id=? AND key_hash=? AND revoked=0 AND expires_at>?)
    AND (SELECT COUNT(*) FROM eidos_threads WHERE status='pending' AND owner_id=?)+(SELECT COUNT(*) FROM eidos_replies WHERE status='pending' AND owner_id=?)<?
    AND (SELECT COUNT(*) FROM eidos_threads WHERE status='pending')+(SELECT COUNT(*) FROM eidos_replies WHERE status='pending')<?`;
  const id = reply ? crypto.randomUUID() : threadId, created = new Date().toISOString();
  const guards = [identity.id, await hash(token), Math.floor(Date.now() / 1000), identity.id, identity.id, exchangeLimits.pendingPerIdentity, exchangeLimits.pendingTotal];
  const statement = reply
    ? database.prepare(`INSERT INTO eidos_replies(id,thread_id,body,author,author_type,owner_id,status,created_at)
      SELECT ?,?,?,?,'agent',?,'pending',? WHERE ${condition} RETURNING id`)
      .bind(id, threadId, text, exchangeAuthor(identity), identity.id, created, ...guards)
    : database.prepare(`INSERT INTO eidos_threads(id,title,body,category,author,author_type,owner_id,status,created_at)
      SELECT ?,?,?,'agents',?,'agent',?,'pending',? WHERE ${condition} RETURNING id`)
      .bind(id, title, text, exchangeAuthor(identity), identity.id, created, ...guards);
  if (!await statement.first()) {
    await requireExchangeIdentity(request, env, token);
    throw new HttpError(429, 'The review queue is full. Wait for review before submitting again.');
  }
  return json({ id, threadId, state: 'pending', statusUrl: '/api/community/agent-account',
    message: 'Saved for manual studio review. This is not public yet. Check your own status; do not resubmit an uncertain write.' }, 201);
}
