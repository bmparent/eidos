import { body, db, guarded, hash, HttpError, json, reserve } from '../../_shared/platform/core';
import { exactFields, exchangeAuthor, exchangeKeyPrefix, requireExchangeIdentity } from '../../_shared/platform/exchangeIdentity';
import { randomToken } from '../../_shared/platform/memberAuth';

function bearer(request: Request) { return (request.headers.get('authorization') || '').replace(/^Bearer /, ''); }
export const onRequestGet = guarded(async ({ request, env }) => {
  const identity = await requireExchangeIdentity(request, env, bearer(request)), database = db(env);
  if (!await reserve(database, 'exchange-status:' + identity.id, 1, 60, 3600)) throw new HttpError(429, 'Check status at most once every five minutes.');
  const contributions = await database.prepare(`SELECT id,'thread' AS kind,id AS threadId,status,created_at FROM eidos_threads WHERE owner_id=?
    UNION ALL SELECT id,'reply' AS kind,thread_id AS threadId,status,created_at FROM eidos_replies WHERE owner_id=? ORDER BY created_at DESC LIMIT 50`)
    .bind(identity.id, identity.id).all<{ id: string; kind: string; threadId: string; status: string; created_at: string }>();
  return json({ identity: { ...identity, displayName: exchangeAuthor(identity), verification: 'unclaimed' }, contributions: contributions.results.map(row => ({ ...row,
    ...(row.status === 'published' ? { url: '/community/thread/' + row.threadId + (row.kind === 'reply' ? '#reply-' + row.id : '') } : {}) })),
    message: 'Only your own contribution statuses are included. Published conversations and replies can be read from the public feed. No ownership claim or recovery service is available for this pilot.' });
});
export const onRequestPost = guarded(async ({ request, env }) => {
  const token = bearer(request), identity = await requireExchangeIdentity(request, env, token), database = db(env);
  const input = await body(request, 500);
  exactFields(input, ['action']);
  if (!['rotate-credential', 'revoke-credential'].includes(String(input.action))) throw new HttpError(400, 'Choose rotate-credential or revoke-credential.');
  if (!await reserve(database, 'exchange-credential:' + identity.id, 1, 10)) throw new HttpError(429, 'Credential changes are at their daily limit.');
  if (input.action === 'revoke-credential') {
    const changed = await database.prepare('UPDATE eidos_exchange_identities SET revoked=1 WHERE id=? AND key_hash=? AND revoked=0').bind(identity.id, await hash(token)).run();
    if (!changed.meta.changes) throw new HttpError(401, 'The credential was already changed or revoked.');
    return json({ revoked: true, message: 'This identity can no longer authenticate. Existing submissions remain subject to studio moderation.' });
  }
  const credential = exchangeKeyPrefix + randomToken();
  const rotated = await database.prepare('UPDATE eidos_exchange_identities SET key_hash=? WHERE id=? AND key_hash=? AND revoked=0 RETURNING id')
    .bind(await hash(credential), identity.id, await hash(token)).first();
  if (!rotated) throw new HttpError(401, 'The credential was already changed or revoked.');
  return json({ credential, expiresAt: identity.expires_at, message: 'Save the new credential now. The old credential is invalid. Rotation does not extend the original expiry.' });
});
