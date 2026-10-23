import { admin, body, db, guarded, HttpError, json, origin } from '../../_shared/platform/core';
import { hostDescriptor } from '../../_shared/platform/exchangeHost';
import { exactFields, exchangeTransport } from '../../_shared/platform/exchangeIdentity';
export const onRequestGet = guarded(async ({ request, env }) => {
  exchangeTransport(request, env);
  await admin(request, env);
  const database = db(env);
  const reviews = await database.prepare(`SELECT e.id,e.identity_id,i.name,e.topic,e.note,e.outcome,e.created_at FROM eidos_exchange_host_events e
    JOIN eidos_exchange_identities i ON i.id=e.identity_id WHERE e.outcome='pending' ORDER BY e.created_at LIMIT 20`).all();
  const events = await database.prepare(`SELECT id,identity_id,action,topic,outcome,created_at,resolved_at FROM eidos_exchange_host_events ORDER BY created_at DESC,id DESC LIMIT 100`).all();
  const quotas = await database.prepare("SELECT bucket,used FROM eidos_quotas WHERE bucket='exchange-host-attempts' AND period=?").bind(Math.floor(Date.now() / 86400000)).all();
  return json({ host: hostDescriptor(env), reviews: reviews.results, events: events.results, quotas: quotas.results });
});
export const onRequestPost = guarded(async ({ request, env }) => {
  exchangeTransport(request, env); origin(request); await admin(request, env);
  const input = await body(request, 500);
  exactFields(input, ['id', 'outcome']);
  if (typeof input.id !== 'string' || !/^[a-f0-9-]{36}$/.test(input.id) || !['closed', 'declined'].includes(String(input.outcome))) throw new HttpError(400, 'Choose a review id and closed or declined.');
  const result = await db(env).prepare("UPDATE eidos_exchange_host_events SET outcome=?,resolved_at=? WHERE id=? AND outcome='pending'")
    .bind(input.outcome, Math.floor(Date.now() / 1000), input.id).run();
  if (!result.meta.changes) throw new HttpError(409, 'This review is no longer pending.');
  return json({ ok: true, message: 'Review recorded. Identity verification and permissions remain separate.' });
});
