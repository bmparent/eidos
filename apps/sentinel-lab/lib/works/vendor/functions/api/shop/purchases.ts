import { body, clean, db, guarded, HttpError, json, origin } from '../../_shared/platform/core';
import { kitDownload, kitOwner } from '../../_shared/platform/kitDelivery';
import type { Order } from '../../_shared/platform/shop';
const owned = ' FROM eidos_orders o LEFT JOIN eidos_kit_attempts a ON a.order_id=o.id LEFT JOIN eidos_kit_recovery r ON r.order_id=o.id WHERE (a.owner_id=? OR (a.owner_id IS NULL AND r.email_hash=?))';
export const onRequestGet = guarded(async ({ request, env }) => {
  const owner = await kitOwner(request, env);
  const result = await db(env).prepare('SELECT o.id,o.status,o.created_at,a.archive_digest' + owned + ' ORDER BY o.created_at DESC LIMIT 100').bind(owner.id, owner.emailHash).all();
  return json({ purchases: result.results });
});
export const onRequestPost = guarded(async ({ request, env }) => {
  origin(request);
  const owner = await kitOwner(request, env), input = await body(request, 1000);
  const order = await db(env).prepare('SELECT o.*' + owned + ' AND o.id=?').bind(owner.id, owner.emailHash, clean(input.id, 80)).first<Order>();
  if (!order) throw new HttpError(404, 'Purchase not found.');
  return kitDownload(env, order);
});
