import {db,guarded,HttpError,json,readText,record} from '../../_shared/platform/core';
import {parseStripeEvent,verifyStripeSignature} from '../../_shared/snapshot/stripe';
import type {PlaygroundOrder} from '../../_shared/platform/playgroundOrders';
import {ensurePlayground} from '../../_shared/platform/playground';
export const onRequestPost = guarded(async ({request,env}) => {
  if (!env.EIDOS_PLAYGROUND_WEBHOOK_SECRET) throw new HttpError(503,'Webhook is not configured.');
  const raw = await readText(request,64000);
  if (!await verifyStripeSignature(raw,request.headers.get('stripe-signature') || '',env.EIDOS_PLAYGROUND_WEBHOOK_SECRET)) throw new HttpError(400,'Invalid signature.');
  let parsed: unknown; try {parsed=JSON.parse(raw);} catch {throw new HttpError(400,'Invalid event.');}
  const event=parseStripeEvent(parsed);
  if (!event || !record(parsed) || parsed.livemode !== false) throw new HttpError(400,'Expected a test-mode event.');
  await ensurePlayground(env);
  const database=db(env), object=event.data.object, now=new Date().toISOString();
  if (await database.prepare('SELECT id FROM eidos_pg_events WHERE id=?').bind(event.id).first()) return json({received:true});
  const eventStatement=database.prepare('INSERT OR IGNORE INTO eidos_pg_events(id,received_at) VALUES(?,?)').bind(event.id,now);
  if (['checkout.session.completed','checkout.session.async_payment_succeeded','checkout.session.async_payment_failed','checkout.session.expired'].includes(event.type)) {
    if (!record(object.metadata) || object.metadata.eidos_product !== 'playground-v1') return json({received:true});
    const order=await database.prepare('SELECT * FROM eidos_pg_orders WHERE id=?').bind(object.metadata.eidos_order_id).first<PlaygroundOrder>();
    if (!order?.session_id) throw new HttpError(503,'Checkout is still being recorded. Retry this event.');
    if (order.mode !== 'test' || object.livemode !== false || object.id !== order.session_id || object.client_reference_id !== order.id || object.mode !== 'payment' || object.currency !== 'usd' || object.amount_total !== order.amount) throw new HttpError(400,'Payment does not match the frozen order.');
    if (['checkout.session.completed','checkout.session.async_payment_succeeded'].includes(event.type) && object.payment_status === 'paid') {
      if (typeof object.payment_intent !== 'string' || !object.payment_intent.startsWith('pi_')) throw new HttpError(400,'Payment intent missing.');
      await database.batch([database.prepare("UPDATE eidos_pg_orders SET status=CASE WHEN status='revoked' OR EXISTS(SELECT 1 FROM eidos_pg_revoked WHERE payment_intent=?) THEN 'revoked' ELSE 'paid' END,payment_intent=?,paid_at=COALESCE(paid_at,?) WHERE id=?") .bind(object.payment_intent,object.payment_intent,now,order.id),eventStatement]);
    } else if (['checkout.session.expired','checkout.session.async_payment_failed'].includes(event.type)) {
      await database.batch([database.prepare("UPDATE eidos_pg_orders SET status='cancelled' WHERE id=? AND status='pending'").bind(order.id),eventStatement]);
    } else await eventStatement.run(); // Delayed payments remain pending until their signed success event.
  } else if (['charge.refunded','charge.dispute.created'].includes(event.type) && typeof object.payment_intent === 'string') {
    await database.batch([database.prepare('INSERT OR IGNORE INTO eidos_pg_revoked(payment_intent) VALUES(?)').bind(object.payment_intent),database.prepare("UPDATE eidos_pg_orders SET status='revoked' WHERE payment_intent=?").bind(object.payment_intent),eventStatement]);
  }
  return json({received:true});
});
