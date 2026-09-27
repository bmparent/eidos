import {
  db,
  guarded,
  HttpError,
  json,
  readText,
  record,
} from '../../_shared/platform/core';
import {
  verifyStripeSignature,
  parseStripeEvent,
} from '../../_shared/snapshot/stripe';
import { KIT_PRICE, type Order } from '../../_shared/platform/shop';
export const onRequestPost = guarded(async ({ request, env }) => {
  if (!env.EIDOS_KIT_WEBHOOK_SECRET)
    throw new HttpError(503, 'Webhook unavailable.');
  const raw = await readText(request, 64000);
  if (
    !(await verifyStripeSignature(
      raw,
      request.headers.get('stripe-signature') || '',
      env.EIDOS_KIT_WEBHOOK_SECRET,
    ))
  )
    throw new HttpError(400, 'Invalid signature.');
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new HttpError(400, 'Invalid event.');
  }
  const event = parseStripeEvent(parsed);
  if (!event) throw new HttpError(400, 'Invalid event.');
  const database = db(env);
  if (
    await database
      .prepare('SELECT id FROM eidos_stripe_events WHERE id=?')
      .bind(event.id)
      .first()
  )
    return json({ received: true });
  const object = event.data.object;
  if (
    [
      'checkout.session.completed',
      'checkout.session.async_payment_succeeded',
    ].includes(event.type)
  ) {
    if (
      !record(object.metadata) ||
      object.metadata.eidos_product !== 'cinematic-starter-v1'
    )
      return json({ received: true });
    if (object.payment_status !== 'paid') return json({ received: true });
    if (
      object.mode !== 'payment' ||
      object.currency !== 'usd' ||
      object.amount_total !== KIT_PRICE
    )
      throw new HttpError(400, 'Payment does not match the product.');
    const order = await database
      .prepare('SELECT * FROM eidos_orders WHERE id=?')
      .bind(object.metadata.eidos_order_id)
      .first<Order>();
    if (!order?.session_id)
      throw new HttpError(503, 'Order is not ready. Retry this event.');
    if (
      order.session_id !== object.id ||
      object.client_reference_id !== order.id
    )
      throw new HttpError(400, 'Payment does not match this order.');
    await database.batch([
      database
        .prepare(
          "UPDATE eidos_orders SET status=CASE WHEN status='refunded' OR EXISTS(SELECT 1 FROM eidos_revoked_payments WHERE payment_intent=?) THEN 'refunded' ELSE 'paid' END,paid_at=COALESCE(paid_at,?),payment_intent=? WHERE id=?",
        )
        .bind(
          typeof object.payment_intent === 'string'
            ? object.payment_intent
            : '',
          new Date().toISOString(),
          typeof object.payment_intent === 'string'
            ? object.payment_intent
            : null,
          order.id,
        ),
      database
        .prepare(
          'INSERT OR IGNORE INTO eidos_stripe_events(id,received_at) VALUES(?,?)',
        )
        .bind(event.id, new Date().toISOString()),
    ]);
  } else if (
    ['charge.refunded', 'charge.dispute.created'].includes(event.type) &&
    typeof object.payment_intent === 'string'
  ) {
    await database.batch([
      database
        .prepare(
          'INSERT OR IGNORE INTO eidos_revoked_payments(payment_intent,received_at) VALUES(?,?)',
        )
        .bind(object.payment_intent, new Date().toISOString()),
      database
        .prepare(
          "UPDATE eidos_orders SET status='refunded' WHERE payment_intent=?",
        )
        .bind(object.payment_intent),
      database
        .prepare(
          'INSERT OR IGNORE INTO eidos_stripe_events(id,received_at) VALUES(?,?)',
        )
        .bind(event.id, new Date().toISOString()),
    ]);
  }
  return json({ received: true });
});
