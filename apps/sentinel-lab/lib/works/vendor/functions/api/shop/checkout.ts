import {
  body,
  challenge,
  clean,
  db,
  fingerprint,
  guarded,
  hash,
  HttpError,
  json,
  origin,
  reserve,
  siteOrigin,
} from '../../_shared/platform/core';
import { KIT_PRICE, shopReady } from '../../_shared/platform/shop';
export const onRequestPost = guarded(async ({ request, env }) => {
  origin(request);
  shopReady(env);
  const input = await body(request, 4000);
  if (input.acceptTerms !== true)
    throw new HttpError(
      400,
      'Please accept the product terms before checkout.',
    );
  if (clean(input.website)) throw new HttpError(400, 'Please try again.');
  const database = db(env),
    visitor = await fingerprint(request, env);
  if (!(await reserve(database, 'checkout:' + visitor, 1, 5)))
    throw new HttpError(
      429,
      'You have opened several checkouts today. Please use an existing checkout or try again tomorrow.',
    );
  await challenge(request, env, input.challenge, 'checkout');
  const id = crypto.randomUUID(),
    receipt =
      crypto.randomUUID().replaceAll('-', '') +
      crypto.randomUUID().replaceAll('-', '');
  await database
    .prepare(
      'INSERT INTO eidos_orders(id,receipt_hash,created_at) VALUES(?,?,?)',
    )
    .bind(id, await hash(receipt), new Date().toISOString())
    .run();
  const useCase = ['own-website', 'client-project', 'learning'].includes(
    String(input.useCase),
  )
    ? String(input.useCase)
    : 'unspecified';
  const values = new URLSearchParams({
    'metadata[eidos_use_case]': useCase,
    mode: 'payment',
    'payment_method_types[0]': 'card',
    client_reference_id: id,
    'metadata[eidos_product]': 'cinematic-starter-v1',
    'metadata[eidos_order_id]': id,
    'payment_intent_data[metadata][eidos_order_id]': id,
    success_url: siteOrigin(env) + '/shop/success#receipt=' + receipt,
    cancel_url: siteOrigin(env) + '/shop/cinematic-starter?checkout=cancelled',
    'line_items[0][quantity]': '1',
    'line_items[0][price_data][currency]': 'usd',
    'line_items[0][price_data][unit_amount]': String(KIT_PRICE),
    'line_items[0][price_data][product_data][name]':
      'Eidos Cinematic Starter — one website license',
    'line_items[0][price_data][product_data][description]':
      'HTML, CSS, JavaScript, setup notes, and a commercial license for one finished website. Digital download; no hosting or custom implementation.',
    expires_at: String(Math.floor(Date.now() / 1000) + 1800),
  });
  const response = await fetch('https://api.stripe.com/v1/checkout/sessions', {
    method: 'POST',
    headers: {
      authorization: 'Bearer ' + env.STRIPE_SECRET_KEY,
      'content-type': 'application/x-www-form-urlencoded',
      'idempotency-key': 'eidos-kit-' + id,
    },
    body: values,
    signal: AbortSignal.timeout(15000),
  });
  if (!response.ok)
    throw new HttpError(
      502,
      'Checkout could not be opened. No payment was taken here. Please try again.',
    );
  const session = (await response.json()) as { id?: string; url?: string };
  if (!/^cs_[A-Za-z0-9_]+$/.test(session.id || ''))
    throw new HttpError(502, 'The payment session was invalid.');
  let checkout: URL;
  try {
    checkout = new URL(session.url || '');
  } catch {
    throw new HttpError(502, 'The payment link was invalid.');
  }
  if (
    checkout.protocol !== 'https:' ||
    checkout.hostname !== 'checkout.stripe.com'
  )
    throw new HttpError(502, 'The payment link was invalid.');
  await database
    .prepare('UPDATE eidos_orders SET session_id=? WHERE id=?')
    .bind(session.id, id)
    .run();
  return json({ url: checkout.toString() });
});
