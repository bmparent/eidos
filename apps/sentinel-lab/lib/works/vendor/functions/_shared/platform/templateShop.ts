import { body, challenge, clean, db, fingerprint, guarded, hash, HttpError, json, origin, record, reserve, siteOrigin, type PlatformEnv } from './core';
import { memberFromRequest, randomToken } from './memberAuth';
import { sendMemberMail } from './memberMail';
import { templateProduct, templateProducts, TEMPLATE_STRIPE_API_VERSION, type TemplateProduct } from './templateCatalog';
import type { Order } from './shop';

export interface TemplateOrder extends Order {
  order_id: string; product_id: string; edition_id: string; version: string; price_cents: number;
  archive_key: string; archive_sha256: string; download_name: string; guide_path: string;
  member_id: string | null; email: string | null; email_hash: string | null; receipt_expires: number;
  session_url: string | null; checkout_expires: number; mail_status: string; fulfillment_type: string; fulfillment_status: string;
  payment_failed: number;
}
const now = () => Math.floor(Date.now() / 1000);
const checkoutSelect = 'SELECT o.*, t.*, EXISTS(SELECT 1 FROM eidos_template_payment_failures f WHERE f.order_id=o.id) AS payment_failed FROM eidos_orders o JOIN eidos_template_orders t ON t.order_id=o.id';
export function templateTestReady(env: PlatformEnv) {
  return env.EIDOS_TEMPLATE_TEST_ENABLED === 'true' && /^[sr]k_test_/.test(env.STRIPE_SECRET_KEY || '') &&
    env.EIDOS_SHOP_ENABLED === 'true' && Boolean(env.EIDOS_KIT_WEBHOOK_SECRET && env.EIDOS_DB && env.EIDOS_TEMPLATE_ARCHIVES) &&
    !/(^|\.)eidos-works\.com$/.test(new URL(siteOrigin(env)).hostname);
}
export function requireTemplateTest(env: PlatformEnv) {
  if (!templateTestReady(env)) throw new HttpError(503, 'Template purchases are available only in the isolated Stripe TEST candidate.');
}
async function archive(env: PlatformEnv, key: string) {
  const value = await env.EIDOS_TEMPLATE_ARCHIVES?.get(key);
  if (!value || !/^[a-f0-9]{64}$/.test(value.sha256) || !value.bytes.length) throw new HttpError(503, 'This version is not available for download. Contact support; do not repurchase.');
  return value;
}
export function verifiedEdition(env: PlatformEnv, product: TemplateProduct) {
  return (env.EIDOS_TEMPLATE_VERIFIED_EDITIONS || '').split(',').map(s => s.trim()).includes(product.id);
}
export const catalog = guarded(async ({ env }) => {
  const ready = templateTestReady(env);
  const editions = await Promise.all(templateProducts.map(async product => {
    const packaged = Boolean(await env.EIDOS_TEMPLATE_ARCHIVES?.get(product.archiveKey));
    const verified = verifiedEdition(env, product);
    return { id: product.id, productId: product.id, editionId: product.editionId, version: product.version,
      priceCents: product.priceCents, guidePath: product.guidePath, available: ready && packaged && verified,
      status: !packaged ? 'package_missing' : !verified ? 'installation_unverified' : ready ? 'test_available' : 'test_disabled' };
  }));
  return json({ mode: 'test', ready, sourceRevision: env.EIDOS_TEMPLATE_SOURCE_REVISION || null, editions: [...editions,
    { id: 'switchboard-squarespace-v1', productId: 'switchboard-squarespace-v1', editionId: 'squarespace', priceCents: 2900, version: null, available: false, status: 'platform_installation_blocked' },
    { id: 'switchboard-godaddy-v1', productId: 'switchboard-godaddy-v1', editionId: 'godaddy', priceCents: 2900, version: null, available: false, status: 'platform_installation_blocked' },
  ] });
});
export async function templateCheckout(request: Request, env: PlatformEnv, input: Record<string, unknown>) {
  requireTemplateTest(env);
  const product = templateProduct(input.productId, input.editionId);
  if (!product) throw new HttpError(400, 'Choose a supported product and edition.');
  for (const field of ['price', 'priceCents', 'amount', 'currency', 'version', 'archiveKey', 'priceId'])
    if (input[field] !== undefined) throw new HttpError(400, 'Prices and package versions are set by the server.');
  if (input.acceptTerms !== true) throw new HttpError(400, 'Please accept the product terms before checkout.');
  if (!verifiedEdition(env, product)) throw new HttpError(503, 'This edition has not passed installation acceptance and cannot be purchased.');
  if (clean(input.website)) throw new HttpError(400, 'Please try again.');
  const item = await archive(env, product.archiveKey);
  const database = db(env), visitor = await fingerprint(request, env);
  if (!(await reserve(database, 'template-checkout:' + visitor, 1, 20))) throw new HttpError(429, 'Please use an existing checkout or try again tomorrow.');
  await challenge(request, env, input.challenge, 'checkout');
  const member = await memberFromRequest(request, env, false);
  const receipt = input.attempt === undefined ? randomToken() : clean(input.attempt, 80);
  if (!/^[a-f0-9]{64}$/.test(receipt)) throw new HttpError(400, 'Start a valid checkout attempt.');
  const receiptHash = await hash(receipt);
  let order = await database.prepare(checkoutSelect + ' WHERE o.receipt_hash=?').bind(receiptHash).first<TemplateOrder>();
  if (order) {
    if (order.product_id !== product.id || order.edition_id !== product.editionId || order.member_id !== (member?.id || null))
      throw new HttpError(409, 'This attempt belongs to a different purchase.');
    if (order.status !== 'pending' || order.payment_failed || order.checkout_expires <= now()) throw new HttpError(409, 'This checkout is complete, failed or expired. Check its status before starting again.');
    if (order.session_url) return json({ url: order.session_url });
  } else {
    const id = crypto.randomUUID(), created = new Date().toISOString(), expires = now() + 1800;
    await database.batch([
      database.prepare('INSERT INTO eidos_orders(id,receipt_hash,created_at) VALUES(?,?,?)').bind(id, receiptHash, created),
      database.prepare('INSERT INTO eidos_template_orders(order_id,product_id,edition_id,version,price_cents,archive_key,archive_sha256,download_name,guide_path,member_id,email,email_hash,receipt_expires,checkout_expires,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)')
        .bind(id, product.id, product.editionId, product.version, product.priceCents, product.archiveKey, item.sha256, product.downloadName, product.guidePath,
          member?.id || null, member?.email || null, member?.email ? await hash(member.email.toLowerCase()) : null, now() + 86400, expires, created),
    ]);
    order = await database.prepare(checkoutSelect + ' WHERE o.id=?').bind(id).first<TemplateOrder>();
  }
  if (!order) throw new HttpError(503, 'Checkout could not be prepared.');
  const values = new URLSearchParams({
    mode: 'payment', client_reference_id: order.id,
    'metadata[eidos_product]': product.id, 'metadata[eidos_edition]': product.editionId,
    'metadata[eidos_version]': product.version, 'metadata[eidos_order_id]': order.id,
    'payment_intent_data[metadata][eidos_order_id]': order.id,
    success_url: siteOrigin(env) + '/shop/start#receipt=' + receipt,
    cancel_url: siteOrigin(env) + '/shop/' + product.id.split('-')[0] + '/?checkout=cancelled',
    'line_items[0][quantity]': '1', 'line_items[0][price_data][currency]': 'usd',
    'line_items[0][price_data][unit_amount]': String(order.price_cents),
    'line_items[0][price_data][product_data][name]': 'Eidos ' + product.name + ' (TEST)',
    'line_items[0][price_data][product_data][description]': 'Version ' + product.version + '. Digital files; third-party hosting, platform plans, booking, forms and commerce services are separate.',
    expires_at: String(order.checkout_expires),
    ...(member?.email ? { customer_email: member.email } : {}),
  });
  const response = await fetch('https://api.stripe.com/v1/checkout/sessions', {
    method: 'POST', headers: { authorization: 'Bearer ' + env.STRIPE_SECRET_KEY, 'content-type': 'application/x-www-form-urlencoded', 'idempotency-key': 'eidos-template-' + order.id, 'stripe-version': TEMPLATE_STRIPE_API_VERSION },
    body: values, signal: AbortSignal.timeout(15000), redirect: 'error',
  });
  if (!response.ok) throw new HttpError(502, 'Checkout could not be confirmed. Retry this same attempt before starting another purchase.');
  const session = await response.json() as { id?: string; url?: string; livemode?: boolean };
  if (!/^cs_test_[A-Za-z0-9_]+$/.test(session.id || '') || session.livemode !== false) throw new HttpError(502, 'A Stripe TEST session is required.');
  let checkout: URL;
  try { checkout = new URL(session.url || ''); } catch { throw new HttpError(502, 'Invalid payment link.'); }
  if (checkout.protocol !== 'https:' || checkout.hostname !== 'checkout.stripe.com') throw new HttpError(502, 'Invalid payment link.');
  await database.batch([
    database.prepare('UPDATE eidos_orders SET session_id=? WHERE id=? AND (session_id IS NULL OR session_id=?)').bind(session.id, order.id, session.id),
    database.prepare('UPDATE eidos_template_orders SET session_url=? WHERE order_id=?').bind(checkout.toString(), order.id),
  ]);
  return json({ url: checkout.toString() });
}
export async function templateOrder(env: PlatformEnv, orderId: string) {
  return db(env).prepare(checkoutSelect + ' WHERE o.id=?').bind(orderId).first<TemplateOrder>();
}
async function authorizedTemplateOrder(request: Request, env: PlatformEnv, order: TemplateOrder) {
  if (order.receipt_expires <= now()) throw new HttpError(401, 'This purchase link expired. Recover your purchase by email.');
  const member = await memberFromRequest(request, env, false);
  if (member && order.member_id && member.id !== order.member_id) throw new HttpError(403, 'This purchase belongs to another account.');
}
export async function templateStatus(request: Request, env: PlatformEnv, order: TemplateOrder) {
  await authorizedTemplateOrder(request, env, order);
  let downloadToken: string | null = null;
  const downloadExpires = now() + 600;
  if (order.status === 'paid') {
    downloadToken = randomToken();
    await db(env).prepare('INSERT INTO eidos_template_downloads(token_hash,order_id,expires) VALUES(?,?,?)').bind(await hash(downloadToken), order.id, downloadExpires).run();
  }
  return json({ status: order.status, orderId: order.id, productId: order.product_id, editionId: order.edition_id,
    version: order.version, downloadName: order.download_name, guidePath: order.guide_path, fulfillmentType: order.fulfillment_type,
    fulfillmentStatus: order.fulfillment_status, mailStatus: order.mail_status, paymentState: order.status === 'pending' ? order.payment_failed ? 'failed' : order.checkout_expires <= now() ? 'expired' : 'pending' : order.status,
    downloadToken, downloadExpires: downloadToken ? downloadExpires : null });
}
export async function templateDownload(request: Request, env: PlatformEnv, order: TemplateOrder, token: unknown) {
  await authorizedTemplateOrder(request, env, order);
  if (order.status !== 'paid' || order.fulfillment_status !== 'ready') throw new HttpError(403, 'A verified, completed purchase is required to download this package.');
  if (typeof token !== 'string' || !/^[a-f0-9]{64}$/.test(token)) throw new HttpError(401, 'Open your purchase start page to create a fresh download link.');
  const allowed = await db(env).prepare('SELECT order_id FROM eidos_template_downloads WHERE token_hash=? AND order_id=? AND expires>?').bind(await hash(token), order.id, now()).first();
  if (!allowed) throw new HttpError(401, 'Download link expired. Reopen your purchase start page.');
  const value = await archive(env, order.archive_key);
  if (value.sha256 !== order.archive_sha256) throw new HttpError(503, 'The original purchased archive does not match its receipt. Contact support; do not repurchase.');
  return new Response(new Uint8Array(value.bytes).buffer, { headers: { 'content-type': 'application/zip', 'content-disposition': `attachment; filename="${order.download_name}"`,
    'cache-control': 'private, no-store', 'referrer-policy': 'no-referrer', 'x-content-type-options': 'nosniff', 'x-eidos-archive-sha256': value.sha256 } });
}
async function recoveryToken(env: PlatformEnv, orderId: string, stableReceipt = false) {
  // Stable receipt-mail body makes Resend idempotency safe across webhook retries.
  let token = randomToken();
  if (stableReceipt) {
    const encoder = new TextEncoder();
    const key = await crypto.subtle.importKey('raw', encoder.encode(env.EIDOS_KIT_WEBHOOK_SECRET || ''), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
    token = [...new Uint8Array(await crypto.subtle.sign('HMAC', key, encoder.encode('template-receipt:' + orderId)))].map(n => n.toString(16).padStart(2, '0')).join('');
  }
  await db(env).prepare('INSERT OR IGNORE INTO eidos_template_recovery(token_hash,order_id,expires) VALUES(?,?,?)').bind(await hash(token), orderId, now() + 1800).run();
  return token;
}
export async function templateReceiptMail(env: PlatformEnv, order: TemplateOrder) {
  if (!order.email || order.status !== 'paid' || order.mail_status === 'sent') return;
  const token = await recoveryToken(env, order.id, true);
  try {
    const providerId = await sendMemberMail(env, { to: order.email, subject: 'Your Eidos template — download and setup (TEST)',
      text: `Your TEST purchase of ${order.product_id}, ${order.edition_id}, version ${order.version} is verified.\nOpen this one-time recovery link within 30 minutes: ${siteOrigin(env)}/shop/start#token=${token}\nInstallation guide: ${siteOrigin(env)}${order.guide_path}\nIf this link expires, use ${siteOrigin(env)}/shop/start to request a new link.\nThis is transactional delivery and does not subscribe you to marketing.` }, 'eidos-template-receipt-' + order.id);
    await db(env).prepare("UPDATE eidos_template_orders SET mail_status='sent',mail_provider_id=? WHERE order_id=?").bind(providerId, order.id).run();
  } catch (error) {
    await db(env).prepare("UPDATE eidos_template_orders SET mail_status='failed' WHERE order_id=?").bind(order.id).run();
    throw error; // Leave Stripe event unacknowledged so provider retries delivery.
  }
}
export async function templatePayment(env: PlatformEnv, event: { id: string; type: string; data: { object: Record<string, unknown> } }, livemode: unknown) {
  if (!['checkout.session.completed', 'checkout.session.async_payment_succeeded', 'checkout.session.async_payment_failed', 'checkout.session.expired'].includes(event.type)) return false;
  const object = event.data.object;
  const metadata = object.metadata;
  if (!record(metadata) || typeof metadata.eidos_product !== 'string' || !templateProducts.some(p => p.id === metadata.eidos_product)) return false;
  requireTemplateTest(env);
  if (livemode !== false) throw new HttpError(400, 'Template candidate requires a Stripe TEST event.');
  const order = await templateOrder(env, clean(metadata.eidos_order_id, 80));
  if (!order?.session_id) throw new HttpError(503, 'Order is not ready. Retry this event.');
  if (order.product_id !== metadata.eidos_product || order.edition_id !== metadata.eidos_edition || order.version !== metadata.eidos_version || order.session_id !== object.id || object.client_reference_id !== order.id)
    throw new HttpError(400, 'Payment does not match this order and edition.');
  const database = db(env);
  if (event.type === 'checkout.session.async_payment_failed') {
    await database.batch([
      database.prepare("INSERT OR IGNORE INTO eidos_template_payment_failures(order_id,failed_at,event_id) SELECT id,?,? FROM eidos_orders WHERE id=? AND status='pending'").bind(new Date().toISOString(), event.id, order.id),
      database.prepare('INSERT OR IGNORE INTO eidos_stripe_events(id,received_at) VALUES(?,?)').bind(event.id, new Date().toISOString()),
    ]);
    return true;
  }
  if (event.type === 'checkout.session.expired') {
    await database.batch([
      database.prepare('UPDATE eidos_template_orders SET checkout_expires=MIN(checkout_expires,?) WHERE order_id=?').bind(now(), order.id),
      database.prepare('INSERT OR IGNORE INTO eidos_stripe_events(id,received_at) VALUES(?,?)').bind(event.id, new Date().toISOString()),
    ]);
    return true;
  }
  if (!['checkout.session.completed', 'checkout.session.async_payment_succeeded'].includes(event.type)) return true;
  if (object.payment_status !== 'paid') return true; // Delayed payment remains pending until the paid event.
  if (order.payment_failed) return true; // A terminal failed attempt cannot be revived by a later or replayed event.
  if (object.mode !== 'payment' || object.currency !== 'usd' || object.amount_total !== order.price_cents || typeof object.payment_intent !== 'string' || !object.payment_intent)
    throw new HttpError(400, 'Payment does not match the server price.');
  const email = record(object.customer_details) ? clean(object.customer_details.email, 260).toLowerCase() : '';
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) throw new HttpError(400, 'Verified payment email is required for recovery.');
  if (order.member_id && order.email && email !== order.email) throw new HttpError(400, 'Payment email does not match the account purchase.');
  const paymentIntent = object.payment_intent;
  await database.batch([
    database.prepare("UPDATE eidos_orders SET status=CASE WHEN status='refunded' OR EXISTS(SELECT 1 FROM eidos_revoked_payments WHERE payment_intent=?) THEN 'refunded' ELSE 'paid' END,paid_at=COALESCE(paid_at,?),payment_intent=? WHERE id=? AND NOT EXISTS(SELECT 1 FROM eidos_template_payment_failures WHERE order_id=?)")
      .bind(paymentIntent, new Date().toISOString(), paymentIntent, order.id, order.id),
    database.prepare("UPDATE eidos_template_orders SET email=?,email_hash=?,fulfillment_status=CASE WHEN EXISTS(SELECT 1 FROM eidos_orders WHERE id=? AND status='paid') THEN 'ready' ELSE 'revoked' END WHERE order_id=?")
      .bind(email, await hash(email), order.id, order.id),
  ]);
  const paid = await templateOrder(env, order.id);
  if (paid) await templateReceiptMail(env, paid);
  await database.prepare('INSERT OR IGNORE INTO eidos_stripe_events(id,received_at) VALUES(?,?)').bind(event.id, new Date().toISOString()).run();
  return true;
}
export const recover = guarded(async ({ request, env }) => {
  origin(request);
  const input = await body(request, 4000), database = db(env);
  await challenge(request, env, input.challenge, 'recovery');
  const visitor = await fingerprint(request, env);
  if (!(await reserve(database, 'template-recovery-ip:' + visitor, 1, 12))) throw new HttpError(429, 'Please wait before requesting another recovery message.');
  const email = clean(input.email, 260).toLowerCase();
  if (/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    const emailHash = await hash(email);
    if (await reserve(database, 'template-recovery-email:' + emailHash, 1, 4)) {
      const orders = await database.prepare(checkoutSelect + " WHERE t.email_hash=? AND o.status='paid' ORDER BY o.created_at DESC LIMIT 10").bind(emailHash).all<TemplateOrder>();
      for (const order of orders.results) {
        const token = await recoveryToken(env, order.id);
        try { await sendMemberMail(env, { to: email, subject: 'Recover your Eidos template purchase (TEST)', text: `Open within 30 minutes: ${siteOrigin(env)}/shop/start#token=${token}\nPurchase: ${order.product_id}, version ${order.version}.\nIf you did not request this message, ignore it.` }, 'eidos-template-recovery-' + await hash(token)); }
        catch { /* Enumeration-safe response; provider failure is recorded internally. */
          await database.prepare("UPDATE eidos_template_orders SET mail_status='failed' WHERE order_id=?").bind(order.id).run();
        }
      }
    }
  }
  return json({ message: 'If an eligible purchase matches that email, a recovery link will arrive shortly. Check spam or contact support if it does not arrive.' });
});
export const redeem = guarded(async ({ request, env }) => {
  origin(request);
  const input = await body(request, 2000), token = clean(input.token, 80), database = db(env);
  if (!/^[a-f0-9]{64}$/.test(token)) throw new HttpError(401, 'This recovery link is invalid or expired.');
  const tokenHash = await hash(token);
  const link = await database.prepare("SELECT r.order_id FROM eidos_template_recovery r JOIN eidos_orders o ON o.id=r.order_id WHERE r.token_hash=? AND r.used=0 AND r.expires>? AND o.status='paid'").bind(tokenHash, now()).first<{ order_id: string }>();
  if (!link) throw new HttpError(401, 'This recovery link is invalid or expired.');
  const order = await templateOrder(env, link.order_id), member = await memberFromRequest(request, env, false);
  if (!order || (member && order.member_id && member.id !== order.member_id)) throw new HttpError(403, 'This purchase belongs to another account.');
  const receipt = randomToken();
  const results = await database.batch([
    database.prepare("UPDATE eidos_template_recovery SET used=1 WHERE token_hash=? AND used=0 AND expires>? AND EXISTS(SELECT 1 FROM eidos_orders WHERE id=? AND status='paid')").bind(tokenHash, now(), link.order_id),
    database.prepare('UPDATE eidos_orders SET receipt_hash=? WHERE id=? AND changes()=1').bind(await hash(receipt), link.order_id),
    database.prepare('UPDATE eidos_template_orders SET receipt_expires=? WHERE order_id=? AND changes()=1').bind(now() + 86400, link.order_id),
  ]);
  const first = results[0] as { meta?: { changes?: number }; rowsAffected?: number };
  if ((first.meta?.changes ?? first.rowsAffected) !== 1) throw new HttpError(401, 'This recovery link is invalid or expired.');
  return json({ receipt });
});
