import { clean, db, hash, HttpError, type PlatformEnv } from './core';
export const KIT_PRICE = 2900;
export interface Order {
  id: string;
  receipt_hash: string;
  session_id: string | null;
  status: string;
  created_at: string;
  paid_at: string | null;
  payment_intent: string | null;
}
export function shopReady(env: PlatformEnv) {
  if (
    env.EIDOS_SHOP_ENABLED !== 'true' ||
    !env.STRIPE_SECRET_KEY ||
    !env.EIDOS_KIT_WEBHOOK_SECRET ||
    !env.EIDOS_DB
  )
    throw new HttpError(
      503,
      'Checkout is being prepared. The free preview is available now.',
    );
}
export async function receiptOrder(env: PlatformEnv, receipt: unknown) {
  const value = clean(receipt, 80);
  if (!/^[a-f0-9]{64}$/.test(value))
    throw new HttpError(401, 'A valid purchase receipt is required.');
  const order = await db(env)
    .prepare('SELECT * FROM eidos_orders WHERE receipt_hash=?')
    .bind(await hash(value))
    .first<Order>();
  if (!order)
    throw new HttpError(
      404,
      'This purchase receipt was not found. Contact billing with your payment reference.',
    );
  return order;
}
