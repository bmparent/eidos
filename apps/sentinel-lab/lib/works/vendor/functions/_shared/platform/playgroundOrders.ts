import { db, HttpError, type PlatformEnv } from './core';
import { identifier, projectRevision } from './playground';
import { exportFiles, zipFiles } from '../../../src/playground/export';

export interface PlaygroundOrder {
  id: string; owner_id: string; request_id: string; project_id: string; revision_id: string;
  name: string; archive: string; amount: number; mode: string; status: string;
  session_id: string | null; checkout_url: string | null; payment_intent: string | null;
  created_at: string; paid_at: string | null;
}
export function testPrice(env: PlatformEnv) {
  const amount = Number(env.EIDOS_PLAYGROUND_TEST_PRICE_CENTS);
  if (!env.EIDOS_PLAYGROUND_STRIPE_KEY?.startsWith('sk_test_') || !env.EIDOS_PLAYGROUND_WEBHOOK_SECRET || !Number.isInteger(amount) || amount < 50 || amount > 100_000)
    throw new HttpError(503, 'Test checkout is not configured. Free preview exports remain available.');
  return amount;
}
export async function createPlaygroundCheckout(env: PlatformEnv, owner: string, input: Record<string, unknown>) {
  const amount = testPrice(env), requestId = identifier(input.requestId), database = db(env);
  const projectId = identifier(input.projectId), revisionId = identifier(input.revisionId);
  let order = await database.prepare('SELECT * FROM eidos_pg_orders WHERE owner_id=? AND request_id=?').bind(owner,requestId).first<PlaygroundOrder>();
  if (!order) {
    const count = await database.prepare('SELECT COUNT(*) AS n FROM eidos_pg_orders WHERE owner_id=?').bind(owner).first<{n:number}>();
    if ((count?.n || 0) >= 100) throw new HttpError(409,'This account has reached the preview checkout limit. Existing purchases remain available.');
    const result = await projectRevision(env,owner,projectId,revisionId);
    const archiveBytes = zipFiles(exportFiles(result.document));
    let binary = '';
    for (let i=0;i<archiveBytes.length;i+=8192) binary += String.fromCharCode(...archiveBytes.subarray(i,i+8192));
    await database.prepare('INSERT OR IGNORE INTO eidos_pg_orders(id,owner_id,request_id,project_id,revision_id,name,archive,amount,mode,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)')
      .bind(crypto.randomUUID(),owner,requestId,projectId,revisionId,result.document.name,btoa(binary),amount,'test',new Date().toISOString()).run();
    order = await database.prepare('SELECT * FROM eidos_pg_orders WHERE owner_id=? AND request_id=?').bind(owner,requestId).first<PlaygroundOrder>();
  }
  if (!order || order.project_id !== projectId || order.revision_id !== revisionId) throw new HttpError(409,'This checkout request belongs to a different revision.');
  if (order.status !== 'pending') return {orderId:order.id,status:order.status};
  if (order.checkout_url) return {orderId:order.id,url:order.checkout_url};
  // Stripe keeps idempotency keys for at least 24 hours. Never recreate an old ambiguous checkout.
  if (Date.now()-Date.parse(order.created_at)>23*60*60*1000) throw new HttpError(409,'This checkout attempt expired. Start a new attempt from purchase history.');
  const site = new URL(env.PUBLIC_SITE_URL || 'https://eidos-works.com').origin;
  const values = new URLSearchParams({ mode:'payment', client_reference_id:order.id,
    success_url:site+'/playground?checkout=returned',cancel_url:site+'/playground?checkout=cancelled',
    'metadata[eidos_product]':'playground-v1','metadata[eidos_order_id]':order.id,
    'line_items[0][quantity]':'1','line_items[0][price_data][currency]':'usd',
    'line_items[0][price_data][unit_amount]':String(order.amount),
    'line_items[0][price_data][product_data][name]':'Eidos Playground export — TEST',
  });
  const response = await fetch('https://api.stripe.com/v1/checkout/sessions',{method:'POST',headers:{authorization:'Bearer '+env.EIDOS_PLAYGROUND_STRIPE_KEY,'content-type':'application/x-www-form-urlencoded','Idempotency-Key':'eidos-pg-'+order.id,'Stripe-Version':'2026-08-26.dahlia'},body:values,signal:AbortSignal.timeout(15000)});
  if (!response.ok) throw new HttpError(503,'Checkout could not be started. Retry the same attempt; your frozen revision is safe.');
  const session = await response.json() as {id?:string;url?:string;livemode?:boolean};
  if (!session.id || !session.url || session.livemode !== false || new URL(session.url).origin !== 'https://checkout.stripe.com') throw new HttpError(503,'Unexpected checkout response.');
  await database.prepare('UPDATE eidos_pg_orders SET session_id=?,checkout_url=? WHERE id=? AND owner_id=?').bind(session.id,session.url,order.id,owner).run();
  return {orderId:order.id,url:session.url};
}
