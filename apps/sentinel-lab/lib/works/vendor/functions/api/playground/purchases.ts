import {body,db,guarded,HttpError,json,origin} from '../../_shared/platform/core';
import {requireMember} from '../../_shared/platform/memberAuth';
import {identifier,ensurePlayground} from '../../_shared/platform/playground';
import {testPrice} from '../../_shared/platform/playgroundOrders';
export const onRequestGet = guarded(async ({request,env}) => {
  const member = await requireMember(request,env,false);
  await ensurePlayground(env);
  let testAmount: number | null = null;
  try {testAmount = testPrice(env);} catch { /* Unconfigured test checkout stays disabled. */ }
  return json({testAmount,purchases:(await db(env).prepare('SELECT id,name,project_id,revision_id,amount,mode,status,created_at FROM eidos_pg_orders WHERE owner_id=? ORDER BY created_at DESC LIMIT 100').bind(member.id).all()).results});
});
export const onRequestPost = guarded(async ({request,env}) => {
  origin(request); const member = await requireMember(request,env,false), input = await body(request);
  await ensurePlayground(env);
  const order = await db(env).prepare("SELECT archive FROM eidos_pg_orders WHERE id=? AND owner_id=? AND status='paid' AND NOT EXISTS(SELECT 1 FROM eidos_pg_revoked WHERE payment_intent=eidos_pg_orders.payment_intent)").bind(identifier(input.id),member.id).first<{archive:string}>();
  if (!order) throw new HttpError(403,'This account has no paid entitlement to that export. A checkout return URL is not proof of payment.');
  const bytes = Uint8Array.from(atob(order.archive),c=>c.charCodeAt(0));
  return new Response(bytes,{headers:{'content-type':'application/zip','content-disposition':'attachment; filename="eidos-purchased-export.zip"','cache-control':'no-store','x-content-type-options':'nosniff'}});
});
