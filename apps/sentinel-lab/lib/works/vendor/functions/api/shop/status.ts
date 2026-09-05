import { body, guarded, json, origin } from '../../_shared/platform/core';
import { receiptOrder } from '../../_shared/platform/shop';
export const onRequestPost = guarded(async ({ request, env }) => {
  origin(request);
  const input = await body(request, 1000);
  const order = await receiptOrder(env, input.receipt);
  return json({ status: order.status, orderId: order.id });
});
