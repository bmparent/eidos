import { body, guarded, json, origin } from '../../_shared/platform/core';
import { receiptOrder } from '../../_shared/platform/shop';
import { templateOrder, templateStatus } from '../../_shared/platform/templateShop';
export const onRequestPost = guarded(async ({ request, env }) => {
  origin(request);
  const input = await body(request, 1000);
  const order = await receiptOrder(env, input.receipt);
  const purchasedTemplate = await templateOrder(env, order.id);
  if (purchasedTemplate) return templateStatus(request, env, purchasedTemplate);
  return json({ status: order.status, orderId: order.id });
});
