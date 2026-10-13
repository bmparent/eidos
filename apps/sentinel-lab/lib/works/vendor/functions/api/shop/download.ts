import { body, guarded, origin } from '../../_shared/platform/core';
import { receiptOrder } from '../../_shared/platform/shop';
import { kitDownload } from '../../_shared/platform/kitDelivery';
export const onRequestPost = guarded(async ({ request, env }) => {
  origin(request);
  const input = await body(request, 1000);
  const order = await receiptOrder(env, input.receipt);
  return kitDownload(env, order);
});
