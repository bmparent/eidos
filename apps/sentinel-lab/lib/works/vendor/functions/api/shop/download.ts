import { body, guarded, HttpError, origin } from '../../_shared/platform/core';
import { receiptOrder } from '../../_shared/platform/shop';
import { kitDownload } from '../../_shared/platform/kitDelivery';
import { templateDownload, templateOrder } from '../../_shared/platform/templateShop';
export const onRequestPost = guarded(async ({ request, env }) => {
  origin(request);
  const input = await body(request, 1000);
  const order = await receiptOrder(env, input.receipt);
  const purchasedTemplate = await templateOrder(env, order.id);
  if (purchasedTemplate) return templateDownload(request, env, purchasedTemplate, input.downloadToken, input.artifact);
  if (input.artifact !== undefined && input.artifact !== 'original') throw new HttpError(400, 'This purchase has no approved page attachment.');
  return kitDownload(env, order);
});
