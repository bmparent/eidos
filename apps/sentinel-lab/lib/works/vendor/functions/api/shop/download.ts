import { body, guarded, HttpError, origin } from '../../_shared/platform/core';
import { receiptOrder } from '../../_shared/platform/shop';
import { kitArchiveBase64 } from '../../_shared/platform/kitArchive';
export const onRequestPost = guarded(async ({ request, env }) => {
  origin(request);
  const input = await body(request, 1000);
  const order = await receiptOrder(env, input.receipt);
  if (order.status !== 'paid')
    throw new HttpError(
      403,
      'A verified, completed purchase is required to download this package.',
    );
  const bytes = Uint8Array.from(atob(kitArchiveBase64), (c) => c.charCodeAt(0));
  return new Response(bytes, {
    headers: {
      'content-type': 'application/zip',
      'content-disposition':
        'attachment; filename="eidos-cinematic-starter-v1.zip"',
      'cache-control': 'private, no-store',
      'referrer-policy': 'no-referrer',
      'x-content-type-options': 'nosniff',
    },
  });
});
