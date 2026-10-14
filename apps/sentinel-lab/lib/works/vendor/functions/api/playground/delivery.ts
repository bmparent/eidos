import { body, guarded, HttpError, json, origin } from '../../_shared/platform/core';
import { requireMember } from '../../_shared/platform/memberAuth';
import { DELIVERY_LIMITS, freezeProjectDelivery, readProjectDelivery, requireProjectDelivery } from '../../_shared/platform/projectDelivery';
import { projectDeliveryCapabilities } from '../../_shared/platform/templateShop';

export const onRequestPost = guarded(async ({ request, env }) => {
  origin(request);
  const member = await requireMember(request, env, false);
  requireProjectDelivery(env);
  const input = await body(request, DELIVERY_LIMITS.requestBytes);
  return json({ delivery: await freezeProjectDelivery(env, member.id, input) });
});
export const onRequestGet = guarded(async ({ request, env }) => {
  const member = await requireMember(request, env, false);
  requireProjectDelivery(env);
  const url = new URL(request.url);
  if (url.search === '?capabilities=1') return json({ capabilities: await projectDeliveryCapabilities(request, env, member.id) });
  if ([...url.searchParams.keys()].some(key => key !== 'id') || url.searchParams.getAll('id').length !== 1) throw new HttpError(400, 'Provide one frozen project reference.');
  return json({ delivery: await readProjectDelivery(env, member.id, url.searchParams.get('id')) });
});
