import { body, guarded, HttpError, json, origin, type PlatformEnv } from '../../_shared/platform/core';
import { requireMember } from '../../_shared/platform/memberAuth';
import { aiBalance, cancelAiAction, ownedAiAction, publicAiAction, reconcileAiDeadlines, reconcileAiReversals, setAiLimit } from '../../_shared/platform/playgroundAi';

function mode(env: PlatformEnv) {
  if (env.EIDOS_PLAYGROUND_AI_MODE !== 'test') throw new HttpError(503, 'AI billing is not configured. Manual editing and downloads remain available.');
  return 'test' as const;
}
export const onRequestGet = guarded(async ({ request, env }) => {
  const member = await requireMember(request, env, false), billingMode = mode(env);
  await reconcileAiDeadlines(env); await reconcileAiReversals(env);
  const url = new URL(request.url);
  if ([...url.searchParams.keys()].some(key => key !== 'id')) throw new HttpError(400, 'Billing scope is derived from the signed-in account.');
  const id = url.searchParams.get('id');
  if (id) return json({ action: publicAiAction(await ownedAiAction(env, member.id, billingMode, id)) });
  return json({ enabled: false, reason: 'Provider access, owner budget and model quality acceptance are pending.', balance: await aiBalance(env, member.id, billingMode) });
});
export const onRequestPost = guarded(async ({ request, env }) => {
  origin(request);
  const member = await requireMember(request, env, false), billingMode = mode(env), input = await body(request, 12000);
  if (input.action === 'limit') {
    if (Object.keys(input).some(key => !['action', 'limitMicros'].includes(key))) throw new HttpError(400, 'Only the signed-in billing owner can change this limit.');
    return json({ balance: await setAiLimit(env, member.id, billingMode, input.limitMicros) });
  }
  if (input.action === 'cancel' && typeof input.id === 'string') {
    if (Object.keys(input).some(key => !['action', 'id'].includes(key))) throw new HttpError(400, 'Invalid cancellation.');
    return json({ action: publicAiAction(await cancelAiAction(env, member.id, billingMode, input.id)) });
  }
  // Deliberately no provider adapter/quote/submission endpoint until the credential and hosted acceptance gates pass.
  throw new HttpError(503, 'Eidos generation is not enabled. Manual editing and downloads remain available.');
});
