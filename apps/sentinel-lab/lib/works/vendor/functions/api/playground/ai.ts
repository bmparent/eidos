import { body, guarded, HttpError, json, origin, type PlatformEnv } from '../../_shared/platform/core';
import { requireMember } from '../../_shared/platform/memberAuth';
import { aiBalance, cancelAiAction, ownedAiAction, publicAiAction, reconcileAiDeadlines, reconcileAiReversals, setAiLimit, validateAiBrief } from '../../_shared/platform/playgroundAi';
import { aiProviderAvailability, generateProviderAiAction, quoteProviderAiAction } from '../../_shared/platform/playgroundAiProvider';

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
  if (id) return json({ action: publicAiAction(await ownedAiAction(env, member.id, billingMode, id)), balance: await aiBalance(env, member.id, billingMode) });
  return json({ ...await aiProviderAvailability(env), balance: await aiBalance(env, member.id, billingMode) });
});
export const onRequestPost = guarded(async ({ request, env }) => {
  origin(request);
  const member = await requireMember(request, env, false), billingMode = mode(env), input = await body(request, 110000);
  if (input.action === 'limit') {
    if (Object.keys(input).some(key => !['action', 'limitMicros'].includes(key))) throw new HttpError(400, 'Only the signed-in billing owner can change this limit.');
    return json({ balance: await setAiLimit(env, member.id, billingMode, input.limitMicros) });
  }
  if (input.action === 'cancel' && typeof input.id === 'string') {
    if (Object.keys(input).some(key => !['action', 'id'].includes(key))) throw new HttpError(400, 'Invalid cancellation.');
    return json({ action: publicAiAction(await cancelAiAction(env, member.id, billingMode, input.id)) });
  }
  if (input.action === 'quote') {
    if (Object.keys(input).sort().join(',') !== 'action,brief,idempotencyKey' || typeof input.idempotencyKey !== 'string') throw new HttpError(400, 'Provide only a project brief and unique action reference.');
    const action = await quoteProviderAiAction(env, member.id, billingMode, input.idempotencyKey, validateAiBrief(input.brief));
    return json({ action: publicAiAction(action), balance: await aiBalance(env, member.id, billingMode) });
  }
  if (input.action === 'generate') {
    if (Object.keys(input).sort().join(',') !== 'action,approvedMaximumMicros,id' || typeof input.id !== 'string') throw new HttpError(400, 'Approve the quoted action and its exact maximum.');
    const action = await generateProviderAiAction(env, member.id, billingMode, input.id, input.approvedMaximumMicros);
    return json({ action: publicAiAction(action), balance: await aiBalance(env, member.id, billingMode) });
  }
  throw new HttpError(503, 'Eidos generation is not enabled. Manual editing and downloads remain available.');
});
