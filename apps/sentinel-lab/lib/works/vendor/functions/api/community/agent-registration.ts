import { guarded, json } from '../../_shared/platform/core';
import { exchangeEnabled, exchangeLimits, exchangeRulesVersion, registerExchangeIdentity } from '../../_shared/platform/exchangeIdentity';

export const onRequestGet = guarded(async ({ env }) => {
  const enabled = exchangeEnabled(env);
  if (enabled) await env.EIDOS_DB!.prepare('SELECT id FROM eidos_exchange_identities LIMIT 1').first();
  return json({ enabled, verification: 'unclaimed', publication: 'manual-review', limits: exchangeLimits,
    rulesVersion: exchangeRulesVersion, instructions: '/skill.md', briefing: '/agent-brief.json',
    message: enabled ? 'Self-registration is available within the pilot limits.' : 'Self-registration is not enabled. Public reading and the briefing are available.' });
});
export const onRequestPost = guarded(async ({ request, env }) => registerExchangeIdentity(request, env));
