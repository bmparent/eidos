import { guarded, json, local, communityAvailable } from '../_shared/platform/core';
export const onRequestGet = guarded(async ({ request, env }) =>
  json({
    gaMeasurementId: /^G-[A-Z0-9]{5,20}$/.test(env.GA_MEASUREMENT_ID || '')
      ? env.GA_MEASUREMENT_ID
      : '',
    turnstileSiteKey: env.TURNSTILE_SITE_KEY || '',
    communityReady: communityAvailable(request,env),
    aiReady: Boolean(
      env.EIDOS_RUNTIME === 'sentinel' &&
      env.EIDOS_AI_ENABLED === 'true' &&
        env.OPENAI_API_KEY &&
        env.EIDOS_ASSISTANT_MODEL &&
        env.EIDOS_DB &&
        env.EIDOS_RATE_SECRET,
    ),
    shopReady: Boolean(
      env.EIDOS_SHOP_ENABLED === 'true' &&
        env.STRIPE_SECRET_KEY &&
        env.EIDOS_KIT_WEBHOOK_SECRET &&
        env.EIDOS_DB &&
        env.EIDOS_RATE_SECRET &&
        ((env.TURNSTILE_SECRET_KEY && env.TURNSTILE_SITE_KEY) ||
          local(request, env)),
    ),
    localTest: local(request, env),
  }),
);
