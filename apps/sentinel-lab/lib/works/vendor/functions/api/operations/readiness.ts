import { admin, db, guarded, json } from '../../_shared/platform/core';
import { catalogueRevision } from '../../_shared/platform/knowledge';
export const onRequestGet = guarded(async ({ request, env }) => {
  await admin(request, env);
  let database: 'ready' | 'unavailable' = 'unavailable';
  try { if (env.EIDOS_DB) { await db(env).prepare('SELECT 1 AS ok').first(); database = 'ready'; } } catch { /* no raw diagnostics */ }
  let outcomes:unknown[]=[];
  let dailyReservedTokens:number|null=null;
  try { dailyReservedTokens=(await db(env).prepare("SELECT used FROM eidos_quotas WHERE bucket='ai-global' AND period=?").bind(Math.floor(Date.now()/86400000)).first<{used:number}>())?.used ?? 0; } catch { /* unknown if the ledger cannot be read */ }
  try{outcomes=(await db(env).prepare('SELECT day,feature,outcome,count,latency_ms,updated FROM eidos_outcomes WHERE day>=? ORDER BY day DESC,feature').bind(new Date(Date.now()-86400000).toISOString().slice(0,10)).all()).results;}catch{/* no observations yet */}
  return json({ contract: 'works-v1', catalogueRevision, database,
    outcomes, outcomeMeaning:'HTTP acceptance is not inbox delivery or paid fulfillment. Source fallback may reflect disabled AI, limits or provider failure. Counts contain no prompts or identities.',
    sourceRevision: env.EIDOS_SOURCE_REVISION || 'unknown',
    assistant: { enabled: env.EIDOS_AI_ENABLED === 'true', model: env.EIDOS_ASSISTANT_MODEL || 'unknown', dailyReservedTokens, dailyTokenLimit: Math.min(100000, Math.max(0, Number(env.EIDOS_AI_DAILY_TOKENS || 20000) || 0)), limitSource:env.EIDOS_AI_DAILY_TOKENS?'runtime configuration':'source default', outputTokenLimit: 320, visitorDailyRequests: 5, globalDailyRequests: 200 },
    accounts: { enabled: env.EIDOS_ACCOUNTS_ENABLED === 'true', passwordEnabled: env.EIDOS_PASSWORD_AUTH_ENABLED === 'true', googleConfigured: Boolean(env.EIDOS_GOOGLE_CLIENT_ID && env.EIDOS_GOOGLE_CLIENT_SECRET && env.EIDOS_GOOGLE_REDIRECT_URI) },
    kit: { enabled: env.EIDOS_SHOP_ENABLED === 'true', mode: env.STRIPE_SECRET_KEY?.startsWith('sk_test_') ? 'test' : env.STRIPE_SECRET_KEY?.startsWith('sk_live_') || env.STRIPE_SECRET_KEY?.startsWith('rk_live_') ? 'live' : 'unknown' },
    playground: { authoringEnabled: env.EIDOS_PLAYGROUND_AUTHORING_ENABLED === 'true', checkoutMode: 'test-only' },
    snapshot: { readiness: 'not accepted for sale' },
  });
});
