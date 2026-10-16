import { db, hash, HttpError, type PlatformEnv } from './vendor/functions/_shared/platform/core';
import { aiPolicyStatus, diagnosticProviderLiability, ensureAiLedger, money, USD } from './vendor/functions/_shared/platform/playgroundAi';
import { providerTestRateCard } from './vendor/functions/_shared/platform/playgroundAiProvider';
import { ownerCountClaimId, ownerCountRequest } from './playground-count-probe';

export interface CountPricingEvidence {
  sourceUrl: 'https://developers.openai.com/api/docs/pricing';
  countGuideUrl: 'https://developers.openai.com/api/docs/guides/token-counting';
  documentSha256: string;
  reviewedAt: string;
}
/** Server/operator only. Replaces the uncertainty hold with a pessimistic
 * token-price ceiling, never a zero-fee assertion or a settled invoice. No I/O. */
export async function boundOwnerCountExpense(env: PlatformEnv, evidence: CountPricingEvidence) {
  if(evidence.sourceUrl!=='https://developers.openai.com/api/docs/pricing'||
    evidence.countGuideUrl!=='https://developers.openai.com/api/docs/guides/token-counting'||
    !/^[a-f0-9]{64}$/.test(evidence.documentSha256)||!Number.isFinite(Date.parse(evidence.reviewedAt)))
    throw new HttpError(400,'Supply the reviewed official pricing receipt.');
  await ensureAiLedger(env);
  const rows=await db(env).prepare('SELECT id,event,detail_json FROM eidos_pg_ai_journal WHERE mode=? AND id IN (?,?)')
    .bind('test',ownerCountClaimId,ownerCountClaimId+'-result').all<{id:string;event:string;detail_json:string}>();
  const hold=rows.results?.find(r=>r.event==='diagnostic_count_hold'), result=rows.results?.find(r=>r.event==='diagnostic_count_result');
  if(!hold||!result)throw new HttpError(409,'The unique count claim and result are required.');
  const h=JSON.parse(hold.detail_json), r=JSON.parse(result.detail_json), built=ownerCountRequest();
  if(r.state!=='counted'||r.providerStatus!==200||r.model!=='gpt-6.1-sol'||!Number.isSafeInteger(r.inputTokens)||r.inputTokens<0||
    r.clientRequestId!==h.clientRequestId||r.requestHash!==h.requestHash||h.requestHash!==await hash(built.countSerialized)||
    h.requestJson!==built.countSerialized||!/^req_[a-zA-Z0-9_-]{1,190}$/.test(r.providerRequestId))
    throw new HttpError(409,'The count evidence does not match the frozen public request.');
  const card=providerTestRateCard('gpt-6.1-sol');
  if(Date.parse(evidence.reviewedAt)>=Date.parse(card.validUntil))throw new HttpError(503,'Review provider prices before extending this bound.');
  const highestRate=Math.max(card.inputMicrosPerMillion,card.cachedInputMicrosPerMillion!,card.cacheWriteMicrosPerMillion!);
  const amount=money(Number((BigInt(r.inputTokens)*BigInt(highestRate)+BigInt(USD)-1n)/BigInt(USD)));
  const detail={...evidence,kind:'pricing-ceiling',claimId:ownerCountClaimId,providerRequestId:r.providerRequestId,requestHash:h.requestHash,
    inputTokens:r.inputTokens,rateCard:card,providerCostMicros:amount,actualCostMicros:null,
    interpretation:'Responses API has no separate API charge; retain every counted token at the highest reviewed input-category rate. This is a conservative inference, not count-specific fee confirmation.',
    additionalCountCallsEnabled:false};
  const encoded=JSON.stringify(detail),now=new Date().toISOString();
  await db(env).batch([
    db(env).prepare("INSERT OR IGNORE INTO eidos_pg_ai_diagnostic_costs(claim_id,mode,status,provider_cost,evidence_json,created_at) VALUES(?,'test','bounded',?,?,?)")
      .bind(ownerCountClaimId,amount,encoded,now),
    db(env).prepare("INSERT OR IGNORE INTO eidos_pg_ai_journal(id,owner_id,mode,action_id,event,detail_json,created_at) VALUES(?,NULL,'test',NULL,'diagnostic_count_pricing_bound',?,?)")
      .bind(ownerCountClaimId+'-pricing-bound',encoded,now),
  ]);
  const saved=await db(env).prepare("SELECT provider_cost,evidence_json FROM eidos_pg_ai_diagnostic_costs WHERE claim_id=? AND status='bounded'")
    .bind(ownerCountClaimId).first<{provider_cost:number;evidence_json:string}>();
  if(!saved||saved.provider_cost!==amount||saved.evidence_json!==encoded)throw new HttpError(409,'A different count-pricing receipt is already recorded.');
  // Preserve every unrelated pause and the disabled customer policy. The new
  // bound is included in this write and all customer/owner admission writes.
  await db(env).prepare(`UPDATE eidos_pg_ai_policies AS p SET paused_reason=NULL,enabled=0,updated_at=?
    WHERE p.mode='test' AND p.paused_reason='owner_count_billing_unknown_entire_5_usd_held' AND p.provider_budget<=${5*USD}
      AND ${diagnosticProviderLiability('p.mode','held')}=0
      AND p.provider_budget>=${diagnosticProviderLiability('p.mode')}+
        (SELECT COALESCE(SUM(CASE WHEN provider_status='held' THEN provider_maximum WHEN provider_status IN ('settled','pessimistic') THEN provider_cost ELSE 0 END),0) FROM eidos_pg_ai_actions WHERE mode=p.mode)`)
    .bind(now).run();
  return {count:detail,policy:await aiPolicyStatus(env,'test')};
}
