import { platformDatabase } from './database';
import { authorizePlaygroundDiagnostic, diagnosticReply as reply } from './playground-provider-check';
import { ownerCountClaimId, ownerCountRequest } from './playground-count-probe';
import { hash, readText, record, type Database } from './vendor/functions/_shared/platform/core';
import { diagnosticProviderLiability, ensureAiLedger, meterAiUsage, providerMaximum, USD } from './vendor/functions/_shared/platform/playgroundAi';
import { parseAiProviderUsage, providerTestRateCard } from './vendor/functions/_shared/platform/playgroundAiProvider';
import { applyContextProposal, validateProposal } from './vendor/functions/_shared/platform/playgroundAiRenderer';

export const ownerGenerationClaimId=ownerCountClaimId+'-generation';
const ascii=(s:unknown):s is string=>typeof s==='string'&&/^[!-~]{1,200}$/.test(s);
/** One fixed public sample, reusing its exact durable count. No new count,
 * customer funding, payment, account, automatic retry or public AI activation. */
export async function probePlaygroundGeneration(request: Request, source: Record<string,string|undefined> = process.env, transport: typeof fetch = fetch, suppliedDb?: Database) {
  const denied=authorizePlaygroundDiagnostic(request,source,'test-generation-one');if(denied)return denied;
  if(!source.EIDOS_PLAYGROUND_OPENAI_API_KEY||source.EIDOS_PLAYGROUND_OPENAI_API_KEY.length<20)return reply({error:'dedicated_key_missing'},503);
  const database=suppliedDb||platformDatabase(source);
  if(!database||(!suppliedDb&&source.EIDOS_DATABASE_BINDING_PREFIX!=='EIDOS_TEMPLATES_TEST'))return reply({error:'approved_test_database_required'},503);
  const env={EIDOS_DB:database},built=ownerCountRequest(),card=providerTestRateCard('gpt-6.1-sol'),now=new Date().toISOString();
  if(Date.parse(now)>=Date.parse(card.validUntil))return reply({error:'reviewed_rate_card_expired'},503);
  const requestHash=await hash(built.serialized),countHash=await hash(built.countSerialized),clientRequestId=crypto.randomUUID(),maximum=providerMaximum(card);
  try {
    await ensureAiLedger(env);
    // Count identity, result, frozen input and reconciled liability are checked
    // INSIDE the same SQLite write that reserves the generation maximum.
    const detail={providerHeldMicros:maximum,billing:'generation_pending',clientRequestId,requestHash,countRequestHash:countHash,requestJson:built.serialized,rateCard:card,model:card.model};
    await database.prepare(`INSERT OR IGNORE INTO eidos_pg_ai_journal(id,owner_id,mode,action_id,event,detail_json,created_at)
      SELECT ?,NULL,'test',NULL,'diagnostic_generation_hold',?,? FROM eidos_pg_ai_policies p
      WHERE p.mode='test' AND p.enabled=0 AND p.paused_reason IS NULL AND p.provider_budget<=${5*USD}
        AND p.provider_budget>=?+${diagnosticProviderLiability('p.mode')}+
          (SELECT COALESCE(SUM(CASE WHEN provider_status='held' THEN provider_maximum WHEN provider_status IN ('settled','pessimistic') THEN provider_cost ELSE 0 END),0) FROM eidos_pg_ai_actions WHERE mode=p.mode)
        AND EXISTS(SELECT 1 FROM eidos_pg_ai_journal h JOIN eidos_pg_ai_journal r ON r.id=h.id||'-result'
          JOIN eidos_pg_ai_diagnostic_costs c ON c.claim_id=h.id AND c.mode=h.mode
          WHERE h.id=? AND h.mode=p.mode AND h.event='diagnostic_count_hold' AND r.event='diagnostic_count_result'
            AND json_extract(h.detail_json,'$.requestHash')=? AND json_extract(h.detail_json,'$.requestJson')=?
            AND json_extract(r.detail_json,'$.requestHash')=? AND json_extract(r.detail_json,'$.state')='counted'
            AND json_extract(r.detail_json,'$.providerStatus')=200 AND json_extract(r.detail_json,'$.model')=?
            AND json_extract(r.detail_json,'$.clientRequestId')=json_extract(h.detail_json,'$.clientRequestId')
            AND json_extract(r.detail_json,'$.inputTokens') BETWEEN 0 AND ?)`)
      .bind(ownerGenerationClaimId,JSON.stringify(detail),now,maximum,ownerCountClaimId,countHash,built.countSerialized,countHash,card.model,card.maxInputTokens).run();
    const claimed=await database.prepare("SELECT json_extract(detail_json,'$.clientRequestId') clientId FROM eidos_pg_ai_journal WHERE id=?").bind(ownerGenerationClaimId).first<{clientId:string}>();
    if(claimed?.clientId!==clientRequestId)return reply({submitted:false,generationRequests:0,automaticRetry:false,error:'generation_claim_unavailable'},409);
  }catch{return reply({error:'durable_generation_claim_failed',generationRequests:0},503);}
  const receipt:Record<string,unknown>={submitted:true,model:card.model,requestHash,countRequestHash:countHash,clientRequestId,providerHeldMicros:maximum,
    generationRequests:1,countRequests:0,automaticRetry:false,sourceCommit:source.EIDOS_SOURCE_COMMIT||null,bundleHash:source.EIDOS_SOURCE_BUNDLE_HASH||null};
  let meteredCost:number|undefined,usable=false,boundExceeded=false;
  const started=Date.now();
  try {
    const response=await transport('https://api.openai.com/v1/responses',{method:'POST',redirect:'error',signal:AbortSignal.timeout(card.maxRuntimeMs),
      headers:{authorization:'Bearer '+source.EIDOS_PLAYGROUND_OPENAI_API_KEY,'content-type':'application/json','x-client-request-id':clientRequestId},body:built.serialized});
    const header=response.headers.get('x-request-id');
    Object.assign(receipt,{providerStatus:response.status,providerRequestId:ascii(header)?header:null});
    const result:unknown=JSON.parse(await readText(response,250000));
    if(record(result)&&record(result.error)) {
      const error=result.error;
      receipt.providerError=Object.fromEntries(['type','code','param'].filter(k=>typeof error[k]==='string'&&/^[a-zA-Z0-9_.-]{1,100}$/.test(String(error[k]))).map(k=>[k,error[k]]));
    }
    if(!record(result)||!ascii(result.id)||!ascii(header))throw Error('provider_identity');
    Object.assign(receipt,{providerId:result.id,returnedModel:result.model,returnedServiceTier:result.service_tier,
      providerResponse:{id:result.id,model:result.model,service_tier:result.service_tier,status:result.status,usage:result.usage,output:result.output}});
    const usage=parseAiProviderUsage(result.usage);
    if(result.model!==card.model||result.service_tier!=='default')throw Error('provider_route');
    const output=Array.isArray(result.output)?result.output:[];
    if(output.some(item=>!record(item)||!['message','reasoning'].includes(String(item.type))))throw Error('unexpected_paid_operation');
    meteredCost=meterAiUsage(usage,card);
    boundExceeded=meteredCost>maximum||usage.inputTokens>card.maxInputTokens||usage.outputTokens>card.maxOutputTokens;
    Object.assign(receipt,{usage,providerCostMicros:meteredCost,billing:'settled_model_tokens'});
    try {
      if(boundExceeded||!response.ok||result.status!=='completed')throw Error('provider_incomplete_or_bound_exceeded');
      const messages=output.filter(item=>record(item)&&item.type==='message');
      if(messages.length!==1||!record(messages[0])||messages[0].role!=='assistant'||messages[0].status!=='completed'||!Array.isArray(messages[0].content))throw Error('provider_message');
      const content=messages[0].content;
      if(content.length!==1||!record(content[0])||content[0].type!=='output_text'||typeof content[0].text!=='string')throw Error('provider_content');
      const proposal=validateProposal(JSON.parse(content[0].text));applyContextProposal(built.context,proposal);
      receipt.proposal=proposal;usable=true;
    }catch{receipt.technicalFailure='unusable_proposal';}
    receipt.state=usable?'generated':'technical_waiver';
    receipt.boundExceeded=boundExceeded;
  }catch{Object.assign(receipt,{state:'reconciliation_pending',billing:'unknown_liability_retained'});meteredCost=undefined;}
  Object.assign(receipt,{usable,providerRoundTripMs:Date.now()-started,customerChargeMicros:0,customerFundingTested:false});
  try {
    await database.batch([
      database.prepare("INSERT INTO eidos_pg_ai_journal(id,owner_id,mode,action_id,event,detail_json,created_at) VALUES(?,NULL,'test',NULL,'diagnostic_generation_result',?,?)")
        .bind(ownerGenerationClaimId+'-result',JSON.stringify(receipt),new Date().toISOString()),
      ...(meteredCost!==undefined?[database.prepare("INSERT INTO eidos_pg_ai_diagnostic_costs(claim_id,mode,status,provider_cost,evidence_json,created_at) VALUES(?,'test','settled',?,?,?)")
        .bind(ownerGenerationClaimId,meteredCost,JSON.stringify({providerRequestId:receipt.providerRequestId,providerId:receipt.providerId,requestHash,usage:receipt.usage}),new Date().toISOString())]
        :[database.prepare("UPDATE eidos_pg_ai_policies SET enabled=0,paused_reason='owner_generation_usage_requires_reconciliation',updated_at=? WHERE mode='test'").bind(new Date().toISOString())]),
      ...(boundExceeded?[database.prepare("UPDATE eidos_pg_ai_policies SET enabled=0,paused_reason='owner_generation_provider_bound_exceeded',updated_at=? WHERE mode='test'").bind(new Date().toISOString())]:[]),
    ]);
  }catch{return reply({...receipt,state:'result_persistence_unknown',billing:'liability_retained'},503);}
  return reply(receipt,usable?200:502);
}
