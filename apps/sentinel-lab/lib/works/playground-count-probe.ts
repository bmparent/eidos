import { platformDatabase } from './database';
import { authorizePlaygroundDiagnostic, diagnosticReply as reply } from './playground-provider-check';
import { buildAiProviderRequest, providerTestRateCard } from './vendor/functions/_shared/platform/playgroundAiProvider';
import { ensureAiLedger, USD } from './vendor/functions/_shared/platform/playgroundAi';
import { hash, readText, type Database } from './vendor/functions/_shared/platform/core';
import { createProject } from './project-delivery/src/playground/model';
import { createProposalContext } from './project-delivery/src/playground/aiOperations';

const claimId = 'playground-owner-count-probe-20261015-sol';
/** One operator-only count. Unknown billing blocks the entire TEST budget indefinitely. */
export async function probePlaygroundCount(request: Request, source: Record<string,string|undefined> = process.env, transport: typeof fetch = fetch, suppliedDb?: Database) {
  const denied = authorizePlaygroundDiagnostic(request,source,'test-count-one');
  if (denied) return denied;
  if (!source.EIDOS_PLAYGROUND_OPENAI_API_KEY || source.EIDOS_PLAYGROUND_OPENAI_API_KEY.length < 20) return reply({ error:'dedicated_key_missing' },503);
  const database = suppliedDb || platformDatabase(source);
  if (!database || (!suppliedDb && source.EIDOS_DATABASE_BINDING_PREFIX !== 'EIDOS_TEMPLATES_TEST')) return reply({ error:'approved_test_database_required' },503);
  const env={EIDOS_DB:database}, project=createProject(), section=project.sections.find(s=>s.id==='hero')!;
  const built=buildAiProviderRequest({documentId:'owner-count-probe-public-fixture',revision:'1',scope:'section',targetId:section.id,
    instruction:'Clarify this public sample headline.',context:JSON.stringify(createProposalContext(project,section.id,'','section'))},providerTestRateCard('gpt-6.1-sol'));
  const requestHash=await hash(built.countSerialized), now=new Date().toISOString(),clientRequestId=crypto.randomUUID();
  try {
    await ensureAiLedger(env);
    // All customer/provider routes remain off. Reconfiguration preserves the pause.
    // This claims the complete $5 allowance for uncertainty; it is not an asserted fee.
    await database.batch([
      database.prepare(`INSERT OR IGNORE INTO eidos_pg_ai_journal(id,owner_id,mode,action_id,event,detail_json,created_at)
        SELECT ?,NULL,'test',NULL,'diagnostic_count_hold',?,? FROM eidos_pg_ai_policies p
        WHERE p.mode='test' AND p.enabled=0 AND p.paused_reason IS NULL AND p.provider_budget=?
          AND NOT EXISTS(SELECT 1 FROM eidos_pg_ai_actions WHERE mode='test' AND provider_status!='none')`)
        .bind(claimId,JSON.stringify({providerHeldMicros:5*USD,billing:'unknown',requestHash,clientRequestId,model:'gpt-6.1-sol',requestJson:built.countSerialized}),now,5*USD),
      database.prepare(`UPDATE eidos_pg_ai_policies SET enabled=0,paused_reason='owner_count_billing_unknown_entire_5_usd_held',updated_at=?
        WHERE mode='test' AND EXISTS(SELECT 1 FROM eidos_pg_ai_journal WHERE id=?)`).bind(now,claimId),
    ]);
    const claimed=await database.prepare("SELECT json_extract(detail_json,'$.clientRequestId') clientId FROM eidos_pg_ai_journal WHERE id=?").bind(claimId).first<{clientId:string}>();
    if (claimed?.clientId!==clientRequestId) return reply({ submitted:false,replayed:true,billing:'unknown',providerHeldMicros:5*USD,generationRequests:0 },409);
  } catch { return reply({error:'durable_budget_claim_failed',generationRequests:0},503); }
  const receipt:Record<string,unknown>={submitted:true,model:'gpt-6.1-sol',requestHash,clientRequestId,providerHeldMicros:5*USD,billing:'unknown',generationRequests:0};
  try {
    const response=await transport('https://api.openai.com/v1/responses/input_tokens',{method:'POST',redirect:'error',signal:AbortSignal.timeout(15000),
      headers:{authorization:'Bearer '+source.EIDOS_PLAYGROUND_OPENAI_API_KEY,'content-type':'application/json','x-client-request-id':clientRequestId},body:built.countSerialized});
    const id=response.headers.get('x-request-id');
    Object.assign(receipt,{providerStatus:response.status,providerRequestId:id&&/^[!-~]{1,200}$/.test(id)?id:null});
    const value=JSON.parse(await readText(response,16000));
    if (!response.ok || value.object!=='response.input_tokens' || !Number.isSafeInteger(value.input_tokens) || value.input_tokens<0) throw Error('count_invalid');
    Object.assign(receipt,{state:'counted',inputTokens:value.input_tokens});
  } catch { Object.assign(receipt,{state:'unknown_or_rejected',automaticRetry:false}); }
  try {
    await database.prepare('INSERT INTO eidos_pg_ai_journal(id,owner_id,mode,action_id,event,detail_json,created_at) VALUES(?,NULL,?,NULL,?,?,?)')
      .bind(claimId+'-result','test','diagnostic_count_result',JSON.stringify(receipt),new Date().toISOString()).run();
  } catch { return reply({...receipt,state:'result_persistence_unknown',automaticRetry:false},503); }
  return reply(receipt,receipt.state==='counted'?200:502);
}
