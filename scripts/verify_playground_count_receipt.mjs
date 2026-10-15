/** Read-only reconciliation evidence for the single approved operator count; never releases its hold. */
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import path from 'node:path';
const args=Object.fromEntries(process.argv.slice(2).filter(s=>s.startsWith('--')&&s.includes('=')).map(s=>{const i=s.indexOf('=');return [s.slice(2,i),s.slice(i+1)];}));
if(!args.configuration||!args.out)throw Error('Supply the approved ignored TEST --configuration and --out artifact directory.');
const root=path.resolve('artifacts/playground-authenticated-20261015'),out=path.resolve(args.out);
if(!out.startsWith(root+path.sep))throw Error('Use a versioned diagnostic artifact directory.');
const {env}=JSON.parse(await readFile(args.configuration,'utf8'));
if(env.EIDOS_DATABASE_BINDING_PREFIX!=='EIDOS_TEMPLATES_TEST'||!new URL(env.EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL).hostname.includes('validation'))throw Error('Unexpected TEST database.');
const require=createRequire(path.resolve('apps/sentinel-lab/package.json')),{createClient}=require('@libsql/client');
const client=createClient({url:env.EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL,authToken:env.EIDOS_TEMPLATES_TEST_TURSO_AUTH_TOKEN});
const receipt={timestampUtc:new Date().toISOString(),freshProcess:true,readOnly:true,secretValuesReadBack:false,actualCountCostUsd:null,billing:'unknown'};
try{
  const id='playground-owner-count-probe-20261015-sol';
  const rows=await client.execute({sql:'SELECT id,event,detail_json FROM eidos_pg_ai_journal WHERE id IN (?,?)',args:[id,id+'-result']});
  const holds=rows.rows.filter(r=>r.event==='diagnostic_count_hold'),results=rows.rows.filter(r=>r.event==='diagnostic_count_result');
  if(holds.length!==1||results.length!==1)throw Error('Missing unique durable count evidence.');
  const hold=JSON.parse(holds[0].detail_json),result=JSON.parse(results[0].detail_json),http=JSON.parse(await readFile(path.join(out,'count-http-receipt.json'),'utf8'));
  const hash=createHash('sha256').update(hold.requestJson).digest('hex'),request=JSON.parse(hold.requestJson);
  const policy=(await client.execute("SELECT enabled,provider_budget,paused_reason FROM eidos_pg_ai_policies WHERE mode='test'")).rows[0];
  const actions=Number((await client.execute("SELECT COUNT(*) count FROM eidos_pg_ai_actions WHERE mode='test'")).rows[0].count);
  const ordinaryClaims=Number((await client.execute("SELECT COUNT(*) count FROM eidos_pg_ai_count_claims WHERE mode='test'")).rows[0].count);
  const diagnosticClaims=Number((await client.execute("SELECT COUNT(*) count FROM eidos_pg_ai_journal WHERE mode='test' AND event='diagnostic_count_hold'")).rows[0].count);
  Object.assign(receipt,{diagnosticClaims,durableResults:results.length,actions,ordinaryCountClaims:ordinaryClaims,
    model:result.model,inputTokens:result.inputTokens,providerRequestId:result.providerRequestId,clientRequestId:result.clientRequestId,requestHash:hash,
    countPayloadBytes:Buffer.byteLength(hold.requestJson),countPayloadFields:Object.keys(request),providerHeldMicros:hold.providerHeldMicros,
    configuredBudgetMicros:Number(policy.provider_budget),generationEnabled:Boolean(policy.enabled),pausedReason:policy.paused_reason,
    remainingBudgetMicros:Number(policy.provider_budget)-hold.providerHeldMicros,generationRequests:result.generationRequests,
    receiptIdentityMatches:hold.clientRequestId===result.clientRequestId&&result.clientRequestId===http.clientRequestId&&result.providerRequestId===http.providerRequestId,
    payloadHashMatches:hash===hold.requestHash&&hash===result.requestHash&&hash===http.requestHash});
  receipt.passed=receipt.receiptIdentityMatches&&receipt.payloadHashMatches&&diagnosticClaims===1&&actions===0&&ordinaryClaims===0&&
    result.state==='counted'&&result.providerStatus===200&&result.model==='gpt-6.1-sol'&&Number.isSafeInteger(result.inputTokens)&&
    hold.providerHeldMicros===5000000&&receipt.configuredBudgetMicros===5000000&&!receipt.generationEnabled&&
    receipt.pausedReason==='owner_count_billing_unknown_entire_5_usd_held'&&result.generationRequests===0;
  // Preserve the actual HTTP bytes, including their lack of a trailing newline.
  await writeFile(path.join(out,'count-request.json'),hold.requestJson);
}catch{Object.assign(receipt,{passed:false,reason:'Durable count reconciliation failed; private database details suppressed.'});}
finally{client.close();await writeFile(path.join(out,'count-ledger-readback.json'),JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt));if(!receipt.passed)process.exitCode=1;}
