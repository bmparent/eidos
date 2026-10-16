/** Fresh-process readback. Never submits, retries, releases or enables anything. */
import { readFile,writeFile } from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { createClient } from '@libsql/client';
import { adaptDatabase } from '../lib/works/database';
import { ownerCountClaimId,ownerCountRequest } from '../lib/works/playground-count-probe';
import { ownerGenerationClaimId } from '../lib/works/playground-generation-probe';
import { aiPolicyStatus } from '../lib/works/vendor/functions/_shared/platform/playgroundAi';
const args=Object.fromEntries(process.argv.slice(2).filter(s=>s.startsWith('--')&&s.includes('=')).map(s=>{const i=s.indexOf('=');return [s.slice(2,i),s.slice(i+1)];}));
if(!args.configuration||!args.out)throw Error('Supply the approved --configuration and versioned --out.');
const out=path.resolve(args.out),root=path.resolve('artifacts/playground-authenticated-20261015');
if(!out.startsWith(root+path.sep))throw Error('Use a dedicated versioned artifact directory.');
const {env}=JSON.parse(await readFile(args.configuration,'utf8'));
if(env.EIDOS_DATABASE_BINDING_PREFIX!=='EIDOS_TEMPLATES_TEST'||!new URL(env.EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL).hostname.includes('validation'))throw Error('Unexpected TEST database.');
const client=createClient({url:env.EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL,authToken:env.EIDOS_TEMPLATES_TEST_TURSO_AUTH_TOKEN}),platform={EIDOS_DB:adaptDatabase(client)};
const sha=(s:string)=>createHash('sha256').update(s).digest('hex');
const receipt:Record<string,unknown>={timestampUtc:new Date().toISOString(),freshProcess:true,readOnly:true,providerCalls:0,providerKeyReadLocally:false,secretValuesPrinted:false};
try{
  const rows=(await client.execute({sql:'SELECT id,event,detail_json FROM eidos_pg_ai_journal WHERE id IN (?,?,?,?)',args:[ownerCountClaimId,ownerCountClaimId+'-result',ownerGenerationClaimId,ownerGenerationClaimId+'-result']})).rows;
  const detail=(id:string)=>JSON.parse(rows.find(r=>r.id===id)!.detail_json as string);
  const count=detail(ownerCountClaimId),countResult=detail(ownerCountClaimId+'-result'),hold=detail(ownerGenerationClaimId),result=detail(ownerGenerationClaimId+'-result');
  const http=JSON.parse(await readFile(path.join(out,'generation-http-receipt.json'),'utf8')).provider;
  const policy=await aiPolicyStatus(platform,'test');
  const diagnostics=Number((await client.execute("SELECT COUNT(*) n FROM eidos_pg_ai_journal WHERE event='diagnostic_count_hold'")).rows[0].n);
  const ordinary=Number((await client.execute('SELECT COUNT(*) n FROM eidos_pg_ai_count_claims')).rows[0].n);
  const actions=Number((await client.execute('SELECT COUNT(*) n FROM eidos_pg_ai_actions')).rows[0].n);
  const protectedCounts=Object.fromEntries(await Promise.all(['eidos_email_members','eidos_orders','eidos_template_orders','eidos_pg_projects','eidos_pg_orders'].map(async table=>[table,Number((await client.execute('SELECT COUNT(*) n FROM '+table)).rows[0].n)])));
  const identityMatches=hold.clientRequestId===result.clientRequestId&&http.clientRequestId===result.clientRequestId&&http.providerRequestId===result.providerRequestId;
  const frozenHashesMatch=sha(count.requestJson)===count.requestHash&&count.requestHash===countResult.requestHash&&count.requestHash===hold.countRequestHash&&
    sha(hold.requestJson)===hold.requestHash&&hold.requestHash===result.requestHash&&hold.requestJson===ownerCountRequest().serialized;
  Object.assign(receipt,{policy,diagnosticCounts:diagnostics,ordinaryCounts:ordinary,customerActions:actions,protectedCounts,identityMatches,frozenHashesMatch,
    originalCountRequestId:countResult.providerRequestId,inputCount:countResult.inputTokens,generation:result,countCostStatus:'pessimistic_ceiling',actualCountCostMicros:null,
    providerMaximumRetained:result.state==='reconciliation_pending',customerFundingTested:false,additionalCounts:0});
  receipt.passed=identityMatches&&frozenHashesMatch&&diagnostics===1&&ordinary===0&&actions===0&&result.state==='generated'&&
    result.usable===true&&result.countRequests===0&&result.generationRequests===1&&policy.committed<=5000000&&policy.configuredBudget===5000000&&policy.enabled===false;
  await writeFile(path.join(out,'generation-request.json'),hold.requestJson);
  if(result.providerResponse)await writeFile(path.join(out,'generation-provider-response.json'),JSON.stringify(result.providerResponse,null,2)+'\n');
}catch{receipt.passed=false;receipt.reason='Generation readback failed; private database details suppressed.';}
finally{client.close();await writeFile(path.join(out,'generation-ledger-readback.json'),JSON.stringify(receipt,null,2)+'\n');
  console.log(JSON.stringify({passed:receipt.passed,policy:receipt.policy,state:(receipt.generation as Record<string,unknown>|undefined)?.state||null,identityMatches:receipt.identityMatches,frozenHashesMatch:receipt.frozenHashesMatch}));if(!receipt.passed)process.exitCode=1;}
