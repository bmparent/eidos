/** Repo-root operator helper. No provider calls or customer/account/payment writes. */
import { readFile,mkdir,writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { createClient } from '@libsql/client';
import { adaptDatabase } from '../lib/works/database';
import { boundOwnerCountExpense,type CountPricingEvidence } from '../lib/works/playground-diagnostic-billing';
import { ownerCountClaimId } from '../lib/works/playground-count-probe';
const args=Object.fromEntries(process.argv.slice(2).filter(s=>s.startsWith('--')&&s.includes('=')).map(s=>{const i=s.indexOf('=');return [s.slice(2,i),s.slice(i+1)];}));
if(!args.configuration||!args['pricing-evidence']||!args.out)throw Error('Supply --configuration, --pricing-evidence and a versioned --out.');
const out=path.resolve(args.out),root=path.resolve('artifacts/playground-authenticated-20261015');
if(!out.startsWith(root+path.sep))throw Error('Use a dedicated versioned artifact directory.');
const input=JSON.parse(await readFile(args['pricing-evidence'],'utf8'));
if(input.interpretation!=='pessimistic-token-price-ceiling'||input.actualCountCostMicros!==null||input.additionalCountCallsEnabled!==false)throw Error('The pricing receipt must preserve invoice uncertainty and disable further counts.');
const evidence:CountPricingEvidence={sourceUrl:input.sourceUrl,countGuideUrl:input.countGuideUrl,reviewedAt:input.reviewedAt,
  documentSha256:createHash('sha256').update(await readFile(args['pricing-evidence'])).digest('hex')};
const {env}=JSON.parse(await readFile(args.configuration,'utf8'));
if(env.EIDOS_DATABASE_BINDING_PREFIX!=='EIDOS_TEMPLATES_TEST'||!new URL(env.EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL).hostname.includes('validation'))throw Error('Refusing a database outside the approved TEST binding.');
const client=createClient({url:env.EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL,authToken:env.EIDOS_TEMPLATES_TEST_TURSO_AUTH_TOKEN}),platform={EIDOS_DB:adaptDatabase(client)};
const protectedTables=['eidos_email_members','eidos_orders','eidos_template_orders','eidos_pg_projects','eidos_pg_orders'];
const counts=async()=>Object.fromEntries(await Promise.all(protectedTables.map(async name=>[name,Number((await client.execute('SELECT COUNT(*) n FROM '+name)).rows[0].n)])));
const receipt:Record<string,unknown>={timestampUtc:new Date().toISOString(),sourceCommit:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),
  mode:'test',dryRun:!process.argv.includes('--apply'),providerCalls:0,providerKeyReadLocally:false,secretValuesPrinted:false,originalClaimId:ownerCountClaimId};
try{
  const before=await counts();receipt.protectedCountsBefore=before;
  const original=(await client.execute({sql:'SELECT detail_json FROM eidos_pg_ai_journal WHERE id=?',args:[ownerCountClaimId]})).rows[0].detail_json as string;
  receipt.originalClaimSha256=createHash('sha256').update(original).digest('hex');
  receipt.before=(await client.execute("SELECT enabled,provider_budget,paused_reason FROM eidos_pg_ai_policies WHERE mode='test'")).rows[0];
  if(process.argv.includes('--apply'))receipt.reconciliation=await boundOwnerCountExpense(platform,evidence);
  const preserved=(await client.execute({sql:'SELECT detail_json FROM eidos_pg_ai_journal WHERE id=?',args:[ownerCountClaimId]})).rows[0].detail_json;
  const after=await counts();receipt.protectedCountsAfter=after;
  if(original!==preserved||JSON.stringify(before)!==JSON.stringify(after))throw Error('Protected records changed.');
  receipt.originalClaimPreserved=true;receipt.protectedRecordsPreserved=true;receipt.passed=true;
}catch{receipt.passed=false;receipt.reason='Count reconciliation failed; private database details suppressed.';process.exitCode=1;}
finally{client.close();await mkdir(out,{recursive:true});await writeFile(path.join(out,process.argv.includes('--apply')?'count-pricing-reconciliation.json':'count-pricing-dry-run.json'),JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt));}
