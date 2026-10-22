import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createClient } from '@libsql/client';
import { adaptDatabase } from '../lib/works/database';
import { probePlaygroundCount,ownerCountClaimId,ownerCountRequest } from '../lib/works/playground-count-probe';
import { boundOwnerCountExpense } from '../lib/works/playground-diagnostic-billing';
import { probePlaygroundGeneration,ownerGenerationClaimId } from '../lib/works/playground-generation-probe';
import { aiPolicyStatus,configureAiPolicy,ensureAiLedger,USD } from '../lib/works/vendor/functions/_shared/platform/playgroundAi';
import { providerTestRateCard } from '../lib/works/vendor/functions/_shared/platform/playgroundAiProvider';

const source={VERCEL_ENV:'preview',EIDOS_PLAYGROUND_PROVIDER_DIAGNOSTICS:'test-generation-one',EIDOS_PLATFORM_TOKEN:'operator-fixture-token-with-at-least-32-characters',EIDOS_PLAYGROUND_OPENAI_API_KEY:'private-provider-fixture-never-exposed'};
const req=()=>new Request('https://preview.test/api/playground-provider-generation',{method:'POST',headers:{'x-eidos-platform-token':source.EIDOS_PLATFORM_TOKEN,'x-eidos-site-origin':'https://eidosworks-playground-20261014.pages.dev'}});
const evidence={sourceUrl:'https://developers.openai.com/api/docs/pricing' as const,countGuideUrl:'https://developers.openai.com/api/docs/guides/token-counting' as const,documentSha256:'a'.repeat(64),reviewedAt:'2026-10-15T22:00:00Z'};
// Exercise the immutable reviewed fixture within its original validity window.
// Production pricing expiry remains unchanged and is checked separately below.
test.beforeEach(t => {
  assert.ok('mock' in t, 'The fixture clock requires an individual test context.');
  t.mock.timers.enable({apis:['Date'],now:new Date(evidence.reviewedAt)});
});
async function fixture() {
  const client=createClient({url:':memory:'}),database=adaptDatabase(client),env={EIDOS_DB:database};
  for(const f of ['0001_eidos_platform.sql','0002_members.sql','0004_paid_templates.sql'])await client.executeMultiple(readFileSync(new URL('../lib/works/vendor/migrations/'+f,import.meta.url),'utf8'));
  await ensureAiLedger(env);await configureAiPolicy(env,'test',providerTestRateCard('gpt-6.1-sol'),5*USD,false);
  await probePlaygroundCount(req(),{...source,EIDOS_PLAYGROUND_PROVIDER_DIAGNOSTICS:'test-count-one'},async()=>Response.json({object:'response.input_tokens',input_tokens:701},{headers:{'x-request-id':'req_fixture_count'}}),database);
  return {client,database,env,close:()=>client.close()};
}
const generated=()=>Response.json({id:'resp_fixture_generation',model:'gpt-6.1-sol',service_tier:'default',status:'completed',
  usage:{input_tokens:701,output_tokens:102,total_tokens:803,input_tokens_details:{cached_tokens:0,cache_write_tokens:0},output_tokens_details:{reasoning_tokens:20}},
  output:[{type:'message',role:'assistant',status:'completed',content:[{type:'output_text',text:JSON.stringify({ops:[{field:'section.title',value:'A clearer studio headline'}]})}]}]},
  {headers:{'x-request-id':'req_fixture_generation'}});

test('count bound is append-only and pessimistic; original receipt survives, customers remain disabled',async()=>{
  const f=await fixture();try{
    const before=await f.database.prepare('SELECT detail_json FROM eidos_pg_ai_journal WHERE id=?').bind(ownerCountClaimId).first();
    await assert.rejects(boundOwnerCountExpense(f.env,{...evidence,sourceUrl:'https://unsupported.test' as never}));
    assert.equal((await aiPolicyStatus(f.env,'test')).diagnosticHeld,5*USD);
    const result=await boundOwnerCountExpense(f.env,evidence);
    assert.equal(result.count.providerCostMicros,1753);assert.equal(result.count.actualCostMicros,null);
    assert.equal(result.policy.diagnosticHeld,0);assert.equal(result.policy.diagnosticPessimistic,1753);assert.equal(result.policy.diagnosticSettled,0);
    assert.equal(result.policy.committed,1753);assert.equal(result.policy.remaining,4998247);assert.equal(result.policy.enabled,false);assert.equal(result.policy.pausedReason,null);
    await boundOwnerCountExpense(f.env,evidence);
    await assert.rejects(boundOwnerCountExpense(f.env,{...evidence,documentSha256:'b'.repeat(64)}),/different/);
    assert.deepEqual(await f.database.prepare('SELECT detail_json FROM eidos_pg_ai_journal WHERE id=?').bind(ownerCountClaimId).first(),before);
    assert.equal((await f.database.prepare('SELECT COUNT(*) n FROM eidos_pg_ai_diagnostic_costs').first<{n:number}>())!.n,1);
  }finally{f.close();}
});

test('expired reviewed pricing refuses generation before transport or a generation hold',async t=>{
  const f=await fixture();let calls=0;
  try{
    await boundOwnerCountExpense(f.env,evidence);
    t.mock.timers.setTime(Date.parse(providerTestRateCard('gpt-6.1-sol').validUntil));
    const response=await probePlaygroundGeneration(req(),source,async()=>{calls++;return generated();},f.database);
    assert.equal(response.status,503);
    assert.equal((await response.json()).error,'reviewed_rate_card_expired');
    assert.equal(calls,0);
    assert.equal(await f.database.prepare('SELECT id FROM eidos_pg_ai_journal WHERE id=?').bind(ownerGenerationClaimId).first(),null);
  }finally{f.close();}
});

test('unreconciled count, exhausted budget, production and missing auth cause no generation',async()=>{
  const f=await fixture();let calls=0;const transport:typeof fetch=async()=>{calls++;return generated();};
  try{
    assert.equal((await probePlaygroundGeneration(req(),source,transport,f.database)).status,409);
    await boundOwnerCountExpense(f.env,evidence);await configureAiPolicy(f.env,'test',providerTestRateCard('gpt-6.1-sol'),121752,false);
    assert.equal((await probePlaygroundGeneration(req(),source,transport,f.database)).status,409);
    for(const bad of [{...source,VERCEL_ENV:'production'},{...source,EIDOS_PLATFORM_TOKEN:'wrong'},{...source,EIDOS_PLAYGROUND_PROVIDER_DIAGNOSTICS:'test-auth'}])
      assert.notEqual((await probePlaygroundGeneration(req(),bad,transport,f.database)).status,200);
    assert.equal(calls,0);assert.equal((await aiPolicyStatus(f.env,'test')).committed,1753);
  }finally{f.close();}
});

test('concurrent fixed public generation claims one maximum before I/O, settles usage, and never counts again',async()=>{
  const f=await fixture();let calls=0;
  try{
    await boundOwnerCountExpense(f.env,evidence);
    const transport:typeof fetch=async(url,init)=>{
      calls++;assert.equal(url,'https://api.openai.com/v1/responses');assert.equal(String(init!.body),ownerCountRequest().serialized);
      assert.equal((await aiPolicyStatus(f.env,'test')).committed,121753);
      return generated();
    };
    const results=await Promise.all(Array.from({length:4},()=>probePlaygroundGeneration(req(),source,transport,f.database)));
    assert.equal(results.filter(r=>r.status===200).length,1);assert.equal(calls,1);
    const body=await results.find(r=>r.status===200)!.json();assert.equal(body.countRequests,0);assert.equal(body.usable,true);assert.equal(body.providerCostMicros,2422);
    const status=await aiPolicyStatus(f.env,'test');assert.equal(status.diagnosticSettled,2422);assert.equal(status.diagnosticPessimistic,1753);assert.equal(status.committed,4175);
    assert.equal((await f.database.prepare('SELECT COUNT(*) n FROM eidos_pg_ai_actions').first<{n:number}>())!.n,0);
    assert.equal((await f.database.prepare('SELECT COUNT(*) n FROM eidos_email_members').first<{n:number}>())!.n,0);
    assert.equal((await probePlaygroundGeneration(req(),source,transport,f.database)).status,409);assert.equal(calls,1);
  }finally{f.close();}
});

test('timeout or missing usage retains upstream maximum, pauses and cannot blind-retry',async()=>{
  for(const failure of ['timeout','usage']){
    const f=await fixture();let calls=0;
    try{
      await boundOwnerCountExpense(f.env,evidence);
      const transport:typeof fetch=async()=>{calls++;if(failure==='timeout')throw Error(source.EIDOS_PLAYGROUND_OPENAI_API_KEY);return Response.json({id:'resp_unknown',usage:{}},{headers:{'x-request-id':'req_unknown'}});};
      const response=await probePlaygroundGeneration(req(),source,transport,f.database),body=await response.text();
      assert.equal(response.status,502);assert.equal(body.includes(source.EIDOS_PLAYGROUND_OPENAI_API_KEY),false);
      const status=await aiPolicyStatus(f.env,'test');assert.equal(status.diagnosticHeld,120000);assert.equal(status.committed,121753);assert.equal(status.pausedReason,'owner_generation_usage_requires_reconciliation');
      assert.equal((await probePlaygroundGeneration(req(),source,transport,f.database)).status,409);assert.equal(calls,1);
      assert.ok(await f.database.prepare('SELECT id FROM eidos_pg_ai_journal WHERE id=?').bind(ownerGenerationClaimId+'-result').first());
    }finally{f.close();}
  }
});

test('reported overrun records actual expense and pauses; provider errors expose only safe classifications',async()=>{
  for(const scenario of ['overrun','rejection']){
    const f=await fixture();try{
      await boundOwnerCountExpense(f.env,evidence);
      const transport:typeof fetch=async()=>{
        if(scenario==='rejection')return Response.json({error:{type:'invalid_request_error',code:'unknown_parameter',param:'fixture_param',message:source.EIDOS_PLAYGROUND_OPENAI_API_KEY}},{status:400,headers:{'x-request-id':'req_rejected'}});
        const body=await generated().json();body.usage.output_tokens=13000;body.usage.total_tokens=13701;return Response.json(body,{headers:{'x-request-id':'req_overrun'}});
      };
      const response=await probePlaygroundGeneration(req(),source,transport,f.database),body=await response.text(),status=await aiPolicyStatus(f.env,'test');
      assert.equal(response.status,502);assert.equal(body.includes(source.EIDOS_PLAYGROUND_OPENAI_API_KEY),false);
      if(scenario==='overrun'){assert.equal(status.diagnosticSettled,131402);assert.equal(status.pausedReason,'owner_generation_provider_bound_exceeded');}
      else{assert.equal(status.diagnosticHeld,120000);assert.equal(JSON.parse(body).providerError.code,'unknown_parameter');}
    }finally{f.close();}
  }
});
