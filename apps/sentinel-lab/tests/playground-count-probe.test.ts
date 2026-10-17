import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createClient } from '@libsql/client';
import { adaptDatabase } from '../lib/works/database';
import { probePlaygroundCount, ownerCountClaimId } from '../lib/works/playground-count-probe';
import { activeAiRateCard, aiPolicyStatus, cancelAiAction, claimAiInputCount, configureAiPolicy, ensureAiLedger, quoteAiAction, setAiLimit, USD } from '../lib/works/vendor/functions/_shared/platform/playgroundAi';
import { buildAiProviderRequest, providerTestRateCard } from '../lib/works/vendor/functions/_shared/platform/playgroundAiProvider';
import { hash } from '../lib/works/vendor/functions/_shared/platform/core';
import { createProject } from '../lib/works/project-delivery/src/playground/model';
import { createProposalContext } from '../lib/works/project-delivery/src/playground/aiOperations';

const source = { VERCEL_ENV:'preview', EIDOS_PLAYGROUND_PROVIDER_DIAGNOSTICS:'test-count-one', EIDOS_PLATFORM_TOKEN:'owner-fixture-token-at-least-32-characters', EIDOS_PLAYGROUND_OPENAI_API_KEY:'private-provider-fixture-never-exposed' };
const request=()=>new Request('https://preview.test/api/playground-provider-check',{method:'POST',headers:{'x-eidos-platform-token':source.EIDOS_PLATFORM_TOKEN,'x-eidos-site-origin':'https://eidosworks-playground-20261014.pages.dev'}});
async function fixture() {
  const client=createClient({url:':memory:'}),database=adaptDatabase(client),env={EIDOS_DB:database};
  for(const file of ['0001_eidos_platform.sql','0002_members.sql','0004_paid_templates.sql']) await client.executeMultiple(readFileSync(new URL('../lib/works/vendor/migrations/'+file,import.meta.url),'utf8'));
  await ensureAiLedger(env); await configureAiPolicy(env,'test',providerTestRateCard('gpt-6.1-sol'),5*USD,false);
  return {database,env,close:()=>client.close()};
}

test('count probe cannot run without its explicit Preview-only mode and owner authentication',async()=>{
  let calls=0;const transport:typeof fetch=async()=>{calls++;throw Error('must not call');};
  for(const env of [{...source,VERCEL_ENV:'production'},{...source,EIDOS_PLAYGROUND_PROVIDER_DIAGNOSTICS:'test-auth'},{...source,EIDOS_PLATFORM_TOKEN:'wrong'},{...source,EIDOS_PLAYGROUND_OPENAI_API_KEY:''}])
    assert.notEqual((await probePlaygroundCount(request(),env,transport)).status,200);
  assert.equal(calls,0);
});

test('concurrent owner probes claim all $5 durably before one count, never infer or retry, and retain the unknown hold',async()=>{
  const f=await fixture();let calls=0;
  try {
    const transport:typeof fetch=async(url,init)=>{
      calls++;assert.equal(url,'https://api.openai.com/v1/responses/input_tokens');
      const status=await aiPolicyStatus(f.env,'test');assert.equal(status.committed,5*USD);assert.equal(status.remaining,0);assert.equal(status.dispatchBlocked,true);
      const body=JSON.parse(String(init?.body));assert.equal(body.model,'gpt-6.1-sol');assert.equal('max_output_tokens' in body,false);assert.ok(body.text.format.schema);
      return Response.json({object:'response.input_tokens',input_tokens:1234},{headers:{'x-request-id':'req_count_test'}});
    };
    const results=await Promise.all(Array.from({length:4},()=>probePlaygroundCount(request(),source,transport,f.database)));
    assert.equal(results.filter(r=>r.status===200).length,1);assert.equal(calls,1);
    const body=await results.find(r=>r.status===200)!.json();assert.equal(body.inputTokens,1234);assert.equal(body.billing,'unknown');assert.equal(body.generationRequests,0);
    await configureAiPolicy(f.env,'test',providerTestRateCard('gpt-6.1-sol'),5*USD,true);
    await assert.rejects(activeAiRateCard(f.env,'test'));
    assert.equal((await aiPolicyStatus(f.env,'test')).diagnosticHeld,5*USD);
    assert.equal((await probePlaygroundCount(request(),source,transport,f.database)).status,409);assert.equal(calls,1);
  } finally {f.close();}
});

test('failed count retains all $5 and hides provider body; active policy rejects probe before I/O',async()=>{
  for(const active of [false,true]) {
    const f=await fixture();let calls=0;
    try {
      if(active)await configureAiPolicy(f.env,'test',providerTestRateCard('gpt-6.1-sol'),5*USD,true);
      const transport:typeof fetch=async()=>{calls++;return new Response(source.EIDOS_PLAYGROUND_OPENAI_API_KEY,{status:401});};
      const response=await probePlaygroundCount(request(),source,transport,f.database),body=await response.text();
      assert.equal(body.includes(source.EIDOS_PLAYGROUND_OPENAI_API_KEY),false);assert.equal(calls,active?0:1);
      assert.equal((await aiPolicyStatus(f.env,'test')).diagnosticHeld,active?0:5*USD);
    } finally {f.close();}
  }
});

async function assertPriorLiabilityBlocksProbe(f: Awaited<ReturnType<typeof fixture>>) {
  const before=await aiPolicyStatus(f.env,'test');
  assert.equal(before.committed,70000);assert.equal(before.enabled,false);assert.equal(before.pausedReason,null);
  let calls=0;
  const response=await probePlaygroundCount(request(),source,async()=>{
    calls++;return Response.json({object:'response.input_tokens',input_tokens:701},{headers:{'x-request-id':'req_fixture'}});
  },f.database);
  assert.equal(response.status,409);assert.equal(calls,0);
  assert.equal(await f.database.prepare('SELECT id FROM eidos_pg_ai_journal WHERE id=?').bind(ownerCountClaimId).first(),null);
  assert.deepEqual(await aiPolicyStatus(f.env,'test'),before);
  assert.equal(before.overBudget,false);
}

test('the full-budget owner count rejects retained held, bounded and settled diagnostic liabilities before I/O',async(t)=>{
  for(const event of ['diagnostic_count_hold','diagnostic_generation_hold']) {
    for(const state of ['held','bounded','settled']) await t.test(event+' '+state,async()=>{
      const f=await fixture(),createdAt=new Date().toISOString(),claimId='prior-diagnostic-fixture';
      try {
        await f.database.prepare('INSERT INTO eidos_pg_ai_journal(id,mode,event,detail_json,created_at) VALUES(?,?,?,?,?)')
          .bind(claimId,'test',event,JSON.stringify({providerHeldMicros:state==='held'?70000:200000}),createdAt).run();
        if(state!=='held') await f.database.prepare("INSERT INTO eidos_pg_ai_diagnostic_costs(claim_id,mode,status,provider_cost,evidence_json,created_at) VALUES(?,'test','bounded',?,'{}',?)")
          .bind(claimId,state==='bounded'?70000:100000,createdAt).run();
        if(state==='settled') await f.database.prepare("INSERT INTO eidos_pg_ai_diagnostic_costs(claim_id,mode,status,provider_cost,evidence_json,created_at) VALUES(?,'test','settled',70000,'{}',?)")
          .bind(claimId,createdAt).run();
        await assertPriorLiabilityBlocksProbe(f);
      } finally {f.close();}
    });
  }
});

test('cancelled customer count liability survives disabled policy and blocks the first owner probe',async()=>{
  const f=await fixture(),at=new Date('2026-10-17T21:00:00.000Z');
  try {
    await f.database.prepare('INSERT INTO eidos_email_members(id,email,username,kind,created_at,newsletter_after) VALUES(?,?,?,?,?,?)')
      .bind('owner','controlled@example.test','controlled','person',at.toISOString(),at.toISOString()).run();
    await setAiLimit(f.env,'owner','test',5*USD,at);
    // Synthetic paid allowance only. No provider, payment, or persistent database is used.
    await f.database.prepare('INSERT INTO eidos_orders(id,receipt_hash,session_id,status,created_at,payment_intent) VALUES(?,?,?,?,?,?)')
      .bind('controlled-order','controlled-hash','cs_test_controlled','paid',at.toISOString(),'pi_test_controlled').run();
    await f.database.prepare('INSERT INTO eidos_pg_ai_grants(id,owner_id,mode,order_id,grant_version,kind,amount,payment_intent,created_at) VALUES(?,?,?,?,?,?,?,?,?)')
      .bind('controlled-grant','owner','test','controlled-order','fixture-v1','purchased',5*USD,'pi_test_controlled',at.toISOString()).run();
    const card={...providerTestRateCard('gpt-6.1-sol'),evidence:'controlled-fixture' as const};
    await configureAiPolicy(f.env,'test',card,5*USD,true);
    const project=createProject(),section=project.sections.find(s=>s.id==='hero')!;
    const brief={documentId:'controlled-project',revision:'one',scope:'section' as const,targetId:section.id,
      instruction:'Clarify this public headline.',context:JSON.stringify(createProposalContext(project,section.id,'','section'))};
    const action=await quoteAiAction(f.env,'owner','test','controlled-customer-count',brief,at);
    const serialized=buildAiProviderRequest(brief,card).countSerialized;
    await claimAiInputCount(f.env,'owner','test',action.id,serialized,at,
      {requestHash:await hash(serialized),maximumMicros:70000,evidenceId:'controlled-full-request-bound-v1'});
    const cancelled=await cancelAiAction(f.env,'owner','test',action.id);
    assert.equal(cancelled.state,'cancelled');assert.equal(cancelled.provider_status,'none');
    await configureAiPolicy(f.env,'test',providerTestRateCard('gpt-6.1-sol'),5*USD,false);
    await assertPriorLiabilityBlocksProbe(f);
  } finally {f.close();}
});
