import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createClient } from '@libsql/client';
import { adaptDatabase } from '../lib/works/database';
import { probePlaygroundCount } from '../lib/works/playground-count-probe';
import { activeAiRateCard, aiPolicyStatus, configureAiPolicy, ensureAiLedger, USD } from '../lib/works/vendor/functions/_shared/platform/playgroundAi';
import { providerTestRateCard } from '../lib/works/vendor/functions/_shared/platform/playgroundAiProvider';

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
