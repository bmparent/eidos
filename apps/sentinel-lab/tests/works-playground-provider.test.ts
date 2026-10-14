import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createClient } from '@libsql/client';
import { adaptDatabase } from '../lib/works/database';
import { hash, type PlatformEnv } from '../lib/works/vendor/functions/_shared/platform/core';
import { aiBalance, aiPolicyStatus, configureAiPolicy, ensureAiLedger, ownedAiAction, publicAiAction, reconcileAiDeadlines, setAiLimit, USD,
  type AiBrief, type AiRateCard } from '../lib/works/vendor/functions/_shared/platform/playgroundAi';
import { aiProviderAvailability, buildAiProviderRequest, generateProviderAiAction, parseAiProviderUsage, providerTestRateCard, quoteProviderAiAction } from '../lib/works/vendor/functions/_shared/platform/playgroundAiProvider';
import { onRequestGet, onRequestPost } from '../lib/works/vendor/functions/api/playground/ai';
import { onRequestGet as publicConfig } from '../lib/works/vendor/functions/api/public-config';
import { createProposalContext } from '../lib/works/project-delivery/src/playground/aiOperations';
import { createProject } from '../lib/works/project-delivery/src/playground/model';

const at = new Date('2026-10-14T20:00:00.000Z');
const card: AiRateCard = { version: 'provider-controlled-20261014', model: 'gpt-6.1-sol', evidence: 'controlled-fixture',
  inputMicrosPerMillion: 2 * USD, cachedInputMicrosPerMillion: 100000, cacheWriteMicrosPerMillion: 2500000, outputMicrosPerMillion: 10 * USD,
  retailNumerator: 1, retailDenominator: 1, maxInputTokens: 32000, maxOutputTokens: 2000, maxCalls: 1, maxRetries: 0,
  maxAgentTurns: 1, maxSubagents: 0, paidTools: 0, maxToolReturnBytes: 0, processing: 'standard', reasoning: 'medium', maxRuntimeMs: 1000, validUntil: '2027-01-01T00:00:00.000Z' };
const project = createProject(), section = project.sections.find(s => s.id === 'hero')!;
const context = createProposalContext(project, section.id, '', 'section');
const brief: AiBrief = { documentId: 'provider-project', revision: 'revision-one', scope: 'section', targetId: section.id,
  instruction: 'Clarify the headline', context: JSON.stringify(context) };
const proposal = { ops: [{ field: 'section.title', value: 'A clearer headline for your studio' }] };
const metered = { input_tokens: 1000, input_tokens_details: { cached_tokens: 100, cache_write_tokens: 200 },
  output_tokens: 100, output_tokens_details: { reasoning_tokens: 80 }, total_tokens: 1100 };
const responseBody = (change: Record<string, unknown> = {}) => ({ id: 'resp_controlled', model: card.model, service_tier: 'default', status: 'completed', usage: metered,
  output: [{ type: 'reasoning', summary: [] }, { type: 'message', role: 'assistant', status: 'completed', content: [{ type: 'output_text', text: JSON.stringify(proposal) }] }], ...change });
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'content-type': 'application/json', 'x-request-id': 'req_controlled' } });
async function fixture(budget = 5 * USD) {
  const client = createClient({ url: ':memory:' }), database = adaptDatabase(client);
  for (const name of ['0001_eidos_platform.sql','0002_members.sql','0004_paid_templates.sql']) await client.executeMultiple(readFileSync(new URL('../lib/works/vendor/migrations/' + name, import.meta.url), 'utf8'));
  const env: PlatformEnv = { EIDOS_DB: database, EIDOS_AI_ENABLED: 'true', EIDOS_PLAYGROUND_AI_MODE: 'test', EIDOS_PLAYGROUND_AI_ENABLED: 'true', EIDOS_PLAYGROUND_OPENAI_API_KEY: 'controlled-not-a-real-provider-secret' };
  await ensureAiLedger(env);
  await database.prepare('INSERT INTO eidos_email_members(id,email,username,kind,created_at,newsletter_after) VALUES(?,?,?,?,?,?)').bind('owner', 'controlled@example.test', 'controlled', 'person', at.toISOString(), at.toISOString()).run();
  await setAiLimit(env, 'owner', 'test', 5 * USD, at);
  // Synthetic accounting seed; never a payment/provider acceptance receipt.
  await database.prepare('INSERT INTO eidos_orders(id,receipt_hash,session_id,status,created_at,payment_intent) VALUES(?,?,?,?,?,?)').bind('controlled-order','controlled-hash','cs_test_controlled','paid',at.toISOString(),'pi_test_controlled').run();
  await database.prepare('INSERT INTO eidos_pg_ai_grants(id,owner_id,mode,order_id,grant_version,kind,amount,payment_intent,created_at) VALUES(?,?,?,?,?,?,?,?,?)')
    .bind('controlled-grant','owner','test','controlled-order','fixture-v1','purchased',5 * USD,'pi_test_controlled',at.toISOString()).run();
  await configureAiPolicy(env, 'test', card, budget, true);
  let counts = 0, generations = 0, generationBody = '', countBody = '', serial = 0;
  let generated: (init: RequestInit) => Response | Promise<Response> = () => json(responseBody());
  let counted: () => Response = () => json({ object: 'response.input_tokens', input_tokens: 1500 });
  const transport: typeof fetch = async (url, init) => {
    assert.ok(init); assert.equal(init.redirect, 'error');
    if (String(url).endsWith('/input_tokens')) { counts++; countBody = String(init.body); return counted(); }
    assert.equal(String(url), 'https://api.openai.com/v1/responses'); generations++; generationBody = String(init.body);
    const requestId = new Headers(init.headers).get('x-client-request-id'); assert.ok(requestId);
    const action = await ownedAiAction(env, 'owner', 'test', requestId);
    assert.equal(action.state, 'submitted'); assert.equal(action.customer_status, 'held'); assert.equal(action.provider_status, 'held');
    return generated(init);
  };
  const options = { transport, at };
  return { client, database, env, options, quote: (change: Partial<AiBrief> = {}, key = 'controlled-action-' + ++serial) => quoteProviderAiAction(env,'owner','test',key,{...brief,...change},options),
    generate: (id: string, maximum: number) => generateProviderAiAction(env,'owner','test',id,maximum,options),
    generated: (fn: typeof generated) => { generated = fn; }, counted: (fn: typeof counted) => { counted = fn; },
    counts: () => counts, generations: () => generations, bodies: () => ({countBody,generationBody}), close: () => client.close() };
}

test('provider request has explicit scope, typed schema, fixed standard tier, total output cap, no cache writes/tools/subagents', () => {
  const built = buildAiProviderRequest(brief, card), request = built.request;
  assert.equal(request.service_tier,'default'); assert.equal(request.max_output_tokens,card.maxOutputTokens); assert.equal(request.reasoning.effort,'medium');
  assert.deepEqual(request.tools,[]); assert.equal(request.tool_choice,'none'); assert.deepEqual(request.prompt_cache_options,{mode:'explicit'});
  assert.equal(request.store,false); assert.equal(request.stream,false); assert.equal(request.background,false);
  assert.equal(request.text.format.strict,true); assert.ok(!('multi_agent' in request)); assert.ok(!('previous_response_id' in request));
  assert.throws(() => buildAiProviderRequest({...brief,targetId:'other'},card),/scope and target/);
  assert.throws(() => buildAiProviderRequest({...brief,context:'{}'},card),/valid bounded/);
  assert.throws(() => buildAiProviderRequest({...brief,...{owner:'someone'}} as AiBrief,card),/server-owned/);
  assert.equal(parseAiProviderUsage(metered).reasoningTokens,80);
});
test('exact full input count and payload hash are durable; idempotent quote does not recount', async () => {
  const f = await fixture(); try {
    const action = await f.quote({},'same-controlled-action'); assert.equal(f.counts(),1); assert.equal(f.generations(),0);
    const stored = await f.database.prepare('SELECT * FROM eidos_pg_ai_preflights WHERE action_id=?').bind(action.id).first<{input_tokens:number;request_hash:string;request_json:string}>();
    assert.equal(stored?.input_tokens,1500); assert.match(stored!.request_hash,/^[a-f0-9]{64}$/); assert.equal(stored?.request_json,f.bodies().countBody);
    assert.equal((await f.quote({},'same-controlled-action')).id,action.id); assert.equal(f.counts(),1);
    await assert.rejects(f.quote({instruction:'Different instruction'},'same-controlled-action'),/different scope|another request/); assert.equal(f.counts(),1);
  } finally {f.close();}
});
test('unsupported, oversized and failed full-input counting never submit generation or create a quote', async () => {
  for (const body of [{object:'response.input_tokens',input_tokens:32001},{object:'wrong',input_tokens:10},{object:'response.input_tokens',input_tokens:null}]) {
    const f = await fixture(); try { f.counted(()=>json(body)); await assert.rejects(f.quote(),/exceeds|could not be counted/); assert.equal(f.generations(),0);
      assert.equal((await f.database.prepare('SELECT COUNT(*) count FROM eidos_pg_ai_actions').first<{count:number}>())?.count,0);
    } finally {f.close();}
  }
});
test('dedicated key, explicit enablement, verified preflight pricing and $5 aggregate budget all fail closed', async () => {
  const f = await fixture(); try {
    const original = {...f.env};
    for (const change of [{EIDOS_PLAYGROUND_OPENAI_API_KEY:undefined,OPENAI_API_KEY:'old-unrelated-key-must-never-be-used'}, {EIDOS_PLAYGROUND_AI_ENABLED:'false'}, {EIDOS_AI_ENABLED:'false'}, {EIDOS_AI_ENABLED:undefined}]) {
      Object.assign(f.env,original,change); await assert.rejects(f.quote(),/key|paused/); assert.equal(f.counts(),0);
    }
    Object.assign(f.env,original);
    await configureAiPolicy(f.env,'test',{...card,evidence:'owner-approved'},5 * USD,true);
    await assert.rejects(f.quote(),/billing must be verified/); assert.equal((await aiProviderAvailability(f.env)).enabled,false);
    await configureAiPolicy(f.env,'test',card,5 * USD + 1,true); await assert.rejects(f.quote(),/bounded to \$5/); assert.equal(f.counts(),0);
    await assert.rejects(quoteProviderAiAction(f.env,'owner','test','network-forbidden',brief),/fixtures cannot call/);
  } finally {f.close();}
});
test('public Playground readiness uses only dedicated bindings and global kill switch, without ledger queries or provider calls',async()=>{
  const originalFetch=globalThis.fetch;let queries=0,calls=0;
  const database={prepare(){queries++;throw Error('public config must not read the ledger');},async batch(){queries++;throw Error('public config must not write the ledger');}};
  const configured:PlatformEnv={EIDOS_RUNTIME:'sentinel',EIDOS_DB:database,EIDOS_AI_ENABLED:'true',EIDOS_PLAYGROUND_AI_ENABLED:'true',EIDOS_PLAYGROUND_AI_MODE:'test',
    EIDOS_PLAYGROUND_OPENAI_API_KEY:'controlled-dedicated-key-never-returned',EIDOS_PLAYGROUND_INPUT_COUNT_PRICING:'verified-zero',EIDOS_RATE_SECRET:'r'.repeat(32)};
  const read=async(env:PlatformEnv)=>{const response=await publicConfig({env,request:new Request('https://preview.example.test/api/public-config')});assert.equal(response.status,200);return response.json() as Promise<{aiReady:boolean;playgroundAiReady:boolean;accountsReady:boolean}>;};
  try{globalThis.fetch=async()=>{calls++;throw Error('public config must not call the provider');};
    const ready=await read(configured);assert.equal(ready.playgroundAiReady,true);assert.equal(ready.aiReady,false);assert.equal(ready.accountsReady,false);
    for(const change of [{EIDOS_RUNTIME:undefined},{EIDOS_AI_ENABLED:undefined},{EIDOS_AI_ENABLED:'false'},{EIDOS_PLAYGROUND_AI_ENABLED:'false'},
      {EIDOS_PLAYGROUND_AI_MODE:'live'},{EIDOS_PLAYGROUND_OPENAI_API_KEY:undefined},{EIDOS_PLAYGROUND_INPUT_COUNT_PRICING:undefined},{EIDOS_PLAYGROUND_INPUT_COUNT_PRICING:'unknown'},
      {EIDOS_DB:undefined},{EIDOS_RATE_SECRET:'short'}])assert.equal((await read({...configured,...change})).playgroundAiReady,false);
    const legacyOnly=await read({...configured,EIDOS_PLAYGROUND_OPENAI_API_KEY:undefined,OPENAI_API_KEY:'controlled-legacy-key',EIDOS_ASSISTANT_MODEL:'legacy-model'});
    assert.equal(legacyOnly.aiReady,true);assert.equal(legacyOnly.playgroundAiReady,false);
    assert.equal(queries,0);assert.equal(calls,0);assert.ok(!JSON.stringify(ready).includes('controlled-dedicated-key'));
  }finally{globalThis.fetch=originalFetch;}
});
test('global platform kill switch blocks previously quoted generation before either reservation or provider dispatch',async()=>{
  const f=await fixture();try{const a=await f.quote();f.env.EIDOS_AI_ENABLED='false';
    await assert.rejects(f.generate(a.id,a.maximum),/generation is paused/);assert.equal(f.counts(),1);assert.equal(f.generations(),0);
    const action=await ownedAiAction(f.env,'owner','test',a.id);assert.equal(action.state,'quoted');assert.equal(action.provider_status,'none');assert.equal(action.customer_status,'none');
    assert.equal((await aiPolicyStatus(f.env,'test')).committed,0);
  }finally{f.close();}
});
test('exact maximum approval and atomic duplicate admission yield one generation, using the identical counted payload', async () => {
  const f = await fixture(); try {
    const a = await f.quote(); await assert.rejects(f.generate(a.id,a.maximum-1),/exact maximum/); assert.equal(f.generations(),0);
    const settled = await Promise.all(Array.from({length:8},()=>f.generate(a.id,a.maximum))); assert.equal(f.generations(),1);
    assert.ok(settled.some(a=>a.state==='settled')); assert.equal(f.bodies().generationBody,f.bodies().countBody);
    const final = await ownedAiAction(f.env,'owner','test',a.id); assert.equal(final.provider_cost,2910); assert.equal(final.charged_purchased,10000);
    assert.deepEqual(publicAiAction(final).result,proposal); assert.equal(publicAiAction(final).targetId,'hero');
    assert.equal((await f.generate(a.id,a.maximum)).state,'settled'); assert.equal(f.generations(),1);
    assert.ok(!JSON.stringify(publicAiAction(final)).includes('controlled-not-a-real-provider-secret'));
  } finally {f.close();}
});
test('malformed response retains header identity and provider hold without inventing response ID; replay never retries', async () => {
  const f = await fixture(); try {
    const a = await f.quote(); f.generated(()=>new Response('{',{headers:{'content-type':'application/json','x-request-id':'req_malformed'}}));
    const action = await f.generate(a.id,a.maximum); assert.equal(action.state,'reconciliation_pending'); assert.equal(action.provider_request_id,'req_malformed'); assert.equal(action.provider_id,null);
    assert.equal((await aiPolicyStatus(f.env,'test')).committed,a.provider_maximum); await f.generate(a.id,a.maximum); assert.equal(f.generations(),1);
    await reconcileAiDeadlines(f.env,new Date(at.getTime()+25*60*60*1000)); const late = await ownedAiAction(f.env,'owner','test',a.id);
    assert.equal(late.customer_status,'waived'); assert.equal(late.provider_status,'pessimistic'); assert.equal(late.provider_cost,a.provider_maximum);
  } finally {f.close();}
});
test('invalid typed, unsafe, incomplete and no-change proposals waive the customer while counting actual provider expense', async () => {
  const bodies = [responseBody({status:'incomplete'}), ...[{ops:[]},{ops:[{field:'section.href',value:'javascript:alert(1)'}]},
    {ops:[{field:'node.text',value:'wrong scope'}]},{ops:[{field:'section.title',value:section.title}]}].map(p=>responseBody({output:[{type:'message',role:'assistant',status:'completed',content:[{type:'output_text',text:JSON.stringify(p)}]}]}))];
  for (const body of bodies) {const f=await fixture();try {const a=await f.quote();f.generated(()=>json(body));const result=await f.generate(a.id,a.maximum);
    assert.equal(result.customer_status,'waived'); assert.equal(result.provider_status,'settled'); assert.equal(result.provider_cost,2910); assert.equal(result.result_json,null);
    assert.equal((await aiBalance(f.env,'owner','test',at)).availablePurchased,5*USD);
  }finally{f.close();}}
});
test('missing usage categories and changed service/model pause the route and preserve unresolved liability', async () => {
  for(const change of [{usage:{...metered,input_tokens_details:{cached_tokens:0}}},{service_tier:'priority'},{model:'gpt-6-luna'}]) {
    const f=await fixture();try{const a=await f.quote();f.generated(()=>json(responseBody(change)));await assert.rejects(f.generate(a.id,a.maximum),/could not be bounded/);
      const final=await ownedAiAction(f.env,'owner','test',a.id); assert.equal(final.provider_id,'resp_controlled'); assert.equal(final.provider_request_id,'req_controlled');
      const status=await aiPolicyStatus(f.env,'test');assert.equal(status.dispatchBlocked,true);assert.equal(status.committed,a.provider_maximum);assert.equal(final.charged_purchased,0);
    }finally{f.close();}
  }
});
test('unexpected tools and subagents retain pessimistic provider liability, waive customer and stop new dispatch', async()=>{
  for(const type of ['web_search_call','multi_agent_call']) {const f=await fixture();try{const a=await f.quote();f.generated(()=>json(responseBody({output:[{type}]})));
    const final=await f.generate(a.id,a.maximum);assert.equal(final.provider_status,'pessimistic');assert.equal(final.provider_cost,a.provider_maximum);assert.equal(final.customer_status,'waived');
    assert.equal((await aiPolicyStatus(f.env,'test')).dispatchBlocked,true);
  }finally{f.close();}}
});
test('tampered or absent frozen full-input proof fails before reservation and dispatch',async()=>{
  for(const edit of ["UPDATE eidos_pg_ai_preflights SET request_json='{}'","DELETE FROM eidos_pg_ai_preflights"]) {const f=await fixture();try{const a=await f.quote();await f.database.prepare(edit).run();
    await assert.rejects(f.generate(a.id,a.maximum),/fresh full-input/);assert.equal(f.generations(),0);assert.equal((await ownedAiAction(f.env,'owner','test',a.id)).state,'quoted');
  }finally{f.close();}}
});
test('network timeout is unknown provider liability with one dispatch and no automatic retry',async()=>{
  const f=await fixture();try{const a=await f.quote();f.generated(init=>new Promise((_resolve,reject)=>init.signal!.addEventListener('abort',()=>reject(Error('aborted')),{once:true})));
    const final=await f.generate(a.id,a.maximum);assert.equal(final.state,'reconciliation_pending');assert.equal(final.provider_id,null);assert.equal(final.provider_request_id,null);
    await f.generate(a.id,a.maximum);assert.equal(f.generations(),1);assert.equal((await aiPolicyStatus(f.env,'test')).committed,a.provider_maximum);
  }finally{f.close();}
});
test('different models share the same lifetime budget and unresolved commitments survive operator reconfiguration',async()=>{
  const f=await fixture(150000);try{const a=await f.quote();f.generated(()=>{throw Error('connection ended');});
    const first=await f.generate(a.id,a.maximum);assert.equal(first.provider_status,'held');assert.equal(a.provider_maximum,100000);
    await configureAiPolicy(f.env,'test',{...card,model:'gpt-6-astra',version:'second-model-controlled'},150000,true);
    const second=await f.quote();await assert.rejects(f.generate(second.id,second.maximum),/owner budget|allowance/);assert.equal(f.generations(),1);
    const status=await aiPolicyStatus(f.env,'test');assert.equal(status.committed,100000);assert.equal(status.remaining,50000);
  }finally{f.close();}
});
test('reported provider overrun remains visible, caps customer charge and pauses the route',async()=>{
  const f=await fixture();try{const a=await f.quote();f.generated(()=>json(responseBody({usage:{...metered,output_tokens:20000,total_tokens:21000}})));
    const final=await f.generate(a.id,a.maximum);assert.equal(final.provider_cost,201910);assert.equal(final.charged_purchased,a.maximum);assert.equal(final.provider_status,'settled');
    const status=await aiPolicyStatus(f.env,'test');assert.equal(status.dispatchBlocked,true);assert.equal(status.committed,201910);
  }finally{f.close();}
});
test('authenticated API derives billing owner, rejects forged routing/caps, and returns direct proposal plus refreshed balance',async()=>{
  const f=await fixture(), originalFetch=globalThis.fetch;
  try{
    // Isolated synthetic transport. The reviewed public-rate card is used only
    // to exercise the HTTP production code path; this sends no network request.
    Object.assign(f.env,{EIDOS_LOCAL_TEST:'true',EIDOS_PLAYGROUND_INPUT_COUNT_PRICING:'verified-zero'});
    await configureAiPolicy(f.env,'test',providerTestRateCard('gpt-6.1-sol'),5*USD,true);
    const session='c'.repeat(64);await f.database.prepare('INSERT INTO eidos_member_sessions(token_hash,member_id,expires) VALUES(?,?,?)').bind(await hash(session),'owner',2000000000).run();
    const headers={'content-type':'application/json',origin:'http://localhost',cookie:'eidos_session_local='+session};
    let calls=0;globalThis.fetch=async(url)=>{calls++;return String(url).endsWith('/input_tokens')?json({object:'response.input_tokens',input_tokens:1500}):json(responseBody());};
    const post=(input:unknown)=>onRequestPost({env:f.env,request:new Request('http://localhost/api/playground/ai',{method:'POST',headers,body:JSON.stringify(input)})});
    assert.equal((await onRequestGet({env:f.env,request:new Request('http://localhost/api/playground/ai')})).status,401);
    for(const forged of [{action:'quote',idempotencyKey:'forged-owner',brief,owner:'other'},{action:'quote',idempotencyKey:'forged-model',brief:{...brief,model:'gpt-6-luna'}}])assert.equal((await post(forged)).status,400);
    assert.equal(calls,0);
    const quoted=await post({action:'quote',idempotencyKey:'http-controlled-quote',brief});assert.equal(quoted.status,200);
    const {action}=await quoted.json() as {action:{id:string;maximum:number;targetId:string}};assert.equal(action.targetId,'hero');assert.equal(calls,1);
    assert.equal((await post({action:'generate',id:action.id,approvedMaximumMicros:action.maximum,owner:'other'})).status,400);assert.equal(calls,1);
    assert.equal((await post({action:'generate',id:action.id,approvedMaximumMicros:action.maximum-1})).status,409);assert.equal(calls,1);
    const generated=await post({action:'generate',id:action.id,approvedMaximumMicros:action.maximum});assert.equal(generated.status,200);
    const received=await generated.json() as {action:{result:unknown;state:string};balance:{settled:number}};
    assert.equal(received.action.state,'settled');assert.deepEqual(received.action.result,proposal);assert.equal(received.balance.settled,10000);assert.equal(calls,2);
    const receipt=await onRequestGet({env:f.env,request:new Request('http://localhost/api/playground/ai?id='+action.id,{headers})});assert.equal(receipt.status,200);assert.equal(calls,2);
  }finally{globalThis.fetch=originalFetch;f.close();}
});
