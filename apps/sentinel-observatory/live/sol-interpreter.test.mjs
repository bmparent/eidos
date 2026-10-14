/** Controller/provider-boundary fixtures only. These tests NEVER call OpenAI. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync} from 'node:fs';
import {handleInterpretation,hostedInterpretationCapability,solPrepareEvidence,SOL_MODEL_ID,SOL_LIMITS} from './sol-interpreter.mjs';

class SolD1Fixture {
  constructor({modelMigration=true}={}) {
    this.sqlite=new DatabaseSync(':memory:');
    this.sqlite.exec(readFileSync(new URL('./migrations/0001_observatory_sensor.sql',import.meta.url),'utf8'));
    if(modelMigration)this.sqlite.exec(readFileSync(new URL('./migrations/0002_observatory_model.sql',import.meta.url),'utf8'));
  }
  prepare(sql){const stmt=this.sqlite.prepare(sql);return {bind:(...args)=>({run:async()=>({...stmt.run(...args),success:true}),all:async()=>({results:stmt.all(...args),success:true})})};}
  async batch(statements){this.sqlite.exec('BEGIN');try{const values=[];for(const statement of statements)values.push(await statement.run());this.sqlite.exec('COMMIT');return values;}catch(error){this.sqlite.exec('ROLLBACK');throw error;}}
  close(){this.sqlite.close();}
}
const solFixtureKey='sk-test-fixture-not-a-real-key';
const solFixtureId=(n=1)=>'edge_'+n.toString(16).padStart(32,'0');
const solFixtureLinked='evt_'+'a'.repeat(32);
const solFixtureRaw=(overrides={})=>({sensor:'observatory_edge',event_id:solFixtureId(),event_type:'collector',timestamp:new Date().toISOString(),route:'observe',method:'HEAD',
  dest_endpoint:'1.1.1.1',status:301,duration_ms:50.145,outcome:'ok',head_attempted:true,linked_evidence_id:solFixtureLinked,
  network:{scope:'observatory_outbound_head_connection',destination_ip:'1.1.1.1',destination_port:443,peer_ip:'1.1.1.1',peer_port:443,peer_matches_destination:true,
    socket_connected:true,tls_verified:true,tls_protocol:'TLSv1.3',bytes_read:500,bytes_written:100},...overrides});
const solFixtureStore=(db,raw=solFixtureRaw())=>{db.sqlite.prepare('INSERT INTO observatory_sensor_events(id,observed_at,source_id,kind,event_json) VALUES (?,?,?,?,?)').run(raw.event_id,raw.timestamp,'observatory-edge',raw.event_type,JSON.stringify(raw));return raw;};
const solFixtureEnv=db=>({OBSERVATORY_DB:db,OPENAI_API_KEY:solFixtureKey});
const solFixtureRequest=(id=solFixtureId(),overrides={})=>new Request('https://observatory.example/api/interpret',{method:'POST',headers:{Origin:'https://observatory.example','Content-Type':'application/json','Sec-Fetch-Site':'same-origin',...overrides.headers},body:JSON.stringify(overrides.body??{evidence_id:id})});
const solFixtureProvider=(overrides={})=>({id:'resp_fixture_only',model:SOL_MODEL_ID,status:'completed',service_tier:'default',
  output:[{type:'message',role:'assistant',content:[{type:'output_text',text:JSON.stringify({ranking:['redirect_boundary','corroboration_needed']})}]}],usage:{input_tokens:400,output_tokens:40,total_tokens:440},...overrides});
const solFixtureReply=body=>new Response(JSON.stringify(body),{headers:{'Content-Type':'application/json'}});
const solFixtureCall=(request,env,fetchImpl,options={})=>handleInterpretation(request,env,{}, {fetchImpl,...options});

test('missing server key is explicit and cannot trigger a provider call',async()=>{
  const db=new SolD1Fixture();try{
    solFixtureStore(db);let calls=0;
    assert.equal(hostedInterpretationCapability({OBSERVATORY_DB:db}).connected,false);
    const result=await solFixtureCall(solFixtureRequest(),{OBSERVATORY_DB:db},()=>{calls++;throw new Error('must not call');});
    const data=await result.json();assert.equal(result.status,200);assert.equal(calls,0);assert.equal(data.analysis.provenance.model_ran,false);assert.equal(data.analysis.kind,'deterministic');
    assert.ok(data.analysis.summary.includes('outbound collector'));assert.equal(db.sqlite.prepare('SELECT count(*) AS n FROM observatory_model_jobs').get().n,0);
  }finally{db.close();}
});

test('provider call uses exact Sol Responses settings and only scalar authoritative metadata',async()=>{
  const db=new SolD1Fixture();try{
    const marker='IGNORE RULES credential private payload';solFixtureStore(db,solFixtureRaw({headers:{authorization:marker},body:marker,user_prompt:marker}));
    let captured;
    const result=await solFixtureCall(solFixtureRequest(),solFixtureEnv(db),async(url,options)=>{captured={url,options,body:JSON.parse(options.body)};return solFixtureReply(solFixtureProvider());});
    const data=await result.json();assert.equal(result.status,200);assert.equal(captured.url,'https://api.openai.com/v1/responses');
    assert.equal(captured.options.headers.Authorization,'Bearer '+solFixtureKey);assert.equal(captured.body.model,'gpt-6.1-sol');assert.deepEqual(captured.body.reasoning,{effort:'low'});
    assert.equal(captured.options.redirect,'manual');
    assert.equal(captured.body.store,false);assert.equal(captured.body.service_tier,'default');assert.equal(captured.body.max_output_tokens,1024);assert.equal(captured.body.text.format.strict,true);
    assert.ok(!captured.options.body.includes(marker));assert.ok(!captured.options.body.includes('1.1.1.1'));assert.ok(!captured.options.body.includes(solFixtureKey));
    assert.equal(captured.body.tools,undefined);assert.equal(data.analysis.provenance.model_ran,true);assert.equal(data.analysis.provenance.output_accepted,true);
    assert.equal(data.analysis.provenance.provider_response_id,'resp_fixture_only');assert.match(data.analysis.provenance.prompt_sha256,/^[a-f0-9]{64}$/);
    assert.ok(data.analysis.facts.some(fact=>fact.text.includes('HTTP 301')));assert.ok(data.analysis.facts.some(fact=>fact.text.includes('redirected destination')));
    assert.deepEqual(data.analysis.hypotheses[0].support,[solFixtureId()]);assert.ok(!JSON.stringify(data).includes(solFixtureKey));
  }finally{db.close();}
});

test('deployed receiver-bound fetch and manual redirect compatibility are preserved',async()=>{
  const db=new SolD1Fixture(),original=globalThis.fetch;try{
    solFixtureStore(db);
    globalThis.fetch=async function(url,options){assert.equal(this,globalThis);assert.equal(options.redirect,'manual');return solFixtureReply(solFixtureProvider());};
    const data=await (await handleInterpretation(solFixtureRequest(),solFixtureEnv(db))).json();assert.equal(data.analysis.provenance.output_accepted,true);
  }finally{globalThis.fetch=original;db.close();}
});

test('API redirects never forward the server key or trigger a follow-up call',async()=>{
  const db=new SolD1Fixture();try{
    solFixtureStore(db);let calls=0;
    const data=await (await solFixtureCall(solFixtureRequest(),solFixtureEnv(db),async(url,options)=>{calls++;assert.equal(options.redirect,'manual');return new Response(null,{status:302,headers:{Location:'https://attacker.invalid/steal'}});})).json();
    assert.equal(calls,1);assert.equal(data.analysis.provenance.output_accepted,false);assert.ok(!JSON.stringify(data).includes('attacker.invalid'));
  }finally{db.close();}
});

test('a followed redirect reported by the runtime is rejected even with HTTP 200',async()=>{
  const db=new SolD1Fixture();try{solFixtureStore(db);const followed=solFixtureReply(solFixtureProvider());Object.defineProperty(followed,'redirected',{value:true});
    const data=await (await solFixtureCall(solFixtureRequest(),solFixtureEnv(db),async()=>followed)).json();assert.equal(data.analysis.provenance.output_accepted,false);assert.equal(data.analysis.provenance.execution_verified,false);
  }finally{db.close();}
});

test('pre-aborted requests do not reserve budget or invoke a provider',async()=>{
  const db=new SolD1Fixture();try{solFixtureStore(db);let calls=0;const controller=new AbortController();controller.abort();
    const request=new Request(solFixtureRequest(),{signal:controller.signal});const response=await solFixtureCall(request,solFixtureEnv(db),()=>{calls++;throw new Error('no');});
    assert.equal(response.status,499);assert.equal(calls,0);assert.equal(db.sqlite.prepare('SELECT count(*) AS n FROM observatory_model_jobs').get().n,0);
  }finally{db.close();}
});

test('client observations, prompts, imported IDs, cross-origin and cross-site calls are rejected before inference',async()=>{
  const db=new SolD1Fixture();try{
    solFixtureStore(db);let calls=0;const fake=async()=>{calls++;return solFixtureReply(solFixtureProvider());};
    for(const request of [solFixtureRequest(undefined,{body:{evidence_id:solFixtureId(),event:{status:200}}}),solFixtureRequest(undefined,{body:{evidence_id:solFixtureId(),prompt:'attack'}}),
      solFixtureRequest('sensor_imported_fixture'),solFixtureRequest(undefined,{headers:{Origin:'https://attacker.invalid'}}),solFixtureRequest(undefined,{headers:{'Sec-Fetch-Site':'cross-site'}})]){
      const result=await solFixtureCall(request,solFixtureEnv(db),fake);assert.ok([400,403].includes(result.status));
    }
    assert.equal(calls,0);
  }finally{db.close();}
});

test('unretained or expired evidence cannot charge or use client narrative',async()=>{
  const db=new SolD1Fixture();try{
    solFixtureStore(db,solFixtureRaw({timestamp:new Date(Date.now()-9*86400000).toISOString()}));let calls=0;
    const result=await solFixtureCall(solFixtureRequest(),solFixtureEnv(db),()=>{calls++;throw new Error('no');});assert.equal(result.status,404);assert.equal(calls,0);
  }finally{db.close();}
});

test('linked HTTP receipt and native ID share a durable cache without a second model call',async()=>{
  const db=new SolD1Fixture();try{
    solFixtureStore(db);let calls=0;const fake=async()=>{calls++;return solFixtureReply(solFixtureProvider());};
    const first=await (await solFixtureCall(solFixtureRequest(solFixtureLinked),solFixtureEnv(db),fake)).json();
    const second=await (await solFixtureCall(solFixtureRequest(),solFixtureEnv(db),fake)).json();
    assert.equal(first.source_event_id,solFixtureId());assert.equal(second.cached,true);assert.equal(calls,1);
    assert.deepEqual(first.analysis.facts,second.analysis.facts);assert.deepEqual(first.analysis.hypotheses,second.analysis.hypotheses);
    assert.equal(first.analysis.provenance.provider_response_id,second.analysis.provenance.provider_response_id);assert.equal(db.sqlite.prepare('SELECT count(*) AS n FROM observatory_model_jobs').get().n,1);
  }finally{db.close();}
});

test('concurrent alias requests reserve one durable attempt and do not double charge',async()=>{
  const db=new SolD1Fixture();try{
    solFixtureStore(db);let release,calls=0;const fake=()=>{calls++;return new Promise(resolve=>{release=()=>resolve(solFixtureReply(solFixtureProvider()));});};
    const first=solFixtureCall(solFixtureRequest(),solFixtureEnv(db),fake);
    for(let i=0;i<30&&!release;i++)await new Promise(resolve=>setTimeout(resolve,1));
    assert.ok(release);
    const second=await solFixtureCall(solFixtureRequest(solFixtureLinked),solFixtureEnv(db),fake);assert.equal(second.status,409);assert.equal(calls,1);
    release();assert.equal((await first).status,200);
  }finally{db.close();}
});

test('concurrent different evidence IDs atomically share the global cooldown and budget',async()=>{
  const db=new SolD1Fixture();try{
    solFixtureStore(db);solFixtureStore(db,solFixtureRaw({event_id:solFixtureId(2),linked_evidence_id:'evt_'+'b'.repeat(32)}));let calls=0;
    const fake=async()=>{calls++;return solFixtureReply(solFixtureProvider());};
    const replies=await Promise.all([solFixtureCall(solFixtureRequest(),solFixtureEnv(db),fake),solFixtureCall(solFixtureRequest(solFixtureId(2)),solFixtureEnv(db),fake)]);
    assert.deepEqual(replies.map(reply=>reply.status).sort(),[200,429]);assert.equal(calls,1);assert.equal(db.sqlite.prepare('SELECT count(*) AS n FROM observatory_model_jobs').get().n,1);
  }finally{db.close();}
});

test('global cooldown, reserved-dollar limit and request count each stop new provider attempts',async()=>{
  for(const mode of ['cooldown','budget','count']){
    const db=new SolD1Fixture();try{
      solFixtureStore(db);const now=new Date().toISOString(),day=now.slice(0,10),old=new Date(Date.now()-7200000).toISOString();
      const count=mode==='budget'?10:mode==='count'?20:1;
      for(let i=0;i<count;i++)db.sqlite.prepare('INSERT INTO observatory_model_jobs(id,evidence_id,model,prompt_version,day,status,created_at,updated_at,reserved_micro_usd,prompt_sha256) VALUES(?,?,?,?,?,?,?,?,?,?)')
        .run('fixturejob'+i,solFixtureId(i+20),'fixture-model','fixture-v1',day,'failed',mode==='cooldown'?now:old,old,mode==='count'?0:50000,'fixturehash');
      let calls=0;const result=await solFixtureCall(solFixtureRequest(),solFixtureEnv(db),()=>{calls++;throw new Error('no');});assert.equal(result.status,429,mode);assert.equal(calls,0,mode);
    }finally{db.close();}
  }
});

test('timeouts retain reservation, expose uncertain execution and never retry the same evidence',async()=>{
  const db=new SolD1Fixture();try{
    solFixtureStore(db);let calls=0;
    const fake=()=>{calls++;return new Promise(()=>{});};
    const first=await (await solFixtureCall(solFixtureRequest(),solFixtureEnv(db),fake,{timeoutMs:10})).json();
    assert.equal(first.analysis.kind,'deterministic');assert.equal(first.analysis.provenance.model_ran,null);assert.equal(first.analysis.provenance.inference_attempted,true);
    const row=db.sqlite.prepare('SELECT * FROM observatory_model_jobs').get();assert.equal(row.status,'failed');assert.equal(row.reserved_micro_usd,50000);
    const second=await (await solFixtureCall(solFixtureRequest(),solFixtureEnv(db),fake,{timeoutMs:10})).json();assert.equal(second.cached,true);assert.equal(calls,1);
  }finally{db.close();}
});

test('malformed, refused, incomplete, mismatched-model, duplicate or fabricated output cannot create accepted claims',async()=>{
  const fixtures=[{model:'gpt-6-astra'},{status:'incomplete'},{output:[{type:'message',role:'assistant',content:[{type:'refusal',refusal:'test only'}]}]},
    {output:[{type:'message',role:'assistant',content:[{type:'output_text',text:'not JSON'}]}]},
    {output:[{type:'message',role:'assistant',content:[{type:'output_text',text:JSON.stringify({ranking:['attack','corroboration_needed']})}]}]},
    {output:[{type:'message',role:'assistant',content:[{type:'output_text',text:JSON.stringify({ranking:['operational_context','operational_context']})}]}]},
    {usage:{input_tokens:400,output_tokens:1025}},{service_tier:'ultrafast'}];
  for(const overrides of fixtures){const db=new SolD1Fixture();try{solFixtureStore(db);const data=await (await solFixtureCall(solFixtureRequest(),solFixtureEnv(db),async()=>solFixtureReply(solFixtureProvider(overrides)))).json();
      assert.equal(data.analysis.kind,'deterministic');assert.equal(data.analysis.provenance.output_accepted,false);assert.equal(data.analysis.hypotheses.length,0);
      assert.equal(db.sqlite.prepare('SELECT reserved_micro_usd FROM observatory_model_jobs').get().reserved_micro_usd,50000);
    }finally{db.close();}}
});

test('verified completed Sol execution is separate from rejected ranking acceptance',async()=>{
  const db=new SolD1Fixture();try{
    solFixtureStore(db);const bad=solFixtureProvider({output:[{type:'message',role:'assistant',content:[{type:'output_text',text:JSON.stringify({ranking:['invented','corroboration_needed']})}]}]});
    const data=await (await solFixtureCall(solFixtureRequest(),solFixtureEnv(db),async()=>solFixtureReply(bad))).json();
    assert.equal(data.analysis.kind,'deterministic');assert.equal(data.analysis.provenance.model_ran,true);assert.equal(data.analysis.provenance.execution_verified,true);
    assert.equal(data.analysis.provenance.output_accepted,false);assert.equal(data.analysis.provenance.provider_response_id,'resp_fixture_only');assert.equal(data.analysis.hypotheses.length,0);
  }finally{db.close();}
});

test('provider errors and oversized responses do not expose error text or credentials',async()=>{
  for(const fake of [async()=>new Response('secret provider echo '+solFixtureKey,{status:403}),async()=>new Response('x'.repeat(65537)),async()=>{throw new Error('private DNS credential '+solFixtureKey);}]){
    const db=new SolD1Fixture();try{solFixtureStore(db);const data=await (await solFixtureCall(solFixtureRequest(),solFixtureEnv(db),fake)).json();
      assert.equal(data.analysis.kind,'deterministic');assert.ok(!JSON.stringify(data).includes(solFixtureKey));assert.ok(!JSON.stringify(data).includes('private DNS credential'));
    }finally{db.close();}
  }
});

test('missing cache migration fails closed before provider inference',async()=>{
  const db=new SolD1Fixture({modelMigration:false});try{solFixtureStore(db);let calls=0;
    const data=await (await solFixtureCall(solFixtureRequest(),solFixtureEnv(db),()=>{calls++;throw new Error('no');})).json();
    assert.equal(calls,0);assert.equal(data.analysis.provenance.model_ran,false);assert.equal(data.model.state,'storage_unavailable');
  }finally{db.close();}
});

test('native policy context excludes endpoint names and preserves known enforcement without intent claims',()=>{
  const raw=solFixtureRaw({event_type:'policy_event',status:400,rule_id:'non_public_target',rule_action:'deny',outcome:'blocked',test_context:'functional_check'});
  const prepared=solPrepareEvidence(raw);assert.ok(!JSON.stringify(prepared.safe).includes('1.1.1.1'));assert.ok(prepared.catalogue.configured_control);
  assert.ok(prepared.summary.includes('does not establish malicious intent'));assert.deepEqual(prepared.facts[0].evidence_ids,[solFixtureId()]);
  assert.throws(()=>solPrepareEvidence({...raw,rule_action:'rate_limit'}));assert.equal(SOL_LIMITS.daily_reserved_usd/SOL_LIMITS.reserved_usd_per_call,10);
});

test('HTTP 304 is a recorded response and never becomes a redirect explanation',()=>{
  const prepared=solPrepareEvidence(solFixtureRaw({status:304}));
  assert.ok(!prepared.catalogue.redirect_boundary);assert.ok(!prepared.facts.some(fact=>fact.text.includes('redirect response')));
  assert.ok(prepared.facts.some(fact=>fact.text.includes('HTTP 304')));
});

test('old model metadata is bounded without erasing current-day budget reservations',async()=>{
  const db=new SolD1Fixture();try{
    solFixtureStore(db);const now=new Date().toISOString(),old=new Date(Date.now()-9*86400000).toISOString();
    for(const [id,created] of [['old-receipt',old],['current-reservation',now]])db.sqlite.prepare('INSERT INTO observatory_model_jobs(id,evidence_id,model,prompt_version,day,status,created_at,updated_at,reserved_micro_usd,prompt_sha256,result_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)')
      .run(id,solFixtureId(id==='old-receipt'?20:21),'fixture','fixture-v1',created.slice(0,10),'failed',created,created,50000,'fixturehash',JSON.stringify({fixture:'historical metadata'}));
    let calls=0;await solFixtureCall(solFixtureRequest(),{OBSERVATORY_DB:db},()=>{calls++;throw new Error('no');});
    assert.equal(calls,0);assert.equal(db.sqlite.prepare("SELECT count(*) AS n FROM observatory_model_jobs WHERE id='old-receipt'").get().n,0);
    assert.equal(db.sqlite.prepare("SELECT reserved_micro_usd FROM observatory_model_jobs WHERE id='current-reservation'").get().reserved_micro_usd,50000);
    assert.ok(hostedInterpretationCapability({OBSERVATORY_DB:db}).privacy.includes('eight days'));
  }finally{db.close();}
});

test('cached facts and catalogue text are rebuilt instead of trusting stored prose',async()=>{
  const db=new SolD1Fixture();try{
    solFixtureStore(db);const fake=async()=>solFixtureReply(solFixtureProvider());await solFixtureCall(solFixtureRequest(),solFixtureEnv(db),fake);
    const row=db.sqlite.prepare('SELECT id,result_json FROM observatory_model_jobs').get(),corrupted=JSON.parse(row.result_json);corrupted.summary='Fabricated attack prose';corrupted.facts=[{text:'Fabricated secret',evidence_ids:['invented']}];
    corrupted.hypotheses[0].text='Untrusted cached diagnosis';db.sqlite.prepare('UPDATE observatory_model_jobs SET result_json=? WHERE id=?').run(JSON.stringify(corrupted),row.id);
    let calls=0;const data=await (await solFixtureCall(solFixtureRequest(),solFixtureEnv(db),()=>{calls++;throw new Error('no');})).json();
    assert.equal(calls,0);assert.equal(data.cached,true);assert.equal(data.analysis.provenance.output_accepted,true);assert.ok(!JSON.stringify(data).includes('Fabricated'));assert.ok(!JSON.stringify(data).includes('Untrusted cached diagnosis'));
    assert.deepEqual(data.analysis.facts[0].evidence_ids,[solFixtureId()]);
  }finally{db.close();}
});
