import test from 'node:test';
import assert from 'node:assert/strict';
import {EventEmitter} from 'node:events';
import {createHash} from 'node:crypto';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync} from 'node:fs';
import {observeTarget,createTargetObservationHandler} from './node-probes.mjs';
import {normalizeTargetProbe} from './target-relay.mjs';
import {normalizeTarget} from './collector.mjs';
import {TargetBrain} from './target-brain.mjs';
import {retainTargetObservation,targetHistory,resolveTargetEvidence} from './target-store.mjs';
import {prepareTargetInterpretation} from './target-interpretation.mjs';
import {handleInterpretation} from './sol-interpreter.mjs';

const secret='a'.repeat(64);
function transport(responses=[{}],inspect=()=>{}) {
  let index=0;
  return (options,callback)=>{
    inspect(options);const request=new EventEmitter();request.destroy=()=>request;
    request.end=()=>{queueMicrotask(()=>{
      const definition=responses[Math.min(index++,responses.length-1)],socket=new EventEmitter();
      Object.assign(socket,{remoteAddress:options.hostname,bytesRead:180,bytesWritten:120,authorized:true,getProtocol:()=>options.protocol==='https:'?'TLSv1.3':null,
        getPeerCertificate:()=>({valid_from:'Oct 01 00:00:00 2026 GMT',valid_to:'Oct 01 00:00:00 2027 GMT',fingerprint256:'ab'.repeat(32),issuer:{O:'Fixture issuer'}})});
      request.emit('socket',socket);socket.emit('connect');if(options.protocol==='https:')socket.emit('secureConnect');
      if(definition.error){request.emit('error',Object.assign(new Error('secret raw error'),{code:definition.error}));return;}
      const response=new EventEmitter();Object.assign(response,{statusCode:definition.status ?? 200,headers:definition.headers ?? {'content-type':'text/html'},socket,destroy(){return this;}});callback(response);
      if(options.method==='GET')queueMicrotask(()=>{response.emit('data',Buffer.from(definition.body ?? 'Fixture content'));response.emit('end');});
    });return request;};return request;
  };
}
const lookup=async()=>[{address:'1.1.1.1',family:4}];

test('domain probes pin every socket and preserve DNS, TCP, TLS and HTTP stages',async()=>{
  const calls=[];const result=await observeTarget('https://example.com',{lookup,requestFactory:transport([{}],x=>calls.push(x))});
  assert.equal(calls.length,1);assert.equal(calls[0].hostname,'1.1.1.1');assert.equal(calls[0].servername,'example.com');assert.equal(calls[0].headers.Host,'example.com');
  assert.equal(calls[0].rejectUnauthorized,true);assert.equal(calls[0].method,'HEAD');assert.ok(!('Authorization' in calls[0].headers));
  for(const key of ['dns','tcp','tls','http'])assert.equal(result.event.probes[key].state,'ok',key);
  assert.equal(result.event.probes.tls.certificate.identity_verified,true);
  assert.ok(result.event.probes.tcp.duration_ms>=0);assert.ok(result.event.probes.http.response_wait_ms>=0);
  const normalized=normalizeTargetProbe(result,normalizeTarget('https://example.com'));
  assert.equal(normalized.coverage.dns.address_pinned,true);assert.equal(normalized.probes.page.state,'disabled');
});

test('literal IP has no invented DNS timing and HTTP has no invented TLS timing',async()=>{
  const result=await observeTarget('http://1.1.1.1',{lookup:()=>assert.fail('DNS must not run for a literal IP'),requestFactory:transport()});
  assert.equal(result.event.probes.dns.state,'not_applicable');assert.equal(result.event.probes.dns.duration_ms,null);
  assert.equal(result.event.probes.tls.state,'not_applicable');assert.equal(result.event.probes.tls.duration_ms,null);
});

test('mixed public/private DNS answers reject before any connection; failure stays evidence',async()=>{
  const result=await observeTarget('example.com',{lookup:async()=>[{address:'1.1.1.1'},{address:'127.0.0.1'}],requestFactory:()=>assert.fail('unsafe DNS must not reach HTTP')});
  assert.equal(result.event.probes.dns.state,'failed');assert.equal(result.event.coverage.head_attempted,false);assert.equal(result.event.status,null);
  assert.equal(result.event.probes.tcp.duration_ms,null);assert.ok(result.event.error);
});

test('optional page check hashes a bounded body without retaining its content',async()=>{
  const body='private fixture text '.repeat(5000),methods=[];
  const result=await observeTarget('example.com',{lookup,page_check:true,requestFactory:transport([{}, {body}],x=>methods.push(x.method))});
  assert.deepEqual(methods,['HEAD','GET']);assert.equal(result.event.probes.page.bytes,65536);assert.equal(result.event.probes.page.complete,false);
  assert.equal(result.event.probes.page.sha256,createHash('sha256').update(Buffer.from(body).subarray(0,65536)).digest('hex'));
  assert.ok(!JSON.stringify(result).includes('private fixture text'));
});

test('redirect following is opt-in and public validation applies to every hop',async()=>{
  let calls=0;const responses=[{status:301,headers:{location:'http://127.0.0.1/private'}}];
  const result=await observeTarget('example.com',{lookup,follow_redirects:true,requestFactory:transport(responses,()=>calls++)});
  assert.equal(calls,1);assert.equal(result.event.probes.http.chain_state,'boundary_rejected');assert.equal(result.event.status,301);
  assert.equal(result.event.response_chain_complete,false);
  const allowed=await observeTarget('example.com',{lookup,follow_redirects:true,page_check:true,requestFactory:transport([{status:301,headers:{location:'https://example.org/'}},{status:200},{body:'OK'}])});
  assert.equal(allowed.event.probes.http.chain.length,2);assert.equal(allowed.event.status,301);assert.equal(allowed.event.probes.http.final_status,200);
  assert.equal(allowed.event.probes.page.complete,true);
});

test('TLS validation failure preserves successful TCP timing without an insecure retry',async()=>{
  let calls=0;const result=await observeTarget('example.com',{lookup,requestFactory:transport([{error:'ERR_TLS_CERT_ALTNAME_INVALID'}],x=>{calls++;assert.equal(x.rejectUnauthorized,true);})});
  assert.equal(calls,1);assert.equal(result.event.error.code,'tls_error');assert.equal(result.event.probes.tcp.state,'ok');assert.equal(result.event.probes.tls.state,'failed');
  assert.ok(!JSON.stringify(result).includes('secret raw error'));
});

test('relay contract rejects a mismatched destination, false probe options and unsafe redirect URLs',async()=>{
  const result=await observeTarget('example.com',{lookup,requestFactory:transport()});
  const target=normalizeTarget('example.com');assert.doesNotThrow(()=>normalizeTargetProbe(result,target));
  const wrong=structuredClone(result);wrong.event.coverage.network.peer_ip='8.8.8.8';assert.throws(()=>normalizeTargetProbe(wrong,target));
  const unsafe=structuredClone(result);unsafe.event.probes.http.chain[0].url='http://127.0.0.1/';assert.throws(()=>normalizeTargetProbe(unsafe,target));
});

test('collector authenticates before parsing or probing and bounds its request schema',async()=>{
  let calls=0;const handler=createTargetObservationHandler({getSecret:()=>secret,collector:async url=>{calls++;return {target:url};}});
  const request=(data,authorization=`Bearer ${secret}`)=>new Request('https://collector.example/api/observatory/target-observe',{method:'POST',headers:{authorization,'content-type':'application/json'},body:JSON.stringify(data)});
  assert.equal((await handler(request({target:'127.0.0.1'},'Bearer wrong'))).status,401);assert.equal(calls,0);
  assert.equal((await handler(request({target:'example.com',page_check:'true'}))).status,400);
  assert.equal((await handler(request({target:'example.com',headers:{Authorization:'secret'}}))).status,400);assert.equal(calls,0);
  assert.equal((await handler(request({target:'example.com',page_check:true}))).status,200);assert.equal(calls,1);
  assert.equal((await handler(request({target:'example.com'}))).status,429);
});

function fixture(index=1,extra={}) {
  const time=new Date(Date.now()-60000+index*1000).toISOString();
  return {id:'evt_'+index.toString(16).padStart(32,'0'),source_id:'target_0123456789abcdef',url:'https://example.com/',target:'example.com',method:'HEAD',kind:'measurement',observed_at:time,completed_at:time,
    status:200,error:null,probe_version:'target-probes-v1',collector_id:'node-pinned-target-v1:test',probe_options:{page_check:false,follow_redirects:false},
    probes:{dns:{state:'ok',duration_ms:5,addresses:['1.1.1.1']},tcp:{state:'ok',duration_ms:12},tls:{state:'ok',duration_ms:15,certificate:{valid_to:'2027-10-01T00:00:00.000Z',fingerprint256:'a'.repeat(64)}},http:{state:'ok',headers_ms:50,response_wait_ms:20,header_signature:'b'.repeat(64),chain:[]},page:{state:'disabled'},path:{state:'unavailable'}},...extra};
}

test('independent metric streams survive checkpoint restore; duplicates do not train twice',()=>{
  let brain=new TargetBrain();for(let i=1;i<=14;i++)brain.observe(fixture(i));
  const prior=brain.checkpoint();brain=new TargetBrain(prior);const result=brain.observe(fixture(15));
  assert.equal(result.history_samples,15);assert.equal(result.signals.find(x=>x.metric==='tcp_ms').sample_count,15);assert.equal(result.signals.find(x=>x.metric==='tcp_ms').prior_median,12);
  assert.equal(brain.observe(fixture(15)).state,'duplicate');assert.equal(brain.checkpoint().samples,15);
  assert.deepEqual(result.signals.find(x=>x.metric==='tcp_ms').baseline_evidence_ids.length,14);
});

test('missing timing is excluded and failure has its own actual outcome stream',()=>{
  const brain=new TargetBrain();brain.observe(fixture(1));const failed=fixture(2,{error:{code:'timeout'},status:null});failed.probes.tcp={state:'failed',duration_ms:null};
  const result=brain.observe(failed),tcp=result.signals.find(x=>x.metric==='tcp_ms');
  assert.equal(tcp.value,null);assert.equal(tcp.sample_count,1);assert.equal(result.signals.find(x=>x.metric==='failure_ratio').value,0.5);
  assert.ok(result.changes.some(x=>x.code==='probe_failed'));
});

test('categorical DNS/certificate changes retain evidence and are not attack verdicts',()=>{
  const brain=new TargetBrain();brain.observe(fixture(1));const changed=fixture(2);changed.probes.dns.addresses=['8.8.8.8'];changed.probes.tls.certificate.fingerprint256='c'.repeat(64);
  const result=brain.observe(changed);assert.ok(result.changes.some(x=>x.code==='dns_changed'));assert.ok(result.changes.some(x=>x.code==='certificate_changed'));
  assert.equal(result.security_verdict,'not_assessed');assert.equal(result.episode.evidence_ids.length,2);
});

test('collector/settings changes rewarm; stale and out-of-order frames never contaminate calibration',()=>{
  const brain=new TargetBrain();for(let i=1;i<=14;i++)brain.observe(fixture(i));
  assert.equal(brain.observe(fixture(3)).state,'duplicate');
  const late=fixture(100,{completed_at:fixture(2).completed_at});assert.equal(brain.observe(late).state,'late_excluded');
  const result=brain.observe(fixture(15,{collector_id:'node-pinned-target-v1:other'}));assert.equal(result.generation,2);assert.equal(result.signals[0].sample_count,1);assert.match(result.calibration_reset,/Collector/);
});

class D1SQLite {
  constructor(){this.sqlite=new DatabaseSync(':memory:');for(const name of ['0001_observatory_sensor','0002_observatory_model','0003_target_streams'])this.sqlite.exec(readFileSync(new URL('./migrations/'+name+'.sql',import.meta.url),'utf8'));}
  prepare(sql){const statement=this.sqlite.prepare(sql);return {bind:(...args)=>({first:async()=>statement.get(...args) ?? null,all:async()=>({success:true,results:statement.all(...args)}),run:async()=>{const r=statement.run(...args);return {success:true,meta:{changes:r.changes}};},_run:()=>statement.run(...args)})};}
  async batch(statements){this.sqlite.exec('BEGIN');try{const rows=statements.map(s=>{const r=s._run();return {success:true,meta:{changes:r.changes}};});this.sqlite.exec('COMMIT');return rows;}catch(e){this.sqlite.exec('ROLLBACK');throw e;}}
  close(){this.sqlite.close();}
}

test('SQLite persistence, server history and duplicate delivery preserve one learned sample per evidence ID',async()=>{
  const db=new D1SQLite(),env={OBSERVATORY_DB:db};
  try{
    for(let i=1;i<=14;i++)assert.equal((await retainTargetObservation(env,fixture(i))).target_storage,'retained');
    const again=await retainTargetObservation(env,fixture(14));assert.equal(again.duplicate,true);assert.equal(again.target_analysis.history_samples,14);
    const history=await targetHistory(env,'target_0123456789abcdef');assert.equal(history.events.length,14);
    assert.equal((await resolveTargetEvidence(env,fixture(1).id)).found,true);
    assert.equal((await resolveTargetEvidence(env,'evt_'+'f'.repeat(32))).found,false);
    assert.equal(db.sqlite.prepare('SELECT revision FROM observatory_target_states').get().revision,14);
  }finally{db.close();}
});

test('concurrent observations use compare-and-swap without lost learning or duplicate rows',async()=>{
  const db=new D1SQLite(),env={OBSERVATORY_DB:db};try{
    const results=await Promise.all([retainTargetObservation(env,fixture(1)),retainTargetObservation(env,fixture(2))]);
    assert.ok(results.every(x=>x.target_storage==='retained'));assert.equal(db.sqlite.prepare('SELECT count(*) AS n FROM observatory_target_records').get().n,2);
    const saved=JSON.parse(db.sqlite.prepare('SELECT state_json FROM observatory_target_states').get().state_json);assert.equal(saved.samples,2);
  }finally{db.close();}
});

test('Sol receives retained metric comparisons and scores while raw endpoints, addresses and fingerprints stay excluded',async()=>{
  const db=new D1SQLite(),env={OBSERVATORY_DB:db,OPENAI_API_KEY:'fixture-not-a-real-key'};let calls=0,context;
  try{
    let retained;for(let i=1;i<=14;i++)retained=await retainTargetObservation(env,fixture(i));
    const prepared=prepareTargetInterpretation(retained);assert.ok(prepared.safe.metrics.some(x=>x.metric==='tcp_ms'&&x.prior_median===12));
    const request=()=>new Request('https://observatory.example/api/interpret',{method:'POST',headers:{Origin:'https://observatory.example','Content-Type':'application/json'},body:JSON.stringify({evidence_id:retained.id})});
    const fetchImpl=async(_url,options)=>{calls++;context=JSON.parse(JSON.parse(options.body).input);const ranking=Object.keys(context.candidates).slice(0,2);
      return Response.json({id:'resp_fixturetarget123',model:'gpt-6.1-sol',status:'completed',usage:{input_tokens:600,output_tokens:40},output:[{type:'message',role:'assistant',content:[{type:'output_text',text:JSON.stringify({ranking})}]}]});};
    const first=await handleInterpretation(request(),env,{}, {fetchImpl});const body=await first.json();assert.equal(first.status,200);assert.equal(body.analysis.provenance.output_accepted,true);
    assert.ok(context.measurement.metrics.some(x=>x.metric==='tcp_ms'&&x.event_score!==null));
    for(const value of ['example.com','1.1.1.1','a'.repeat(64)])assert.ok(!JSON.stringify(context).includes(value));
    const second=await handleInterpretation(request(),env,{}, {fetchImpl});assert.equal((await second.json()).cached,true);assert.equal(calls,1);
  }finally{db.close();}
});
