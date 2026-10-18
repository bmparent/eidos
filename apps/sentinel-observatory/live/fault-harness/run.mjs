/** Repo-root controlled engineering run. Never probes an external endpoint. */
import {readFileSync,writeFileSync,mkdirSync,readdirSync,statSync} from 'node:fs';
import {resolve,relative,join} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import assert from 'node:assert/strict';
import {observeTarget} from '../node-probes.mjs';
import {normalizeTarget} from '../collector.mjs';
import {normalizeTargetProbe,collectTargetRelayObservation} from '../target-relay.mjs';
import {retainTargetObservation,targetHistory,resolveTargetEvidence} from '../target-store.mjs';
import {TargetBrain} from '../target-brain.mjs';
import {CompanionDatabase} from '../sqlite-store.mjs';
import {probeExplanation} from '../probe-explanation.mjs';
import {prepareTargetInterpretation} from '../target-interpretation.mjs';
import {handleInterpretation} from '../sol-interpreter.mjs';
import {createFaultTransport} from './transport.mjs';

const here=fileURLToPath(new URL('.',import.meta.url)),hash=value=>createHash('sha256').update(value).digest('hex');
const read=name=>JSON.parse(readFileSync(join(here,name),'utf8'));
export const registry=read('scenarios.json'),protocol=read('protocol.json');
const assertions=read('assertions.json').cases;
function verifyLock(){for(const [name,digest] of Object.entries(read('scoring-lock.json').hashes))assert.equal(hash(readFileSync(join(here,name))),digest,`Locked input changed: ${name}`);}

async function modelExercise(fault,env,retained) {
  let calls=0;const configured={...env,OPENAI_API_KEY:'fixture-only-'+'x'.repeat(32)};
  const request=id=>new Request('https://observatory.example/api/interpret',{method:'POST',headers:{Origin:'https://observatory.example','Content-Type':'application/json'},body:JSON.stringify({evidence_id:id})});
  const provider=async(_url,options)=>{
    calls++;const context=JSON.parse(JSON.parse(options.body).input);
    for(const forbidden of ['example.com','1.1.1.1','ground_truth','expected_next_action'])assert.ok(!JSON.stringify(context).includes(forbidden));
    if(fault==='model_rate_limit')return new Response('controlled unavailable',{status:429});
    const ranking=fault==='model_invalid'?['invented','attack']:Object.keys(context.candidates).slice(0,2);
    return Response.json({id:'resp_controlled_fixture',model:'gpt-6.1-sol',status:'completed',usage:{input_tokens:500,output_tokens:20},output:[{type:'message',role:'assistant',content:[{type:'output_text',text:JSON.stringify({ranking})}]}]});
  };
  if(fault==='model_budget'){
    const now=new Date().toISOString();env.OBSERVATORY_DB.sqlite.prepare('INSERT INTO observatory_model_jobs(id,evidence_id,model,prompt_version,day,status,created_at,updated_at,reserved_micro_usd,prompt_sha256) VALUES(?,?,?,?,?,?,?,?,?,?)').run('fixture-budget',retained.id,'fixture','fixture',now.slice(0,10),'failed',now,now,500000,'fixture');
  }
  const first=await handleInterpretation(request(fault==='model_wrong_source'?'edge_'+'f'.repeat(32):retained.id),fault==='model_unavailable'?env:configured,{}, {fetchImpl:provider});
  const result=await first.json();
  if(fault==='model_wrong_source'){assert.equal(first.status,404);assert.equal(calls,0);return {status:first.status,calls,result};}
  if(fault==='model_budget'){assert.equal(first.status,429);assert.equal(calls,0);return {status:first.status,calls,result};}
  if(fault==='model_cache_context'){
    assert.equal(result.analysis.provenance.output_accepted,true);
    env.OBSERVATORY_DB.sqlite.prepare("UPDATE observatory_model_jobs SET prompt_sha256='controlled-prior-context' WHERE evidence_id=?").run(retained.id);
    const cached=await (await handleInterpretation(request(retained.id),configured,{}, {fetchImpl:provider})).json();
    assert.equal(cached.cached,true);assert.equal(cached.analysis.provenance.rejection_code,'cached_context_changed');assert.equal(calls,1);
    return {status:first.status,calls,result,cached};
  }
  assert.equal(result.analysis.kind,'deterministic');assert.equal(result.analysis.provenance.output_accepted,false);
  if(fault==='model_unavailable')assert.equal(calls,0);
  if(fault==='model_rate_limit')assert.equal(result.analysis.provenance.rejection_code,'provider_rate_limited');
  if(fault==='model_invalid')assert.equal(result.analysis.provenance.rejection_code,'provider_output_rejected');
  return {status:first.status,calls,result};
}

export async function executeCase(definition,transport) {
  const {fault}=definition,db=new CompanionDatabase(),env={OBSERVATORY_DB:db};
  try{
    if(fault==='collector_failure'){
      const result=await collectTargetRelayObservation(normalizeTarget('https://example.com/'),{OBSERVATORY_TARGET_PROBES_ENABLED:'true',OBSERVATORY_COLLECTOR_URL:'https://collector.vercel.app/api/observatory/ip-observe',OBSERVATORY_COLLECTOR_SECRET:'a'.repeat(64)}, {fetchImpl:async()=>new Response('controlled collector outage',{status:503})});
      assert.equal(result.collector_error,503);assert.equal(result.event,undefined);
      return {measurement:null,explanation:{stage:'collector',summary:'Collector unavailable; endpoint state is unknown.'},control:result,record_count:0};
    }
    const logical=`${['tcp_refused','tcp_timeout','alternate_refused','alternate_timeout','all_refused','bounded_answers','headers_timeout','headers_delay','headers_disconnect','http408','cancel_headers'].includes(fault)?'http':'https'}://${fault==='tls_hostname'?'wrong.example.com':'example.com'}/${fault}`;
    const options=transport.optionsFor(fault),raw=await observeTarget(logical,options);
    // Fault labels and expected actions exist only in this evaluator object;
    // neither is supplied to the collector, engine or model context.
    const event=normalizeTargetProbe(raw,normalizeTarget(logical),options);
    if(fault==='stale_evidence')event.completed_at=event.observed_at=new Date(Date.now()-8*86400000).toISOString();
    const retained=await retainTargetObservation(env,event);assert.equal(retained.target_storage,'retained');
    const diagnosis=probeExplanation(retained),prepared=prepareTargetInterpretation(retained),ground=assertions[fault];
    assert.equal(diagnosis.stage,ground.stage,'known failure stage');assert.equal(retained.status,ground.status,'observed HTTP status');
    if(ground.final_status!==undefined)assert.equal(retained.probes.http.final_status,ground.final_status);
    if(ground.method_coverage)assert.equal(retained.probes.http.method_coverage,ground.method_coverage);
    assert.ok(retained.elapsed_ms<=retained.deadline_ms+protocol.acceptance.shared_deadline_tolerance_ms,'one bounded deadline');
    let checks={};
    if(fault==='stale_evidence'){assert.equal((await resolveTargetEvidence(env,event.id)).found,false);checks.stale_retrieval_denied=true;}
    else{
      const resolved=await resolveTargetEvidence(env,event.id);assert.equal(resolved.found,true);assert.deepEqual(resolved.event,retained);
      const history=await targetHistory(env,event.source_id);assert.ok(history.events.some(x=>x.id===event.id));
      const exported=JSON.parse(JSON.stringify({schema:'sentinel-live-evidence-v2',evidence_class:'synthetic_controlled',events:history.events}));
      const replay=exported.events.find(x=>x.id===event.id);assert.deepEqual(probeExplanation(replay),diagnosis);checks.export_replay_equal=true;
      for(const fact of prepared.facts)for(const id of fact.evidence_ids){assert.equal(id,event.id);assert.equal((await resolveTargetEvidence(env,id)).event.source_id,event.source_id);}
    }
    if(fault==='missing_evidence'){assert.equal((await resolveTargetEvidence(env,'evt_'+'f'.repeat(32))).found,false);checks.missing_is_unknown=true;}
    if(fault==='duplicate_delivery'){
      const again=await retainTargetObservation(env,event);assert.equal(again.duplicate,true);assert.equal(again.target_analysis.history_samples,1);checks.duplicate_not_trained=true;
    }
    if(fault==='multi_target'){
      const other=await observeTarget('https://example.org/normal',transport.optionsFor('normal'));const otherEvent=normalizeTargetProbe(other,normalizeTarget('https://example.org/normal'));
      await retainTargetObservation(env,otherEvent);assert.ok((await targetHistory(env,event.source_id)).events.every(x=>x.source_id===event.source_id));checks.wrong_target_excluded=true;
    }
    if(['baseline_reset','stable_restore','stable_repeat'].includes(fault)){
      let brain=new TargetBrain();const base=Date.now()-120000;
      for(let i=0;i<16;i++)brain.observe({...event,id:'evt_'+(i+1).toString(16).padStart(32,'0'),completed_at:new Date(base+i*1000).toISOString()});
      brain=new TargetBrain(brain.checkpoint());const next=brain.observe({...event,id:'evt_'+'e'.repeat(32),completed_at:new Date(base+17000).toISOString(),collector_id:fault==='baseline_reset'?'node-pinned-target-v1:other':event.collector_id});
      assert.equal(next.generation,fault==='baseline_reset'?2:1);checks.baseline_restore=next;
    }
    if(fault==='tls_cleanup'||fault==='tls_partial_headers'||fault==='https_headers_timeout'){
      assert.equal(retained.probes.tls.state,'ok');assert.ok(retained.probes.tls.protocol);assert.equal(retained.coverage.network.tls_verified,true);checks.partial_tls_preserved=true;
    }
    if(fault==='alternate_refused'||fault==='alternate_timeout'){assert.equal(event.status,200);assert.ok(event.attempts.length>1);assert.equal(event.attempts[0].status,null);checks.alternate_recovery=true;}
    if(fault==='bounded_answers')assert.equal(event.attempts.length,3);
    if(fault==='page_prefix'){assert.equal(event.probes.page.bytes,65536);assert.equal(event.probes.page.complete,false);}
    if(fault==='page_complete')assert.equal(event.probes.page.complete,true);
    if(fault==='page_dynamic'){
      const later=await observeTarget(logical,transport.optionsFor(fault));assert.notEqual(later.event.probes.page.sha256,event.probes.page.sha256);checks.dynamic_changed=true;
    }
    if(fault.startsWith('model_'))checks.model=await modelExercise(fault,env,retained);
    return {measurement:retained,explanation:diagnosis,prepared,checks,record_count:db.sqlite.prepare('SELECT count(*) AS n FROM observatory_target_records').get().n};
  }finally{db.close();}
}

export async function run(out) {
  verifyLock();mkdirSync(out,{recursive:true});
  const started=new Date().toISOString(),transport=await createFaultTransport(),rows=[];
  try{for(const definition of registry.cases){
    const before=transport.trace.length;
    try{const result=await executeCase(definition,transport);rows.push({definition,status:'passed',...result,transport_trace:transport.trace.slice(before)});}
    catch(error){rows.push({definition,status:'failed',failure:{message:error.message,stack:error.stack},transport_trace:transport.trace.slice(before)});}
    writeFileSync(join(out,'raw_cases.jsonl'),rows.map(row=>JSON.stringify(row)).join('\n')+'\n');
  }}finally{await transport.close();}
  const summary={schema:'observatory-fault-results-v1',evidence_class:'synthetic_controlled',started_at:started,completed_at:new Date().toISOString(),cases:rows.length,
    passed:rows.filter(x=>x.status==='passed').length,failed:rows.filter(x=>x.status==='failed').length,families:new Set(rows.map(x=>x.definition.family)).size,
    splits:Object.fromEntries(['development','locked_evaluation'].map(split=>[split,{total:rows.filter(x=>x.definition.split===split).length,passed:rows.filter(x=>x.definition.split===split&&x.status==='passed').length}])),
    external_endpoint_requests:0,actual_openai_calls:0,limitations:['Synthetic local faults; not real pilot observations.','DNS answers and TCP blackholes are explicit test simulations; HTTP/TLS/refusal cases use real local sockets.','Wall-clock timing and event UUIDs vary; assertions use frozen tolerances.','No independent human explanation scores.','No tenant isolation, hosted unattended scheduling or second observer acceptance inferred.']};
  writeFileSync(join(out,'summary.json'),JSON.stringify(summary,null,2)+'\n');
  writeFileSync(join(out,'case_scores.csv'),'case,family,split,status\n'+rows.map(x=>[x.definition.id,x.definition.family,x.definition.split,x.status].join(',')).join('\n')+'\n');
  for(const name of ['scenarios.json','protocol.json','assertions.json','scoring-lock.json'])writeFileSync(join(out,name),readFileSync(join(here,name)));
  const sourceRoot=resolve(here,'..'),sourceHashes={};
  for(const file of readdirSync(sourceRoot))if(/\.(mjs|py)$/.test(file))sourceHashes[file]=hash(readFileSync(join(sourceRoot,file)));
  for(const file of readdirSync(here))if(/\.(mjs|json)$/.test(file))sourceHashes['fault-harness/'+file]=hash(readFileSync(join(here,file)));
  const manifest={schema:'observatory-engineering-run-v1',source_commit:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),dirty:execFileSync('git',['status','--porcelain'],{encoding:'utf8'}).trim().length>0,command:`node apps/sentinel-observatory/live/fault-harness/run.mjs --out ${relative(process.cwd(),out).replaceAll('\\','/')}`,node:process.version,platform:process.platform,architecture:process.arch,engine:'portable scalar Sentinel V3, 32 reservoirs; unchanged brain.mjs',prompt_version:'observatory-sol-target-stages-v3',input_hashes:read('scoring-lock.json').hashes,source_hashes:sourceHashes,raw_cases_sha256:hash(readFileSync(join(out,'raw_cases.jsonl'))),summary};
  writeFileSync(join(out,'run_manifest.json'),JSON.stringify(manifest,null,2)+'\n');
  console.log(JSON.stringify(summary));return {summary,rows};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href){
  const out=resolve(process.argv[process.argv.indexOf('--out')+1]??'artifacts/proof_runs/2026-10-18/observatory-m1/corpus');
  const {summary}=await run(out);if(summary.failed)process.exitCode=1;
}
