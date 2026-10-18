import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {observeTarget} from './node-probes.mjs';
import {normalizeTarget} from './collector.mjs';
import {normalizeTargetProbe} from './target-relay.mjs';
import {probeExplanation} from './probe-explanation.mjs';
import {createFaultTransport} from './fault-harness/transport.mjs';
import {executeCase,registry} from './fault-harness/run.mjs';

test('real fixture stages persist, explain and replay; negative controls remain unknown',async()=>{
  const transport=await createFaultTransport();
  try{for(const fault of ['tls_partial_headers','tls_cleanup','alternate_refused','alternate_timeout','get405_ok','http408','redirect_final503','stale_evidence','duplicate_delivery','model_cache_context','model_wrong_source']){
    const result=await executeCase(registry.cases.find(x=>x.fault===fault),transport);assert.ok(result.measurement);
  }}finally{await transport.close();}
});

test('real body prefix completion precedes socket abort; interrupted bodies retain received HTTP status',async()=>{
  const transport=await createFaultTransport();try{
    for(const fault of ['page_prefix','page_interrupted']){
      const result=await executeCase(registry.cases.find(x=>x.fault===fault),transport),get=result.measurement.attempts.find(x=>x.method==='GET');
      assert.equal(get.status,200);assert.ok(get.timing.headers_ms>=0);
      assert.equal(result.measurement.probes.page.state,fault==='page_prefix'?'ok':'failed');
    }
  }finally{await transport.close();}
});

test('attempt contract rejects forged peer, selected response, family, option and chain status',async()=>{
  const transport=await createFaultTransport();try{
    const raw=await observeTarget('https://example.com/normal',transport.optionsFor('normal')),target=normalizeTarget('https://example.com/normal');
    for(const mutate of [x=>x.event.attempts[0].address='127.0.0.1',x=>x.event.attempts[0].family=6,x=>x.event.selected_attempt=9,x=>x.event.probe_options.get_fallback=true,x=>x.event.probes.http.final_status=503,x=>x.event.attempts[0].observer='forged']){
      const clone=structuredClone(raw);mutate(clone);assert.throws(()=>normalizeTargetProbe(clone,target));
    }
    assert.equal(normalizeTargetProbe(raw,target).measurement_contract,'target-evidence-v2');
    const legacy=structuredClone(raw);delete legacy.event.measurement_contract;delete legacy.event.attempts;delete legacy.event.selected_attempt;
    assert.equal(normalizeTargetProbe(legacy,target).status,200);
  }finally{await transport.close();}
});

test('production Node entry and Worker builder have no import path to the fixture adapter',()=>{
  for(const file of ['node-probes.mjs','node-companion.mjs','worker-entry.mjs','build_worker.py'])assert.ok(!readFileSync(new URL(file,import.meta.url),'utf8').includes('fault-harness/'));
  assert.ok(!readFileSync(new URL('../../sentinel-lab/app/api/observatory/target-observe/route.ts',import.meta.url),'utf8').includes('fault-harness'));
});

test('HEAD coverage and cancellation wording do not manufacture an outage or an HTTP timeout response',()=>{
  for(const status of [405,501])assert.match(probeExplanation({status,probes:{http:{get_fallback:{attempted:true,status:200}}}}).summary,/method coverage.*separate GET fallback returned HTTP 200/);
  assert.equal(probeExplanation({status:null,error:{code:'cancelled'}}).stage,'cancelled');
});

test('unmeasured TLS validation and a missing peer remain unknown, rather than failed',async()=>{
  const transport=await createFaultTransport();try{
    const raw=await observeTarget('https://example.com/tcp_timeout',transport.optionsFor('tcp_timeout'));
    assert.equal(raw.event.coverage.network.tls_verified,null);assert.equal(raw.event.coverage.network.peer_matches_destination,null);
    const event=normalizeTargetProbe(raw,normalizeTarget('https://example.com/tcp_timeout'));
    assert.equal(event.coverage.network.tls_verified,null);assert.equal(event.coverage.network.peer_matches_destination,null);
  }finally{await transport.close();}
});
