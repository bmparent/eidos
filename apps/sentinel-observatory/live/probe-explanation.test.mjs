import test from 'node:test';
import assert from 'node:assert/strict';
import {probeExplanation,probeResponseText} from './probe-explanation.mjs';
import {prepareTargetInterpretation} from './target-interpretation.mjs';

const event=(overrides={})=>({id:'evt_'+'a'.repeat(32),probe_version:'target-probes-v1',status:200,error:null,elapsed_ms:42,
  probes:{dns:{state:'ok'},tcp:{state:'ok'},tls:{state:'ok'},http:{state:'ok',final_status:200},page:{state:'disabled'}},
  target_analysis:{source:'authoritative_server',state:'observing',generation:1,signals:[],changes:[]},...overrides});

test('a failed TCP check is not an HTTP timeout status and retains the actual 12-second wait',()=>{
  const raw=event({status:null,elapsed_ms:12000,error:{code:'timeout',scope:'connection'},probes:{dns:{state:'not_applicable'},tcp:{state:'failed'},tls:{state:'unavailable'},http:{state:'unavailable'}}});
  const result=probeExplanation(raw),prepared=prepareTargetInterpretation(raw);
  assert.equal(result.stage,'tcp');assert.equal(result.http_status,null);assert.match(result.summary,/12.00 seconds/);assert.match(result.summary,/before a TCP connection/);
  assert.equal(prepared.safe.stage,'tcp');assert.equal(prepared.safe.stages.dns,'not_applicable');assert.match(prepared.catalogue.probe_failed.next_observation,/public IP alone need not host a website/);
  assert.equal(probeResponseText(raw),'Connection timeout');assert.equal(probeResponseText(event({status:408})),'408');
});

test('DNS, TLS and completed-connection HTTP failures give different review instructions',()=>{
  const dns=event({status:null,error:{code:'dns_failure',scope:'dns'}});
  assert.equal(probeExplanation(dns).stage,'dns');assert.match(probeExplanation(dns).next_check,/DNS/);
  const tls=event({status:null,error:{code:'tls_error',scope:'connection'}});
  assert.equal(probeExplanation(tls).stage,'tls');assert.match(probeExplanation(tls).next_check,/Keep certificate validation enabled/);
  const http=event({status:null,error:{code:'timeout',scope:'connection'}});
  assert.equal(probeExplanation(http).stage,'http');assert.match(probeExplanation(http).summary,/response headers timed out/);
});

test('unknown connection detail stays unknown instead of inventing a failed handshake',()=>{
  const raw=event({status:null,error:{code:'timeout'},probes:{}});
  assert.equal(probeExplanation(raw).stage,'connection');assert.match(probeExplanation(raw).summary,/does not locate a more specific failure stage/);
});

test('stable observations cannot select explanations for a change that was never detected',()=>{
  const result=prepareTargetInterpretation(event());
  assert.equal(result.catalogue.normal_variation,undefined);assert.ok(result.catalogue.response_scope);
  assert.ok(result.facts.some(row=>row.text.includes('No configured change threshold')));
});

test('HTTP failures are received responses with status-specific next checks',()=>{
  const result=prepareTargetInterpretation(event({status:503}));
  assert.ok(result.catalogue.http_response);assert.match(result.catalogue.http_response.next_observation,/origin or edge logs/);
  assert.match(result.summary,/HTTP 503/);assert.match(probeExplanation(event({status:403})).next_check,/unauthenticated HEAD/);
});

test('followed and incomplete redirects keep first-hop and destination evidence separate',()=>{
  const raw=event({status:301,coverage:{redirect_followed:true},response_chain_complete:true});
  assert.match(probeExplanation(raw).summary,/HTTP 301.*HTTP 200/);
  assert.match(probeExplanation(event({status:301})).summary,/destination was not verified/);
});

test('calibration and context resets are stated before baseline numbers',()=>{
  const raw=event();raw.target_analysis.state='calibrating';raw.target_analysis.calibration_reset='fixture';
  const result=prepareTargetInterpretation(raw);
  assert.ok(result.facts.some(row=>row.text.includes('baseline restarted')));assert.ok(result.facts.some(row=>row.text.includes('still calibrating')));
});
