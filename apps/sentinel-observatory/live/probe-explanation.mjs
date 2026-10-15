/** Shared factual presentation for collected evidence; no inferred root cause. */
const number=value=>typeof value==='number'&&Number.isFinite(value)&&value>=0?value:null;
const status=value=>Number.isInteger(value)&&value>=100&&value<=599?value:null;
const stageState=value=>['ok','failed','unavailable','not_applicable','disabled'].includes(value)?value:'unavailable';

export function probeExplanation(event={}) {
  const p=event.probes ?? {},http=status(event.status),elapsed=number(event.elapsed_ms) ?? number(event.latency_ms);
  const stages=Object.fromEntries(['dns','tcp','tls','http'].map(key=>[key,stageState(p[key]?.state)]));
  const duration=elapsed===null?'':` after ${(elapsed/1000).toFixed(2)} seconds`;
  let stage='unavailable',summary,next;
  if(event.error){
    const timeout=event.error.code==='timeout';
    if(event.error.scope==='dns'||stages.dns==='failed'){
      stage='dns';summary=`${timeout?'Timed out during DNS lookup':'DNS lookup or public-address validation failed'}${duration}; no first-hop HTTP response was received.`;
      next='Check the hostname and public DNS answers, then compare a later lookup from another authorized observation point.';
    }else if(stages.tcp==='failed'){
      stage='tcp';summary=`${timeout?'Timed out before a TCP connection was established':'A TCP connection could not be established'}${duration}; no first-hop HTTP response was received.`;
      next='Confirm that this address is intended to serve public HTTP/S on the selected port. A public IP alone need not host a website; compare another authorized observation point.';
    }else if(event.error.code==='tls_error'||stages.tls==='failed'){
      stage='tls';summary=`TLS validation or negotiation failed${duration}; no first-hop HTTP response was received.`;
      next='Check the intended hostname and certificate validity. Keep certificate validation enabled.';
    }else if(stages.tcp==='ok'&&(stages.tls==='ok'||stages.tls==='not_applicable')){
      stage='http';summary=`The connection stages completed, but ${timeout?'the wait for first-hop HTTP response headers timed out':'no usable first-hop HTTP response was received'}${duration}.`;
      next='Compare a later HEAD response and authorized application logs. A HEAD check does not establish browser page-load behavior.';
    }else{
      stage='connection';summary=`The endpoint probe ${timeout?'timed out':'failed'}${duration}; no usable first-hop HTTP response was received. The retained record does not locate a more specific failure stage.`;
      next='Compare the recorded connection stages and a later check before attributing the cause.';
    }
  }else if(http!==null){
    stage='http';summary=`The first-hop HEAD request returned HTTP ${http}.`;
    const final=status(p.http?.final_status);
    if(event.coverage?.redirect_followed&&final!==null)summary+=` The followed redirect chain ended with HTTP ${final}${event.response_chain_complete===false?' before collection stopped':''}.`;
    else if([301,302,303,307,308].includes(http))summary+=' The redirect destination was not verified by this first-hop response.';
    next=http>=500?'Compare subsequent responses and authorized origin or edge logs; this status alone does not identify the failing component.':http===429?'Review the endpoint’s retry policy and compare a later request without increasing the sampling rate.':[401,403].includes(http)?'Check whether this endpoint permits unauthenticated HEAD requests; the probe does not send login credentials.':'Compare subsequent observations; this response alone does not verify application functionality or security.';
  }else{
    summary='No first-hop HTTP status was measured.';
    next='Inspect the collection record and retry when the collector is available.';
  }
  return {stage,stages,elapsed_ms:elapsed,http_status:http,summary,next_check:next};
}

export function probeResponseText(event) {
  if(!event)return '—';
  const http=status(event.status);
  if(http!==null)return String(http);
  return event.error?.code==='timeout'?'Connection timeout':event.error?'Probe failed':'No response';
}
