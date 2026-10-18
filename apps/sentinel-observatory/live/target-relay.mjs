/** Allowlisted Worker-to-Node probe contract; credentials remain server-only. */
import {normalizeTarget, ipCollectorConfiguration} from './collector.mjs';

export function targetProbeConfiguration(env={}) {
  const current=ipCollectorConfiguration({...env,OBSERVATORY_COLLECTOR_URL:env.OBSERVATORY_TARGET_COLLECTOR_URL ?? env.OBSERVATORY_COLLECTOR_URL});
  if(!current||env.OBSERVATORY_TARGET_PROBES_ENABLED!=='true')return null;
  return {...current,url:current.url.replace('/api/observatory/ip-observe','/api/observatory/target-observe')};
}
const probeNumber=value=>typeof value==='number'&&Number.isFinite(value)&&value>=0&&value<=86400000?value:null;
const probeTime=value=>typeof value==='string'&&/(?:Z|[+-]\d\d:\d\d)$/.test(value)&&Number.isFinite(Date.parse(value))?new Date(value).toISOString():null;
const probeState=value=>['ok','failed','unavailable','not_applicable','disabled'].includes(value)?value:'unavailable';
const probeHash=value=>typeof value==='string'&&/^[a-f0-9]{64}$/.test(value)?value:null;
const probeStatus=value=>Number.isInteger(value)&&value>=100&&value<=599?value:null;
const probePublic=value=>{try{const target=normalizeTarget(value);return target.kind==='hostname'?null:target.hostname;}catch{return null;}};
const probeEnum=(value,allowed)=>allowed.includes(value)?value:null;
const probeText=value=>typeof value==='string'?value.replace(/[\u0000-\u001f\u007f]/g,' ').slice(0,160):null;

export function normalizeTargetProbe(payload,target,{blockedHosts=[],page_check=false,follow_redirects=false,get_fallback=false}={}) {
  const event=payload?.event,p=event?.probes;
  if(!event||event.source_id!==target.id||payload.target?.url!==target.url||event.url!==target.url||!/^evt_[a-f0-9]{32}$/.test(event.id ?? '')||event.probe_version!=='target-probes-v1'||!/^node-pinned-target-v1(?::[a-z0-9-]{1,32})?$/.test(event.collector_id ?? '')||event.method!=='HEAD'||event.kind!=='measurement'||!probeTime(event.observed_at)||!probeTime(event.completed_at)||Date.parse(event.observed_at)>Date.parse(event.completed_at)||!p)throw new Error('invalid_target_probe_contract');
  const addresses=Array.isArray(p.dns?.addresses)?p.dns.addresses.map(probePublic):[];
  if(addresses.length>32||addresses.some(x=>!x)||target.kind!=='hostname'&&JSON.stringify(addresses)!==JSON.stringify([target.hostname]))throw new Error('invalid_dns_contract');
  const chain=Array.isArray(p.http?.chain)?p.http.chain.map(row=>({status:probeStatus(row.status),url:normalizeTarget(row.url,{blockedHosts}).url,
    ...(row.method?{method:'HEAD',attempt:row.attempt}:{}),...(row.get_status!==undefined?{get_status:probeStatus(row.get_status),get_attempt:row.get_attempt}:{})})):[];
  if(chain.length>4||chain.some(x=>!x.status)||chain.length&&chain[0].url!==target.url||!follow_redirects&&chain.length>1)throw new Error('invalid_redirect_contract');
  const cert=p.tls?.certificate;
  const certificate=cert?{valid_from:probeTime(cert.valid_from),valid_to:probeTime(cert.valid_to),fingerprint256:probeHash(cert.fingerprint256),issuer:probeText(cert.issuer),identity_verified:cert.identity_verified===true}:null;
  const network=event.coverage?.network,peer=probePublic(network?.peer_ip),destination=probePublic(network?.destination_ip);
  if(network&&(destination&&!addresses.includes(destination)||peer&&peer!==destination))throw new Error('unpinned_target_peer');
  const error=event.error?{code:probeEnum(event.error.code,['timeout','cancelled','dns_failure','tls_error','transport_error']) ?? 'transport_error',message:'The observer did not obtain a usable response at this observation point.',scope:probeEnum(event.error.scope,['dns','connection','page']) ?? 'connection'}:null;
  const headers=Object.fromEntries(Object.entries(event.headers ?? {}).filter(([key])=>['server','content-type','cf-cache-status'].includes(key)).map(([key,value])=>[key,probeText(value)]));
  const flags=Object.fromEntries(Object.entries(event.security_headers ?? {}).filter(([key])=>['hsts_present','csp_present','x_content_type_options','frame_options','referrer_policy','cache_private','cache_no_store'].includes(key)).map(([key,value])=>[key,typeof value==='boolean'?value:probeEnum(value,['absent','present','nosniff','deny','sameorigin','other'])]));
  const v2=event.measurement_contract==='target-evidence-v2';
  let attempts=[],selected=null,fallback=null;
  if(event.measurement_contract&&!v2)throw new Error('unknown_measurement_contract');
  if(v2){
    if(!Array.isArray(event.attempts)||event.attempts.length>27||event.probe_options?.get_fallback!==get_fallback||event.probe_options?.page_check!==page_check||event.probe_options?.follow_redirects!==follow_redirects)throw new Error('invalid_attempt_contract');
    attempts=event.attempts.map((row,index)=>{
      const address=probePublic(row.address),url=normalizeTarget(row.url,{blockedHosts}).url,net=row.network;
      if(row.index!==index||!address||row.family!==(address.includes(':')?6:4)||row.observer!==event.collector_id||!Number.isInteger(row.hop)||row.hop<0||row.hop>3||!['HEAD','GET'].includes(row.method)||row.method==='GET'&&!page_check&&!get_fallback||row.hop===0&&(url!==target.url||!addresses.includes(address))||!probeTime(row.started_at)||!probeTime(row.completed_at)||Date.parse(row.started_at)>Date.parse(row.completed_at)||probeNumber(row.elapsed_ms)===null||probeNumber(row.budget_ms)===null||!['response','timeout','cancelled','dns_failure','tls_error','transport_error'].includes(row.outcome))throw new Error('invalid_attempt_contract');
      if(net&&(net.destination_ip!==address||net.peer_ip!==null&&net.peer_ip!==address)||row.outcome==='response'&&probeStatus(row.status)===null||row.outcome!=='response'&&row.status!==null&&!(row.method==='GET'&&probeStatus(row.status)!==null&&probeNumber(row.timing?.headers_ms)!==null))throw new Error('invalid_attempt_outcome');
      const c=row.certificate;
      return {index,hop:row.hop,method:row.method,url,address,family:row.family,observer:row.observer,started_at:probeTime(row.started_at),completed_at:probeTime(row.completed_at),elapsed_ms:probeNumber(row.elapsed_ms),budget_ms:probeNumber(row.budget_ms),outcome:row.outcome,status:probeStatus(row.status),
        timing:Object.fromEntries(['tcp_ms','tls_ms','response_wait_ms','headers_ms'].map(key=>[key,probeNumber(row.timing?.[key])])),tls_protocol:probeEnum(row.tls_protocol,['TLSv1.2','TLSv1.3','other']),
        certificate:c?{valid_from:probeTime(c.valid_from),valid_to:probeTime(c.valid_to),fingerprint256:probeHash(c.fingerprint256),issuer:probeText(c.issuer),identity_verified:c.identity_verified===true}:null,
        network:net?{destination_ip:address,peer_ip:probePublic(net.peer_ip),socket_connected:net.socket_connected===true,tls_verified:typeof net.tls_verified==='boolean'?net.tls_verified:null,tls_protocol:probeEnum(net.tls_protocol,['TLSv1.2','TLSv1.3','other']),bytes_read:probeNumber(net.bytes_read),bytes_written:probeNumber(net.bytes_written)}:null};
    });
    for(const hop of [0,1,2,3])for(const method of ['HEAD','GET'])if(attempts.filter(x=>x.hop===hop&&x.method===method).length>(method==='GET'?6:3))throw new Error('unbounded_attempts');
    selected=Number.isInteger(event.selected_attempt)?event.selected_attempt:null;
    if(selected!==null&&(!attempts[selected]||attempts[selected].hop!==0||attempts[selected].method!=='HEAD'||attempts[selected].outcome!=='response'||attempts[selected].status!==event.status)||event.status!==null&&selected===null)throw new Error('invalid_selected_attempt');
    for(const row of chain)if(!attempts[row.attempt]||attempts[row.attempt].url!==row.url||attempts[row.attempt].status!==row.status||attempts[row.attempt].method!=='HEAD')throw new Error('invalid_chain_attempt');
    const get=p.http?.get_fallback;
    if(get){
      if(!get_fallback||get.enabled!==true||get.attempted!==true||![405,501].includes(get.head_status)||!['ok','failed'].includes(get.state))throw new Error('invalid_get_fallback');
      if(get.state==='ok'&&(!attempts[get.attempt]||attempts[get.attempt].method!=='GET'||attempts[get.attempt].status!==get.status))throw new Error('invalid_get_fallback');
      fallback={enabled:true,attempted:true,hop:get.hop,head_status:get.head_status,state:get.state,status:probeStatus(get.status),attempt:Number.isInteger(get.attempt)?get.attempt:null,reason:probeEnum(get.reason,['timeout','cancelled','dns_failure','tls_error','transport_error'])};
    }
    if(!get_fallback&&chain.some(x=>x.get_status!==undefined))throw new Error('unexpected_get_fallback');
    const last=chain.at(-1),expected=last?.get_status ?? last?.status ?? null;
    if(p.http?.final_status!==expected||chain.length&&event.status!==chain[0].status)throw new Error('inconsistent_status');
  }
  return {id:event.id,source_id:target.id,target:target.label,url:target.url,method:'HEAD',kind:'measurement',
    ...(v2?{measurement_contract:'target-evidence-v2',attempts,selected_attempt:selected,deadline_ms:probeNumber(event.deadline_ms)}:{}),
    observed_at:probeTime(event.observed_at),completed_at:probeTime(event.completed_at),latency_ms:probeNumber(event.latency_ms),elapsed_ms:probeNumber(event.elapsed_ms),status:probeStatus(event.status),error,headers,security_headers:flags,
    redirect:chain.length>1?{followed:true,target:chain[1].url,validation:'public_pinned_on_each_hop'}:event.redirect?{followed:false,target:null,validation:'not_followed'}:null,
    redirect_count:Math.max(0,chain.length-1),response_chain_complete:event.response_chain_complete===true,
    probe_version:'target-probes-v1',collector_id:event.collector_id,probe_options:{page_check,follow_redirects,...(v2?{get_fallback}:{})},
    probes:{dns:{state:probeState(p.dns?.state),duration_ms:probeNumber(p.dns?.duration_ms),addresses},
      tcp:{state:probeState(p.tcp?.state),duration_ms:probeNumber(p.tcp?.duration_ms)},
      tls:{state:probeState(p.tls?.state),duration_ms:probeNumber(p.tls?.duration_ms),protocol:probeEnum(p.tls?.protocol,['TLSv1.2','TLSv1.3','other']),certificate},
      http:{state:probeState(p.http?.state),response_wait_ms:probeNumber(p.http?.response_wait_ms),headers_ms:probeNumber(p.http?.headers_ms),header_signature:probeHash(p.http?.header_signature),chain,final_status:probeStatus(p.http?.final_status),chain_state:probeEnum(p.http?.chain_state,['not_followed','limit_reached','boundary_rejected','later_hop_failed','method_fallback_failed']),
        ...(v2?{method_coverage:probeEnum(p.http?.method_coverage,['head_rejected']),get_fallback:fallback}:{})},
      page:page_check?{state:probeState(p.page?.state),sha256:probeHash(p.page?.sha256),bytes:typeof p.page?.bytes==='number'&&Number.isInteger(p.page.bytes)&&p.page.bytes>=0&&p.page.bytes<=65536?p.page.bytes:null,complete:typeof p.page?.complete==='boolean'?p.page.complete:null,status:probeStatus(p.page?.status),reason:probeEnum(p.page?.reason,['unsupported_content_type_or_encoding','incomplete_redirect_chain','timeout','tls_error','dns_failure','transport_error','cancelled']),scope:'Hash of up to 64 KiB; no body retained'}:{state:'disabled'},
      path:{state:'unavailable',reason:'local_companion_required'}},
    coverage:{vantage:'Node public-only pinned endpoint observer',transport:{kind:'authenticated_pinned_target_relay',effective_url:target.url},
      sampling:'First-hop timings; optional public redirect chain and bounded page fingerprint',head_attempted:event.coverage?.head_attempted===true,redirect_followed:chain.length>1,body_collected:false,
      dns:{state:p.dns?.state==='failed'?'lookup_failed':target.kind==='hostname'?'resolved_public_and_pinned':'literal_public_ip_pinned',address_pinned:addresses.length>0},
      ...(v2?{addresses:{validated:addresses.length,attempted:attempts.filter(x=>x.hop===0&&x.method==='HEAD').length,limit:3,scope:'Bounded attempts from this observer; target-wide availability remains unknown'}}:{}),
      network:network&&destination?{scope:'observatory_outbound_head_connection',destination_ip:destination,destination_port:target.url.startsWith('https:')?443:80,protocol:'TCP',application_protocol:target.url.startsWith('https:')?'HTTPS':'HTTP',peer_ip:peer,peer_port:target.url.startsWith('https:')?443:80,peer_matches_destination:peer===null?null:peer===destination,socket_connected:network.socket_connected===true,tls_verified:typeof network.tls_verified==='boolean'?network.tls_verified:null,tls_protocol:probeEnum(network.tls_protocol,['TLSv1.2','TLSv1.3','other']),bytes_read:probeNumber(network.bytes_read),bytes_written:probeNumber(network.bytes_written),packet_counts_available:false,counter_scope:'This HEAD connection only; application socket counters'}:null,
      missing:['Target visitor traffic','Internal network','Packet capture','Origin processing time','Hosted ICMP or traceroute']}};
}

export async function collectTargetRelayObservation(target,env,options={}) {
  const config=targetProbeConfiguration(env);if(!config)throw new Error('target_probes_not_configured');
  const headers={'Content-Type':'application/json',Authorization:`Bearer ${config.secret}`};if(config.bypass)headers['x-vercel-protection-bypass']=config.bypass;
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),15000);const cancel=()=>controller.abort();options.signal?.addEventListener('abort',cancel,{once:true});
  try{
    const response=await (options.fetchImpl ?? globalThis.fetch.bind(globalThis))(config.url,{method:'POST',headers,body:JSON.stringify({target:target.url,page_check:options.page_check===true,follow_redirects:options.follow_redirects===true,...(options.get_fallback===true?{get_fallback:true}:{})}),redirect:'manual',signal:controller.signal});
    if(!response.ok||response.redirected){void response.body?.cancel().catch(()=>{});return {collector_error:response.status,retry_after_ms:15000};}
    const reader=response.body?.getReader();if(!reader)throw new Error('empty_probe_response');const chunks=[];let size=0;
    try{for(;;){const {done,value}=await reader.read();if(done)break;size+=value.byteLength;if(size>65536){await reader.cancel();throw new Error('probe_response_too_large');}chunks.push(value);}}finally{reader.releaseLock();}
    const bytes=new Uint8Array(size);let at=0;for(const chunk of chunks){bytes.set(chunk,at);at+=chunk.length;}
    const payload=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes));
    const event=normalizeTargetProbe(payload,target,options);return {event,target:{id:target.id,url:target.url,label:target.label},retry_after_ms:15000};
  }catch{return {collector_error:503,retry_after_ms:15000};}
  finally{clearTimeout(timer);options.signal?.removeEventListener('abort',cancel);}
}
