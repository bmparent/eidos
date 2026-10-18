/** Bounded, pinned public endpoint probes. This module runs only in Node.
 * A probe observes its own connection, never a target's visitors or packets.
 */
import {lookup} from 'node:dns/promises';
import * as http from 'node:http';
import * as https from 'node:https';
import {isIP} from 'node:net';
import {checkServerIdentity} from 'node:tls';
import {createHash, randomUUID, timingSafeEqual} from 'node:crypto';
import {normalizeTarget, SAMPLE_INTERVAL_MS} from './collector.mjs';

export const PROBE_VERSION = 'target-probes-v1';
export const MEASUREMENT_CONTRACT = 'target-evidence-v2';
export const MAX_ADDRESS_ATTEMPTS = 3;
const MAX_BODY = 65536, MAX_REDIRECTS = 3, DEADLINE = 12000;
const states = ['ok','failed','unavailable','not_applicable','disabled'];
const stamp = () => new Date().toISOString();
const ms = value => Number.isFinite(value) && value >= 0 ? Math.round(value * 1000) / 1000 : null;
const digest = value => createHash('sha256').update(value).digest('hex');
const clean = (value, limit=160) => typeof value === 'string' ? value.replace(/[\u0000-\u001f\u007f]/g,' ').slice(0,limit) : null;
const redirect = value => [301,302,303,307,308].includes(value);

function publicAddress(value) {
  if (!isIP(value)) throw Object.assign(new Error('Invalid public address'),{code:'dns_invalid'});
  return normalizeTarget(value).hostname;
}

async function resolveTarget(target, options, deadline) {
  if(options.signal?.aborted)throw Object.assign(new Error('Probe cancelled'),{code:'ABORT_ERR'});
  if (target.kind !== 'hostname') return {addresses:[target.hostname],address:target.hostname,family:isIP(target.hostname),duration_ms:null,state:'not_applicable'};
  const start=performance.now();
  let timer,cancel;
  try {
    const rows=await Promise.race([(options.lookup ?? lookup)(target.hostname,{all:true,verbatim:true}),new Promise((_,reject)=>{timer=setTimeout(()=>reject(Object.assign(new Error('DNS deadline'),{code:'ETIMEDOUT'})),Math.max(1,deadline-performance.now()));
      cancel=()=>reject(Object.assign(new Error('Probe cancelled'),{code:'ABORT_ERR'}));options.signal?.addEventListener('abort',cancel,{once:true});})]);
    if (!Array.isArray(rows) || !rows.length || rows.length>32) throw Object.assign(new Error('No bounded DNS answer'),{code:'ENOTFOUND'});
    const addresses=[...new Set(rows.map(row=>publicAddress(row.address)))].sort((a,b)=>isIP(a)-isIP(b)||a.localeCompare(b));
    return {addresses,address:addresses.find(x=>isIP(x)===4) ?? addresses[0],family:isIP(addresses.find(x=>isIP(x)===4) ?? addresses[0]),duration_ms:ms(performance.now()-start),state:'ok'};
  } finally { clearTimeout(timer);options.signal?.removeEventListener('abort',cancel); }
}

function responseHeaders(headers) {
  const value=(key,allowed)=>typeof headers[key]==='string' ? allowed.includes(headers[key].toLowerCase().split(';')[0].trim()) ? headers[key].toLowerCase().split(';')[0].trim() : 'other' : null;
  const flags={hsts_present:!!headers['strict-transport-security'],csp_present:!!headers['content-security-policy'],
    x_content_type_options:headers['x-content-type-options']===undefined?'absent':String(headers['x-content-type-options']).toLowerCase()==='nosniff'?'nosniff':'other',
    frame_options:headers['x-frame-options']===undefined?'absent':['deny','sameorigin'].includes(String(headers['x-frame-options']).toLowerCase())?String(headers['x-frame-options']).toLowerCase():'other',
    referrer_policy:headers['referrer-policy']===undefined?'absent':'present',
    cache_private:/(?:^|,)\s*private(?:,|$)/i.test(String(headers['cache-control'] ?? '')),cache_no_store:/(?:^|,)\s*no-store(?:,|$)/i.test(String(headers['cache-control'] ?? ''))};
  const selected={server:value('server',['cloudflare','nginx','apache','vercel','cloudfront']),
    'content-type':value('content-type',['text/html','text/plain','application/json','text/css','application/javascript']),
    'cf-cache-status':value('cf-cache-status',['hit','miss','dynamic','bypass','expired','stale','revalidated','updating'])};
  return {selected,flags,signature:digest(JSON.stringify({server:selected.server,content_type:selected['content-type'],flags}))};
}

function certMetadata(socket) {
  try {
    const cert=socket.getPeerCertificate?.();
    if (!cert || !cert.valid_to) return null;
    const date=value=>Number.isFinite(Date.parse(value))?new Date(value).toISOString():null;
    const fingerprint=typeof cert.fingerprint256==='string'?cert.fingerprint256.replaceAll(':','').toLowerCase():null;
    return {valid_from:date(cert.valid_from),valid_to:date(cert.valid_to),fingerprint256:/^[a-f0-9]{64}$/.test(fingerprint ?? '')?fingerprint:null,
      issuer:clean(cert.issuer?.O ?? cert.issuer?.CN),identity_verified:socket.authorized===true};
  } catch { return null; }
}

function requestOnce(target, dns, {method='HEAD',collectBody=false,signal,requestFactory,deadline}) {
  return new Promise((resolve,reject)=>{
    const start=performance.now();let outgoing,socket,connectAt=null,tlsAt=null,settled=false;
    const result={status:null,headers:{},security_headers:{},header_signature:null,location:null,
      timing:{tcp_ms:null,tls_ms:null,response_wait_ms:null,headers_ms:null},certificate:null,tls_protocol:null,
      peer_ip:null,bytes_read:null,bytes_written:null,page:null};
    const capture=()=>{
      if(!socket)return;
      const peer=String(socket.remoteAddress ?? '').replace(/^::ffff:/,'');
      try {result.peer_ip=publicAddress(peer);}catch{/* A private proxy peer is not retained. */}
      result.bytes_read=Number.isSafeInteger(socket.bytesRead)?Math.max(result.bytes_read ?? 0,socket.bytesRead):result.bytes_read;
      result.bytes_written=Number.isSafeInteger(socket.bytesWritten)?Math.max(result.bytes_written ?? 0,socket.bytesWritten):result.bytes_written;
      result.certificate=certMetadata(socket) ?? result.certificate;
      // Node may return null after socket.destroy(). Capture monotonically.
      const protocol=socket.getProtocol?.();if(protocol)result.tls_protocol=['TLSv1.2','TLSv1.3'].includes(protocol)?protocol:'other';
    };
    const complete=(error)=>{
      if(settled)return;settled=true;clearTimeout(timer);signal?.removeEventListener('abort',cancel);capture();
      if(error){error.probe_result=result;reject(error);}else resolve(result);
    };
    const cancel=()=>{const error=Object.assign(new Error('Probe cancelled'),{code:'ABORT_ERR'});outgoing?.destroy(error);complete(error);};
    const timer=setTimeout(()=>{const error=Object.assign(new Error('Probe deadline'),{code:'ETIMEDOUT'});outgoing?.destroy(error);complete(error);},Math.max(1,deadline-performance.now()));
    try {
      if(signal?.aborted){cancel();return;}
      if(deadline<=performance.now()){complete(Object.assign(new Error('Probe deadline'),{code:'ETIMEDOUT'}));return;}
      const transport=requestFactory ?? (target.url.startsWith('https:')?https.request:http.request);
      const secure=target.url.startsWith('https:');
      outgoing=transport({protocol:secure?'https:':'http:',hostname:dns.address,family:dns.family,port:secure?443:80,
        path:new URL(target.url).pathname,method,agent:false,maxHeaderSize:16384,rejectUnauthorized:true,
        servername:secure&&target.kind==='hostname'?target.hostname:'',checkServerIdentity:(_name,cert)=>checkServerIdentity(target.hostname,cert),
        lookup:(_host,_options,callback)=>callback(Object.assign(new Error('Unexpected DNS resolution'),{code:'DNS_FORBIDDEN'}),'',0),
        headers:{Host:new URL(target.url).host,'User-Agent':'Eidos-Sentinel-Observatory/0.3 (bounded public endpoint probe)',Accept:method==='GET'?'text/html,text/plain,application/json':'*/*','Accept-Encoding':'identity','Cache-Control':'no-cache',Connection:'close'}},response=>{
        const now=performance.now();result.status=response.statusCode ?? null;
        if(!Number.isInteger(result.status)||result.status<100||result.status>599){response.destroy();complete(Object.assign(new Error('Invalid HTTP response'),{code:'INVALID_HTTP'}));return;}
        const headers=responseHeaders(response.headers);result.headers=headers.selected;result.security_headers=headers.flags;result.header_signature=headers.signature;
        result.location=typeof response.headers.location==='string'?response.headers.location.slice(0,769):null;
        result.timing.headers_ms=ms(now-start);result.timing.response_wait_ms=ms(now-(tlsAt ?? connectAt ?? start));capture();
        if(!collectBody){complete();response.destroy();return;}
        // Bodies are opt-in, fixed-size, hashed in memory and never returned.
        const mime=String(response.headers['content-type'] ?? '').split(';')[0].toLowerCase();
        if(!['text/html','text/plain','application/json'].includes(mime)||!['identity',''].includes(String(response.headers['content-encoding'] ?? ''))){
          result.page={state:'unavailable',reason:'unsupported_content_type_or_encoding',bytes:0,complete:false,sha256:null};complete();response.destroy();return;
        }
        let size=0;const chunks=[];
        response.on('data',chunk=>{
          if(settled)return;const bytes=Buffer.isBuffer(chunk)?chunk:Buffer.from(chunk);const remaining=MAX_BODY-size;
          if(bytes.length>remaining){if(remaining>0)chunks.push(bytes.subarray(0,remaining));size=MAX_BODY;
            result.page={state:'ok',bytes:size,complete:false,sha256:digest(Buffer.concat(chunks))};complete();response.destroy();return;}
          chunks.push(bytes);size+=bytes.length;
        });
        response.once('end',()=>{if(settled)return;result.page={state:'ok',bytes:size,complete:true,sha256:digest(Buffer.concat(chunks))};complete();});
        response.once('error',complete);response.once('aborted',()=>complete(Object.assign(new Error('Body interrupted'),{code:'ECONNRESET'})));
      });
      outgoing.once('socket',value=>{socket=value;value.once('connect',()=>{connectAt=performance.now();result.timing.tcp_ms=ms(connectAt-start);});
        if(secure)value.once('secureConnect',()=>{tlsAt=performance.now();result.timing.tls_ms=connectAt===null?null:ms(tlsAt-connectAt);capture();});});
      outgoing.once('error',complete);signal?.addEventListener('abort',cancel,{once:true});outgoing.end();
    } catch(error) {outgoing?.destroy();complete(error);}
  });
}

function classify(error) {
  const code=String(error?.code ?? '');
  return /^(?:ERR_TLS_|ERR_SSL_|CERT_|DEPTH_ZERO|SELF_SIGNED|UNABLE_TO_)/.test(code)?'tls_error':code==='ABORT_ERR'?'cancelled':code==='ETIMEDOUT'?'timeout':/^(?:EAI_|ENOTFOUND|dns_|DNS_)/.test(code)?'dns_failure':'transport_error';
}

function connectionEvidence(target,dns,result) {
  return {scope:'observatory_outbound_head_connection',destination_ip:dns.address,destination_port:target.url.startsWith('https:')?443:80,
    protocol:'TCP',application_protocol:target.url.startsWith('https:')?'HTTPS':'HTTP',peer_ip:result.peer_ip,
    peer_port:target.url.startsWith('https:')?443:80,peer_matches_destination:result.peer_ip===dns.address,
    socket_connected:result.timing.tcp_ms!==null,tls_verified:target.url.startsWith('https:')?result.certificate?.identity_verified===true:null,
    tls_protocol:result.tls_protocol,bytes_read:result.bytes_read,bytes_written:result.bytes_written,packet_counts_available:false,
    counter_scope:'Selected HEAD attempt only; application socket counters'};
}

// All addresses were validated as a set before entering this loop. No new DNS
// resolution, unvalidated address, TLS bypass or timeout multiplication here.
async function attemptAddresses(target,dns,options,deadline,event,hop,method='HEAD',collectBody=false) {
  const addresses=dns.addresses.slice(0,MAX_ADDRESS_ATTEMPTS);let lastError;
  for(let index=0;index<addresses.length;index++) {
    if(options.signal?.aborted)throw Object.assign(new Error('Probe cancelled'),{code:'ABORT_ERR'});
    if(performance.now()>=deadline)break;
    const address=addresses[index],pinned={...dns,address,family:isIP(address)},start=performance.now(),started_at=stamp();
    // Reserve an equal share of remaining time for remaining addresses. Every
    // hop/method also shares the single outer deadline.
    const attemptDeadline=Math.min(deadline,start+(deadline-start)/(addresses.length-index));
    let result,error;
    try{result=await requestOnce(target,pinned,{...options,method,collectBody,deadline:attemptDeadline});}
    catch(failure){error=failure;lastError=failure;result=failure.probe_result;}
    const row={index:event.attempts.length,hop,method,url:target.url,address,family:isIP(address),observer:event.collector_id,
      started_at,completed_at:stamp(),elapsed_ms:ms(performance.now()-start),budget_ms:ms(attemptDeadline-start),
      outcome:error?classify(error):'response',status:result?.status ?? null,
      timing:result?.timing ?? {tcp_ms:null,tls_ms:null,response_wait_ms:null,headers_ms:null},
      tls_protocol:result?.tls_protocol ?? null,certificate:result?.certificate ?? null,
      network:result?connectionEvidence(target,pinned,result):null};
    event.attempts.push(row);
    if(!error)return {result,dns:pinned,attempt:row.index};
    if(classify(error)==='cancelled')throw error;
  }
  throw lastError ?? Object.assign(new Error('Probe deadline'),{code:'ETIMEDOUT'});
}

function retainFirstStages(event,current,dns,result,error=null) {
  if(!result)return;
  const code=error?classify(error):null,secure=current.url.startsWith('https:');
  event.probes.tcp={state:result.timing.tcp_ms!==null?'ok':code==='cancelled'?'unavailable':'failed',duration_ms:result.timing.tcp_ms};
  event.probes.tls={state:!secure?'not_applicable':code==='tls_error'?'failed':result.certificate?.identity_verified===true?'ok':'unavailable',
    duration_ms:result.timing.tls_ms,protocol:result.tls_protocol,certificate:result.certificate};
  event.coverage.network=connectionEvidence(current,dns,result);
}

export async function observeTarget(input, options={}) {
  const target=normalizeTarget(input,{blockedHosts:options.blockedHosts ?? []});
  const start=performance.now(),timeout=Number.isFinite(options.timeoutMs)?Math.max(1,Math.min(DEADLINE,options.timeoutMs)):DEADLINE,deadline=start+timeout,observedAt=stamp();
  const event={id:`evt_${randomUUID().replaceAll('-','')}`,source_id:target.id,url:target.url,target:target.label,
    method:'HEAD',kind:'measurement',observed_at:observedAt,completed_at:null,latency_ms:null,elapsed_ms:0,status:null,error:null,headers:{},security_headers:{},
    redirect:null,redirect_count:0,response_chain_complete:false,
    probe_version:PROBE_VERSION,measurement_contract:MEASUREMENT_CONTRACT,attempts:[],selected_attempt:null,deadline_ms:timeout,
    collector_id:'node-pinned-target-v1:'+(/^[a-z0-9-]{1,32}$/.test(process.env.VERCEL_REGION ?? '')?process.env.VERCEL_REGION:'local'),probe_options:{page_check:options.page_check===true,follow_redirects:options.follow_redirects===true,get_fallback:options.get_fallback===true},
    probes:{dns:{state:'unavailable',duration_ms:null,addresses:[]},tcp:{state:'unavailable',duration_ms:null},tls:{state:target.url.startsWith('https:')?'unavailable':'not_applicable',duration_ms:null,certificate:null},http:{state:'unavailable',response_wait_ms:null,headers_ms:null,header_signature:null,chain:[]},page:{state:options.page_check?'unavailable':'disabled'},path:{state:'unavailable',reason:'local_companion_required'}},
    coverage:{vantage:'Node public-only pinned endpoint observer',transport:{kind:'authenticated_pinned_target_relay',effective_url:target.url},sampling:'One HEAD per hop; optional bounded GET; at most three public redirects',
      head_attempted:false,redirect_followed:false,body_collected:false,dns:{state:'not_checked',address_pinned:true},missing:['Target visitor traffic','Internal network','Packet capture','Origin processing time'],network:null}};
  let current=target,first=true,stage='dns',currentDns=null;
  try {
    for(let hop=0;hop<=MAX_REDIRECTS;hop++) {
      stage='dns';const dns=await resolveTarget(current,options,deadline);currentDns=dns;
      if(first){event.probes.dns={state:dns.state,duration_ms:dns.duration_ms,addresses:dns.addresses};event.coverage.dns={state:dns.state==='not_applicable'?'literal_public_ip_pinned':'resolved_public_and_pinned',address_pinned:true,resolver:dns.state==='not_applicable'?'none':'Node system resolver'};}
      stage='connection';event.coverage.head_attempted=true;
      const selected=await attemptAddresses(current,dns,options,deadline,event,hop),result=selected.result;
      const chain={status:result.status,url:current.url,method:'HEAD',attempt:selected.attempt};event.probes.http.chain.push(chain);
      if(first){
        event.selected_attempt=selected.attempt;
        event.status=result.status;event.latency_ms=result.timing.headers_ms;event.headers=result.headers;event.security_headers=result.security_headers;
        retainFirstStages(event,current,selected.dns,result);
        event.probes.http={...event.probes.http,state:'ok',response_wait_ms:result.timing.response_wait_ms,headers_ms:result.timing.headers_ms,header_signature:result.header_signature};
      }
      let response=result;
      if([405,501].includes(result.status)) {
        event.probes.http.method_coverage='head_rejected';
        if(options.get_fallback===true) {
          const fallback={enabled:true,attempted:true,hop,head_status:result.status,state:'unavailable',status:null,attempt:null};
          event.probes.http.get_fallback=fallback;
          try{const get=await attemptAddresses(current,dns,options,deadline,event,hop,'GET');response=get.result;fallback.state='ok';fallback.status=response.status;fallback.attempt=get.attempt;
            chain.get_status=response.status;chain.get_attempt=get.attempt;
          }catch(error){fallback.state='failed';fallback.reason=classify(error);event.probes.http.chain_state='method_fallback_failed';first=false;break;}
        }
      }
      first=false;
      if(!redirect(response.status)||!response.location){event.response_chain_complete=!redirect(response.status);break;}
      event.redirect={followed:false,target:null,validation:'not_requested'};
      if(!options.follow_redirects||hop===MAX_REDIRECTS){event.probes.http.chain_state=hop===MAX_REDIRECTS?'limit_reached':'not_followed';break;}
      let next;
      try{next=normalizeTarget(new URL(response.location,current.url).toString(),{blockedHosts:options.blockedHosts ?? []});}
      catch{event.probes.http.chain_state='boundary_rejected';break;}
      event.redirect={followed:true,target:next.url,validation:'validated_public_on_each_hop'};event.redirect_count++;event.coverage.redirect_followed=true;current=next;
    }
    if(options.page_check&&event.response_chain_complete){
      stage='page';try{
        const dns=await resolveTarget(current,options,deadline);
        const {result:page}=await attemptAddresses(current,dns,options,deadline,event,event.probes.http.chain.length-1,'GET',true);
        event.probes.page={...(page.page ?? {state:'unavailable'}),status:page.status,scope:'SHA-256 of at most 65536 response bytes; no body retained'};
      }catch(error){event.probes.page={state:'failed',reason:classify(error)};}
    } else if(options.page_check)event.probes.page={state:'unavailable',reason:'incomplete_redirect_chain'};
  } catch(error) {
    const code=classify(error),partial=error?.probe_result;
    if(first){event.error={code,message:'The endpoint probe did not obtain a usable response at this observation point.',scope:stage};
      if(stage==='dns')event.probes.dns={state:'failed',duration_ms:ms(performance.now()-start),addresses:[]};
      if(partial){const last=event.attempts.at(-1);retainFirstStages(event,current,{...currentDns,address:last?.address ?? currentDns.address},partial,error);}}
    else event.probes.http.chain_state='later_hop_failed';
  }
  event.completed_at=stamp();event.elapsed_ms=ms(performance.now()-start);
  const lastHop=event.probes.http.chain.at(-1);event.probes.http.final_status=lastHop?.get_status ?? lastHop?.status ?? null;
  event.coverage.addresses={validated:event.probes.dns.addresses.length,attempted:event.attempts.filter(x=>x.hop===0&&x.method==='HEAD').length,
    limit:MAX_ADDRESS_ATTEMPTS,scope:'Bounded attempts from this observer; untried addresses and target-wide availability remain unknown'};
  return {event,target:{id:target.id,url:target.url,label:target.label},retry_after_ms:SAMPLE_INTERVAL_MS};
}

export function createTargetObservationHandler(options={}) {
  const rates=new Map();let active=0;
  return async request=>{
    const secret=(options.getSecret ?? (()=>process.env.OBSERVATORY_COLLECTOR_SECRET))();
    if(typeof secret!=='string'||secret.length<32||secret.length>1024)return Response.json({error:{code:'collector_not_configured'}},{status:503});
    const supplied=request.headers.get('authorization') ?? '',expected=`Bearer ${secret}`;
    if(Buffer.byteLength(supplied)!==Buffer.byteLength(expected)||!timingSafeEqual(Buffer.from(supplied),Buffer.from(expected)))return Response.json({error:{code:'unauthorized'}},{status:401});
    const headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'};let reserved=false;
    try{
      if(request.method!=='POST')return Response.json({error:{code:'method_not_allowed'}},{status:405,headers});
      if(request.headers.get('content-type')?.split(';')[0].trim()!=='application/json')return Response.json({error:{code:'json_required'}},{status:415,headers});
      const reader=request.body?.getReader();if(!reader)throw new Error('invalid_request');let size=0;const chunks=[];
      try{for(;;){const {done,value}=await reader.read();if(done)break;size+=value.byteLength;if(size>1536){await reader.cancel();return Response.json({error:{code:'payload_too_large'}},{status:413,headers});}chunks.push(value);}}finally{reader.releaseLock();}
      const payload=JSON.parse(Buffer.concat(chunks).toString('utf8'));
      if(!payload||Array.isArray(payload)||Object.keys(payload).some(key=>!['target','page_check','follow_redirects','get_fallback'].includes(key))||!Object.hasOwn(payload,'target')||['page_check','follow_redirects','get_fallback'].some(key=>payload[key]!==undefined&&typeof payload[key]!=='boolean'))throw new Error('invalid_request');
      const blockedHosts=[new URL(request.url).hostname,...String(process.env.OBSERVATORY_HOSTS ?? '').split(',').filter(Boolean)];
      const target=normalizeTarget(payload.target,{blockedHosts});const now=Date.now();
      for(const [key,value] of rates)if(now-value>=SAMPLE_INTERVAL_MS)rates.delete(key);
      if(active>=4||rates.size>=512||now-(rates.get(target.hostname) ?? -Infinity)<SAMPLE_INTERVAL_MS)return Response.json({error:{code:'cooldown'},retry_after_ms:SAMPLE_INTERVAL_MS},{status:429,headers:{...headers,'Retry-After':'15'}});
      rates.set(target.hostname,now);active++;reserved=true;
      const result=await (options.collector ?? observeTarget)(target.url,{page_check:payload.page_check===true,follow_redirects:payload.follow_redirects===true,get_fallback:payload.get_fallback===true,blockedHosts,signal:request.signal});
      return Response.json(result,{headers});
    }catch(error){return Response.json({error:{code:['non_public_target','recursive_target','credentials_not_allowed','unsupported_protocol','unsupported_port','query_not_allowed'].includes(error?.code)?error.code:'invalid_request',message:'The bounded public target request could not be accepted.'}},{status:400,headers});}
    finally{if(reserved)active--;}
  };
}
