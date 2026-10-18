import {LiveBrain, analyzeEvidence, ENGINE_PROVENANCE} from '/brain.mjs';
import {renderTargetPanel} from '/target-panel.mjs';
import {probeExplanation, probeResponseText} from '/probe-explanation.mjs';

// All received content is inert data. Neither measurements nor stored records
// can execute code, provide instructions, or initiate defensive actions.
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const finite = value => typeof value === 'number' && Number.isFinite(value) ? value : null;
const stamp = value => typeof value === 'number' ? value : Date.parse(value);
const formatNumber = value => finite(value) === null ? '—' : Math.round(value).toLocaleString();
const clock = value => Number.isFinite(stamp(value)) ? new Date(value).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}) : '—';
const dateTime = value => Number.isFinite(stamp(value)) ? new Date(value).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}) : '—';
const list = value => Array.isArray(value) ? value : [];
const textValue = value => value === null || value === undefined ? 'Not supplied' : typeof value === 'object' ? JSON.stringify(value) : String(value);
const uuid = () => typeof crypto.randomUUID === 'function' ? crypto.randomUUID() : `client-${Date.now()}-${Math.random().toString(16).slice(2)}`;
const STORAGE_KEY = 'sentinel-live-observations-v1';
const MIN_INTERVAL = 15000;
const MAX_EVENTS = 500;
const MAX_TARGETS = 4;
const state = {targets:[],events:[],selectedId:null,followLatest:true,sourceFilter:null,filter:'all',limit:8,capabilities:null,sessionStarted:new Date().toISOString(),seen:new Set(),fieldRendered:false,inspectorSignature:null,storageAvailable:true,restored:false,sessionNotice:null};
const modelState={engine:'hosted',hostedStatus:null,companionStatus:null,busy:false,issue:null,lastAttemptId:null};
let networkPanel=null;
let toastTimer;
let fieldFreshnessState='';
const visible = () => document.visibilityState !== 'hidden';
const connected = target => state.targets.includes(target) && target.active;
const latest = sourceId => state.events.find(event => event.source_id === sourceId);
const sourceEvents = sourceId => state.events.filter(event => event.source_id === sourceId);
const selectedEvent = () => state.events.find(event => event.id === state.selectedId);
const targetFor = sourceId => state.targets.find(target => target.sourceId === sourceId);
const labelFor = event => targetFor(event.source_id)?.label || event.target || safeLabel(event.url) || 'Public endpoint';
const isIncomplete = event => !!event.error || event.status === null || event.status === undefined || event.response_chain_complete === false;
const responseText = probeResponseText;
const safeLabel = value => { try { const url=new URL(value);return url.host+(url.pathname==='/'?'':url.pathname); } catch { return String(value || '').slice(0,120); } };
const statusColor = (event, changes=new Set()) => isIncomplete(event) ? '#ec9a8f' : event.status>=400 || changes.has(event.id) ? '#efbc7f' : '#b4eedc';

function notify(message) {
  $('toast').textContent=message; $('toast').hidden=false;
  clearTimeout(toastTimer); toastTimer=setTimeout(()=>{$('toast').hidden=true;},4500);
}
function formMessage(message) { $('form-message').textContent=message || ''; $('form-message').hidden=!message; }
function newBrain() { return new LiveBrain({epochId:uuid(),startedAt:new Date().toISOString()}); }
function readableError(payload, fallback) { const error=payload?.error; return String(typeof error==='string'?(payload?.message||error):error?.message || payload?.message || fallback).slice(0,400); }
function normalizeInput(value) {
  const trimmed=String(value || '').trim();
  if(!trimmed || trimmed.length>768)throw new Error('Enter a public URL or IP address, up to 768 characters.');
  if(/[\u0000-\u0020\u007f\\]/.test(trimmed))throw new Error('Use a public URL without spaces or control characters.');
  let candidate=trimmed;
  if(!/^[a-z][a-z0-9+.-]*:\/\//i.test(candidate)) {
    if(/^[0-9a-f:]+$/i.test(candidate)&&candidate.includes(':'))candidate=`[${candidate}]`;
    candidate='https://'+candidate;
  }
  let url;try{url=new URL(candidate);}catch{throw new Error('That does not look like a public website or IP address.');}
  if(!['https:','http:'].includes(url.protocol))throw new Error('Only HTTP and HTTPS endpoints can be observed.');
  if(url.username||url.password)throw new Error('Do not include credentials in a monitoring URL. Use a public endpoint.');
  if(candidate.includes('?')||candidate.includes('#'))throw new Error('Choose a fixed public path without a query string or fragment.');
  if(url.port&&!['80','443'].includes(url.port))throw new Error('Only HTTP/S ports 80 and 443 are supported.');
  return url.toString();
}

function persist() {
  try {
    const targets=state.targets.map(target=>({id:target.id,sourceId:target.sourceId,url:target.url,label:target.label,createdAt:target.createdAt}));
    sessionStorage.setItem(STORAGE_KEY,JSON.stringify({version:1,saved_at:new Date().toISOString(),session_started_at:state.sessionStarted,targets,events:state.events.slice(0,MAX_EVENTS)}));
  } catch { state.storageAvailable=false; }
}
function restore() {
  try {
    const saved=JSON.parse(sessionStorage.getItem(STORAGE_KEY)||'null');
    if(!saved || saved.version!==1)return;
    state.events=list(saved.events).filter(event=>event&&typeof event.id==='string'&&typeof event.source_id==='string'&&Number.isFinite(stamp(event.observed_at))).slice(0,MAX_EVENTS).sort((a,b)=>stamp(b.observed_at)-stamp(a.observed_at));
    state.targets=list(saved.targets).slice(0,MAX_TARGETS).flatMap(target=>{
      try {
        const url=normalizeInput(target.url);
        return [{id:String(target.id||uuid()),sourceId:typeof target.sourceId==='string'?target.sourceId:null,url,label:String(target.label||safeLabel(url)).slice(0,200),createdAt:target.createdAt||new Date().toISOString(),active:false,inFlight:false,timer:null,controller:null,nextAt:null,lastIssue:null,blocked:false,brain:null,currentReceiptId:null,run:0}];
      }catch{return [];}
    });
    state.sessionStarted=saved.session_started_at||state.sessionStarted;
    state.selectedId=state.events[0]?.id||null;
    state.restored=!!(state.events.length||state.targets.length);state.followLatest=!state.restored;
    state.events.forEach(event=>state.seen.add(event.id));
  }catch{state.storageAvailable=false;}
}

function cancelWork(target) {
  target.run++;
  clearTimeout(target.timer);target.timer=null;target.nextAt=null;
  target.controller?.abort();
}
function pause(target, message=false) {
  target.active=false;cancelWork(target);
  if(message)notify(`Paused ${target.label}. No further checks are scheduled.`);
  persist();renderAll();
}
function resume(target) {
  if(target.blocked){formMessage(target.lastIssue||'Remove this target and connect a supported public endpoint.');return;}
  state.followLatest=true;
  target.active=true;target.lastIssue=null;target.currentReceiptId=null;target.brain=newBrain();target.run++;
  if(!target.inFlight&&visible())schedule(target,0);
  persist();renderAll();
}
function removeTarget(target) {
  target.active=false;cancelWork(target);state.targets=state.targets.filter(item=>item!==target);
  if(state.sourceFilter===target.sourceId)state.sourceFilter=null;
  persist();renderAll();notify('Target removed. Its actual observations remain in this session’s record.');
}
function schedule(target, delay=MIN_INTERVAL) {
  clearTimeout(target.timer);target.timer=null;target.nextAt=null;
  if(!connected(target)||!visible()||target.inFlight)return;
  const wait=Math.max(0,delay);target.nextAt=Date.now()+wait;
  target.timer=setTimeout(()=>{target.timer=null;sampleTarget(target);},wait);
}
async function sampleTarget(target) {
  if(!connected(target)||!visible()||target.inFlight)return;
  const run=target.run;
  target.inFlight=true;target.nextAt=null;target.controller=new AbortController();
  let watchdogExpired=false;
  const watchdog=setTimeout(()=>{watchdogExpired=true;target.controller?.abort();},18000);
  const clientStarted=new Date().toISOString();
  let retryDelay=Math.max(MIN_INTERVAL,state.capabilities?.interval_ms||0);
  renderCards();renderStatus();
  try {
    const response=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},credentials:'same-origin',cache:'no-store',body:JSON.stringify({target:target.url,...(state.capabilities?.target_probes?.connected?{page_check:$('probe-page').checked,follow_redirects:$('probe-redirects').checked,...($('probe-get').checked?{get_fallback:true}:{})}:{})}),signal:target.controller.signal});
    let payload;try{payload=await response.json();}catch{throw new Error('The collector returned an unreadable response. No observation was added.');}
    const clientReceived=new Date().toISOString();
    if(!connected(target)||target.run!==run)return;
    if(response.status===429) {
      const headerSeconds=Number(response.headers.get('Retry-After'));
      retryDelay=Math.max(MIN_INTERVAL,finite(payload?.retry_after_ms)||0,Number.isFinite(headerSeconds)?headerSeconds*1000:0);
      target.lastIssue='The collector asked us to wait. The next check will respect its retry window.';
      return;
    }
    if([400,401,403,422].includes(response.status)) {
      target.active=false;target.blocked=true;
      const code=typeof payload?.error==='object'?payload.error.code:payload?.code||(typeof payload?.error==='string'?payload.error:null);
      if(code==='platform_ip_unsupported')target.lastIssue='Direct IP monitoring needs the local collector. Run python apps/sentinel-observatory/live/local_server.py, then open http://localhost:8766.';
      else if(response.status===401||response.status===403)target.lastIssue='Access was denied. Reopen this private observatory with the authorized account before reconnecting.';
      else target.lastIssue=readableError(payload,'This target was not accepted. No monitoring connection was created.');
      formMessage(target.lastIssue);return;
    }
    if(!response.ok){target.lastIssue=readableError(payload,`Collector request failed with HTTP ${response.status}. No observation was added.`);return;}
    const event=payload?.event;
    if(!event||typeof event.id!=='string'||typeof event.source_id!=='string'||!Number.isFinite(stamp(event.observed_at))||!Number.isFinite(stamp(event.completed_at)))throw new Error('The collector returned an invalid measurement. Nothing was added to the record.');
    if(state.events.some(previous=>previous.id===event.id)){target.lastIssue='A duplicate observation was ignored. Waiting for fresh evidence.';return;}
    // These browser timestamps belong to THIS request, never to replay or history.
    event.client_request_started_at=clientStarted;
    event.client_received_at=clientReceived;
    const metadata=payload.target||{};
    const duplicate=state.targets.find(item=>item!==target&&item.sourceId===event.source_id);
    if(duplicate){target.active=false;target.blocked=true;target.lastIssue='This endpoint is already connected. Remove this duplicate target.';}
    target.sourceId=event.source_id;target.currentReceiptId=event.id;
    if(typeof metadata.url==='string')target.url=metadata.url;
    target.label=String(metadata.label||event.target||safeLabel(target.url)).slice(0,200);
    if(!duplicate)target.lastIssue=null;
    try { if(!hydrateTargetAnalysis(event))event.brain=(target.brain||=newBrain()).observe(event); }
    catch { event.brain={state:'unavailable',reason:'The numerical analysis could not process this observation. Original evidence is preserved.',signals:[],forecast:{issued:null,comparison:null}}; }
    event.analysis=analyzeEvidence(event,[...state.events].reverse());
    state.events.unshift(event);state.events=state.events.slice(0,MAX_EVENTS);
    if(!state.selectedId||!state.events.some(previous=>previous.id===state.selectedId))state.selectedId=event.id;
    if(state.followLatest&&(!state.sourceFilter||state.sourceFilter===target.sourceId))state.selectedId=event.id;
    retryDelay=Math.max(retryDelay,finite(payload.retry_after_ms)||0);
    persist();renderAll();if(event.target_storage==='retained'&&!target.historyLoaded)restoreTargetHistory(target);
  } catch(error) {
    if(connected(target)&&target.run===run)target.lastIssue=watchdogExpired?'The collector did not answer within 18 seconds. No measurement was invented; a later check will try again.':error.name==='AbortError'?'The local request was cancelled. No result was received.':String(error.message||'The collector is unreachable. Retained evidence is unchanged.').slice(0,400);
  } finally {
    clearTimeout(watchdog);target.inFlight=false;target.controller=null;
    if(connected(target)&&visible())schedule(target,target.run===run?retryDelay:MIN_INTERVAL);
    persist();renderAll();
  }
}

function hydrateTargetAnalysis(event) {
  const a=event.target_analysis;if(!a)return false;
  event.brain={...a,signals:(a.signals||[]).filter(s=>finite(s.value)!==null),baseline_scope:a.baseline_scope,calibration_note:a.calibration_reset||'Separate retained metric streams; current-frame change detection.',session_epoch:'server-'+a.generation,event_active:!!a.episode,forecast:{issued:null,comparison:null}};return true;
}
async function restoreTargetHistory(target) {
  if(!target.sourceId||target.historyLoaded||!state.capabilities?.target_probes?.connected)return;
  target.historyLoaded=true;
  try {
    const response=await fetch('/api/target-history?target_id='+encodeURIComponent(target.sourceId),{credentials:'same-origin',cache:'no-store',signal:AbortSignal.timeout(6000)});
    if(!response.ok)throw new Error('history_unavailable');const payload=await response.json();
    if(!payload.available||!Array.isArray(payload.events))throw new Error('history_unavailable');
    const known=new Set(state.events.map(e=>e.id));
    for(const event of payload.events.slice(0,128))if(event.source_id===target.sourceId&&/^evt_[a-f0-9]{32}$/.test(event.id||'')&&!known.has(event.id)){hydrateTargetAnalysis(event);event.restored_from_server=true;state.events.push(event);known.add(event.id);}
    state.events.sort((a,b)=>stamp(b.completed_at)-stamp(a.completed_at));state.events=state.events.slice(0,MAX_EVENTS);persist();renderAll();
  }catch{target.historyLoaded=false;}
}
function changeIds() {
  const previous=new Map(),changed=new Set();
  for(const event of [...state.events].reverse()){
    const signature=event.error?.code||event.status;
    if(previous.has(event.source_id)&&previous.get(event.source_id)!==signature)changed.add(event.id);
    previous.set(event.source_id,signature);
  }
  return changed;
}
function freshness(target) {
  const event=latest(target.sourceId);
  const received=stamp(event?.client_received_at||event?.completed_at);
  const age=Number.isFinite(received)?Math.max(0,Date.now()-received):Infinity;
  const limit=Math.max(45000,(state.capabilities?.interval_ms||MIN_INTERVAL)*3);
  return {event,age,fresh:!!event&&target.currentReceiptId===event.id&&age<=limit};
}
function renderStatus() {
  const active=state.targets.filter(target=>target.active);
  const badge=$('global-state');badge.className='mode-badge';
  if(active.length&&!visible()){badge.textContent='TAB HIDDEN · PAUSED';badge.classList.add('paused');}
  else if(active.length){
    const fresh=active.filter(target=>!target.lastIssue&&freshness(target).fresh).length;
    const established=active.some(target=>!!target.currentReceiptId);
    const allIssues=active.every(target=>!!target.lastIssue);
    if(allIssues){badge.textContent='RETRYING · COLLECTOR ISSUE';badge.classList.add('paused');}
    else if(fresh){badge.textContent=`LIVE · ${fresh}/${active.length} FRESH`;badge.classList.add('live');}
    else if(established){badge.textContent='STALE · AWAITING EVIDENCE';badge.classList.add('paused');}
    else badge.textContent='CONNECTING · NO DATA YET';
  }
  else if(state.targets.length||state.events.length){badge.textContent='PAUSED · RECORDS RETAINED';badge.classList.add('paused');}
  else badge.textContent='READY TO CONNECT';
  $('inspector-kind').textContent=state.followLatest?'Following latest':'Follow latest';
  $('inspector-kind').setAttribute('aria-pressed',String(state.followLatest));$('inspector-kind').disabled=!state.events.length;
  $('pause-all').hidden=!active.length;
  $('connect-button').disabled=state.targets.length>=MAX_TARGETS;
  $('metric-active').innerHTML=`${visible()?active.length:0}<small>/ ${MAX_TARGETS}</small>`;
  $('metric-active-note').textContent=active.length&&!visible()?'Foreground checks paused':active.length?'Only while this tab is visible':state.targets.length?'Resume a target to observe':'Connect your first target';
  const last=state.events[0];
  $('sample-clock').textContent=last?`Latest receipt ${clock(last.client_received_at||last.completed_at)}`:active.length?'Waiting for a real response':'Awaiting connection';
  let notice=state.sessionNotice;
  if(!visible()&&active.length)notice='This tab is hidden. Further checks are paused and resume only when it becomes visible.';
  else if(state.restored)notice='Saved observations restored. Every target is paused; choose Resume to start a new live calibration.';
  else if(!state.storageAvailable)notice='Session storage is unavailable. These observations remain in memory; export JSON before closing this tab.';
  $('session-notice').textContent=notice||'';$('session-notice').hidden=!notice;
}
function targetState(target) {
  if(target.blocked)return {label:'Connection not available',style:'retry'};
  if(!target.active)return {label:'Paused · last evidence retained',style:''};
  if(!visible())return {label:'Paused while tab is hidden',style:''};
  if(target.lastIssue)return {label:target.inFlight?'Retrying · last evidence retained':'Collector issue · last evidence retained',style:'retry'};
  const receipt=freshness(target);
  if(!receipt.event||!target.currentReceiptId)return {label:receipt.event?'Connecting · previous evidence retained':target.inFlight?'Connecting · no measurement yet':'Waiting for first response',style:''};
  if(!receipt.fresh)return {label:target.inFlight?'Refreshing stale evidence':'Stale · last receipt '+Math.floor(receipt.age/1000)+'s ago',style:'retry'};
  return {label:target.inFlight?'Checking · last receipt '+Math.floor(receipt.age/1000)+'s ago':'Fresh · received '+Math.floor(receipt.age/1000)+'s ago',style:'active'};
}
function routeLabel(event) {
  const transport=event?.coverage?.transport;
  if(transport?.kind==='named_service_adapter')return 'Named service · DNS route, not a pinned IP';
  if(transport?.kind==='authenticated_pinned_ip_relay')return 'Direct public IP · address pinned';
  if(transport?.kind==='authenticated_pinned_target_relay')return 'Public endpoint · address pinned';
  if(event?.coverage?.dns?.pinned_address||event?.coverage?.dns?.address_pinned||event?.coverage?.address_pinned)return 'Address pinned by local collector';
  return event?'Public HTTP/S · collector view':'';
}
function routeMarkup(event) {
  const transport=event?.coverage?.transport;if(!transport)return '';
  return `<div class="evidence-block"><h3>Observed connection path</h3><p>${esc(routeLabel(event))}</p>${transport.requested_url?`<p class="evidence-hint">Requested: ${esc(transport.requested_url)}</p>`:''}${transport.effective_url?`<p class="evidence-hint">Measured: ${esc(transport.effective_url)}</p>`:''}${transport.limitation?`<p class="evidence-hint">${esc(transport.limitation)}</p>`:''}</div>`;
}
function networkMarkup(event) {
  const network=event?.coverage?.network;
  if(!network||network.scope!=='observatory_outbound_head_connection')return '';
  const yesNo=value=>value===true?'yes':value===false?'no':'not measured';
  const ipPort=(address,port)=>!address?'not measured':port?`${String(address).includes(':')?'['+address+']':address}:${port}`:address;
  const entries=[['Destination',ipPort(network.destination_ip,network.destination_port)],['Transport',network.protocol||'not measured'],['Application protocol',network.application_protocol||'not measured'],['Public peer',ipPort(network.peer_ip,network.peer_port)],['Socket connected',yesNo(network.socket_connected)],['Peer matched destination',yesNo(network.peer_matches_destination)],['TLS identity verified',network.application_protocol==='HTTP'?'not applicable':network.tls_verified===false?'verification failed':yesNo(network.tls_verified)],['TLS protocol',network.application_protocol==='HTTP'?'not applicable':network.tls_protocol||'not measured'],['Socket bytes written',finite(network.bytes_written)!==null?formatNumber(network.bytes_written):'not measured'],['Socket bytes read',finite(network.bytes_read)!==null?formatNumber(network.bytes_read):'not measured']];
  return `<div class="evidence-block"><h3>This collector connection</h3><dl class="security-grid">${entries.map(([key,value])=>`<div><dt>${esc(key)}</dt><dd>${esc(value)}</dd></div>`).join('')}</dl><p class="evidence-hint">${esc(network.counter_scope||'Application socket counters, not packet counts or interface traffic.')}</p><p class="evidence-hint">${esc(network.limitation||'Only this Observatory connection is visible. Target visitor traffic and internal networks remain outside this collector.')}</p></div>`;
}
function renderCards() {
  const activeId=document.activeElement?.dataset?.action;
  const activeTarget=document.activeElement?.dataset?.target;
  $('target-list').innerHTML=state.targets.map(target=>{
    const event=latest(target.sourceId),status=targetState(target);
    const countdown=target.nextAt?Math.max(0,Math.ceil((target.nextAt-Date.now())/1000)):null;
    return `<article class="target-card ${state.sourceFilter===target.sourceId&&target.sourceId?'selected':''}" data-card="${esc(target.id)}"><div class="target-card-top"><button class="target-name" data-action="select" data-target="${esc(target.id)}" title="${esc(target.url)}">${esc(target.label)}</button><button class="target-remove" data-action="remove" data-target="${esc(target.id)}" aria-label="Remove ${esc(target.label)}">×</button></div><div class="target-status ${status.style}" data-status="${esc(target.id)}"><i></i><span>${esc(status.label)}</span></div><div class="target-stats"><span style="color:${event?statusColor(event):'var(--faint)'}">${esc(responseText(event))}</span><span>${formatNumber(event?.latency_ms)} <small>ms</small></span></div>${event?`<p class="target-transport">${esc(routeLabel(event))}</p>`:''}${target.lastIssue?`<p class="target-issue">${esc(target.lastIssue)}</p>`:''}<div class="target-card-bottom"><span class="target-next" data-countdown="${esc(target.id)}">${target.inFlight?'Request in progress':countdown!==null?'Next check in '+countdown+'s':event?'Last '+clock(event.completed_at):'No observation yet'}</span><button class="target-toggle" data-action="toggle" data-target="${esc(target.id)}" ${target.blocked?'disabled':''} aria-label="${target.active?'Pause':'Resume'} ${esc(target.label)}">${target.active?'Pause':'Resume'}</button></div></article>`;
  }).join('');
  $('target-list').querySelectorAll('[data-action]').forEach(button=>button.addEventListener('click',()=>{
    const target=state.targets.find(item=>item.id===button.dataset.target);if(!target)return;
    if(button.dataset.action==='remove')removeTarget(target);
    else if(button.dataset.action==='toggle'){state.restored=false;target.active?pause(target,true):resume(target);}
    else if(target.sourceId)selectSource(target.sourceId);
  }));
  if(activeId&&activeTarget)[...$('target-list').querySelectorAll('[data-action]')].find(button=>button.dataset.action===activeId&&button.dataset.target===activeTarget)?.focus({preventScroll:true});
}
function fieldFreshnessSignature() {return state.targets.map(target=>[target.id,target.sourceId,target.active,visible(),!!target.lastIssue,freshness(target).fresh].join(':')).join('|');}
function tick() {
  renderStatus();
  for(const target of state.targets){
    const status=targetState(target),statusElement=[...$('target-list').querySelectorAll('[data-status]')].find(item=>item.dataset.status===target.id);
    if(statusElement){statusElement.className='target-status '+status.style;statusElement.querySelector('span').textContent=status.label;}
    const el=[...$('target-list').querySelectorAll('[data-countdown]')].find(item=>item.dataset.countdown===target.id);
    if(el&&target.nextAt)el.textContent='Next check in '+Math.max(0,Math.ceil((target.nextAt-Date.now())/1000))+'s';
  }
  if(fieldFreshnessState!==fieldFreshnessSignature())renderField();
}
function renderMetrics() {
  const last=state.events[0],first=state.events.at(-1);
  $('metric-count').textContent=state.events.length.toLocaleString();
  $('metric-window').textContent=first?`${dateTime(first.observed_at)} → ${clock(last.completed_at)}`:'No measurements yet';
  $('metric-latency').innerHTML=`${formatNumber(last?.latency_ms)}<small>ms</small>`;
  $('metric-latency-note').textContent=last?(isIncomplete(last)?'Incomplete request · ':labelFor(last)+' · ')+'received '+clock(last.client_received_at||last.completed_at):'Collector round trip';
  $('metric-changes').textContent=changeIds().size;
  $('export-button').disabled=!state.events.length;
}

const positions=[{x:685,y:140,tx:735,ty:137,anchor:'start',path:'M386 221 C480 172 566 98 685 140'},{x:805,y:333,tx:755,ty:330,anchor:'end',path:'M394 252 C551 218 733 235 805 333'},{x:535,y:454,tx:585,ty:451,anchor:'start',path:'M365 286 C411 343 415 454 535 454'},{x:160,y:374,tx:211,ty:371,anchor:'start',path:'M304 268 C245 290 184 324 160 374'}];
function renderField() {
  fieldFreshnessState=fieldFreshnessSignature();
  const sources=state.targets.filter(target=>target.sourceId&&sourceEvents(target.sourceId).length);
  $('field-empty').hidden=!!sources.length;
  document.querySelector('.field-wrap').classList.toggle('has-evidence',!!sources.length);
  $('field-pan-hint').hidden=!sources.length||window.innerWidth>700;
  if(!sources.length){$('evidence-field').innerHTML='';return;}
  const changes=changeIds();
  const parts=['<desc id="field-description">One HTTP collector linked to selected public endpoints. Each small dot is an actual retained observation, up to 48 per target. Curves represent measurement relationships, not traffic or internal topology.</desc><defs><linearGradient id="measured-link"><stop stop-color="#8acbde" stop-opacity=".14"/><stop offset=".55" stop-color="#8acbde" stop-opacity=".62"/><stop offset="1" stop-color="#b4eedc" stop-opacity=".3"/></linearGradient></defs><g aria-hidden="true"><ellipse cx="455" cy="280" rx="284" ry="189" fill="none" stroke="#1b2d3a" stroke-width=".7" transform="rotate(14 455 280)"/><ellipse cx="455" cy="280" rx="375" ry="244" fill="none" stroke="#142532" stroke-width=".7" transform="rotate(14 455 280)"/><path d="M58 275H938M455 37V526" stroke="#1a2b39" stroke-width=".6" stroke-dasharray="2 9"/></g>'];
  sources.forEach((target,index)=>{
    const position=positions[index],events=sourceEvents(target.sourceId).slice(0,48),last=events[0];
    const selected=state.sourceFilter===target.sourceId,dim=state.sourceFilter&&!selected;
    const active=target.active&&visible()&&!target.lastIssue&&freshness(target).fresh;
    const color=active?statusColor(last,changes):'#758a99';
    const label=target.label.length>25?target.label.slice(0,23)+'…':target.label;
    parts.push(`<g class="${dim?'dimmed':''}"><path d="${position.path}" fill="none" stroke="${color}" stroke-width="6" opacity=".04" aria-hidden="true"/><path d="${position.path}" fill="none" stroke="url(#measured-link)" stroke-width="1.2" aria-hidden="true"/><g role="button" tabindex="0" class="source-node" data-source="${esc(target.sourceId)}" aria-pressed="${selected}" aria-label="Select ${esc(target.label)}. Latest ${esc(responseText(last))}. ${events.length} visible observations."><circle class="node-hit" cx="${position.x}" cy="${position.y}" r="48"/><circle cx="${position.x}" cy="${position.y}" r="35" fill="#0d1924" stroke="${selected?'#608a7e':'#2d4350'}" stroke-width=".8"/><circle cx="${position.x}" cy="${position.y}" r="16" fill="none" stroke="${color}" opacity=".35" stroke-width=".7"/><circle class="node-core" cx="${position.x}" cy="${position.y}" r="${selected?7:5}" fill="${color}"/><text class="node-title" x="${position.tx}" y="${position.ty}" text-anchor="${position.anchor}">${esc(label)}</text><text class="node-number" x="${position.tx}" y="${position.ty+19}" text-anchor="${position.anchor}">${esc(responseText(last))} · ${formatNumber(last.latency_ms)} ms</text><text class="node-note" x="${position.tx}" y="${position.ty+36}" text-anchor="${position.anchor}">${sourceEvents(target.sourceId).length} observations${active?'':' · last known'}</text></g>`);
    [...events].reverse().forEach((event,number)=>{
      const radius=35+Math.floor(number/24)*8,angle=(number%24)/24*Math.PI*2-Math.PI/2;
      const x=position.x+Math.cos(angle)*radius,y=position.y+Math.sin(angle)*radius;
      const fresh=!state.seen.has(event.id)&&state.fieldRendered;
      parts.push(`<circle class="event-dot" role="button" tabindex="0" data-event="${esc(event.id)}" cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${event.id===state.selectedId?4.5:2.5}" fill="${statusColor(event,changes)}" ${event.id===state.selectedId?'stroke="#fff" stroke-width="1"':''} aria-label="Inspect ${esc(target.label)}, ${esc(responseText(event))}, ${clock(event.completed_at)}"><title>${esc(target.label)} · ${esc(responseText(event))} · ${formatNumber(event.latency_ms)} ms · ${clock(event.completed_at)}</title></circle>`);
      if(fresh)parts.push(`<circle class="birth-ring" cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="5" aria-hidden="true"/>`);
    });parts.push('</g>');
  });
  parts.push('<g aria-hidden="true"><circle cx="345" cy="238" r="55" fill="#10202c" stroke="#3e5b66" stroke-width="1"/><circle cx="345" cy="238" r="48" fill="none" stroke="#203745" stroke-width=".7"/><path d="M333 222a17 17 0 1 0 23 26M343 226a10 10 0 1 0 8 17" fill="none" stroke="#b4eedc" stroke-width="1.2"/><circle cx="347" cy="234" r="2" fill="#b4eedc"/><text x="345" y="321" text-anchor="middle" fill="#eaf0f1" font-family="sans-serif" font-size="16">Your observer</text><text x="345" y="340" text-anchor="middle" fill="#9eacb8" font-family="monospace" font-size="9">PUBLIC HTTP/S · READ ONLY</text><text x="25" y="541" fill="#758a99" font-family="monospace" font-size="8" letter-spacing="1">MEASURED RELATIONSHIPS · NOT VISITOR TRAFFIC</text></g>');
  const focusedEvent=document.activeElement?.dataset?.event,focusedSource=document.activeElement?.dataset?.source;
  $('evidence-field').innerHTML=parts.join('');
  $('evidence-field').querySelectorAll('[data-source]').forEach(element=>activate(element,()=>selectSource(element.dataset.source)));
  $('evidence-field').querySelectorAll('[data-event]').forEach(element=>activate(element,()=>selectEvent(element.dataset.event)));
  if(focusedEvent||focusedSource)[...$('evidence-field').querySelectorAll('[data-event],[data-source]')].find(element=>focusedEvent&&element.dataset.event===focusedEvent||focusedSource&&element.dataset.source===focusedSource)?.focus({preventScroll:true});
  state.seen=new Set(state.events.map(event=>event.id));state.fieldRendered=true;
}
function activate(element,action) {element.addEventListener('click',action);element.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();action();}});}
function selectSource(id) {state.followLatest=true;renderStatus();state.sourceFilter=state.sourceFilter===id?null:id;state.limit=8;if(state.sourceFilter)state.selectedId=latest(id)?.id||state.selectedId;renderCards();renderField();renderInspector();renderRecord();renderInterpreter();}
function selectEvent(id) {state.followLatest=false;renderStatus();state.selectedId=id;renderField();renderInspector();renderRecord();renderInterpreter();if(window.innerWidth<641)document.querySelector('.inspector').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});}
function evidenceLink(id,label='Inspect cited observation') {
  if(state.events.some(event=>event.id===id))return `<button class="evidence-link" data-evidence="${esc(id)}" title="${esc(id)}">${esc(label)} · ${esc(String(id).slice(0,16))}…</button>`;
  const sensor=list(networkPanel?.getEvidence?.()?.events).find(event=>event.id===id||event.source_event_id===id||event.fields?.linked_evidence_id===id);
  if(sensor)return `<button class="evidence-link" data-sensor-evidence="${esc(sensor.id)}" title="${esc(id)}">${esc(label)} · ${esc(String(id).slice(0,16))}…</button>`;
  return `<span class="evidence-hint">Cited evidence outside the retained record: ${esc(id)}</span>`;
}
function observedText(event) {
  if(event.probe_version==='target-probes-v1'){const explanation=probeExplanation(event);return explanation.summary+' Next check: '+explanation.next_check;}
  if(event.error)return `${event.error.message||'The collector could not complete the observation.'}${finite(event.status)!==null?` HTTP ${event.status} was received before completion stopped.`:''}`;
  return `The collector received HTTP ${responseText(event)} from this endpoint${finite(event.latency_ms)!==null?' after '+formatNumber(event.latency_ms)+' ms elapsed':''}.${event.response_chain_complete===false?' The response chain is incomplete.':''}${event.redirect||[301,302,303,307,308].includes(event.status)?(event.coverage?.redirect_followed?' The optional probe followed a bounded public redirect chain; first-hop timings remain separate.':' This is a redirect response; the collector did not follow it.'):''}`;
}
function hypothesesFor(event) {
  const next='Compare subsequent checks and an independent authorized vantage before attributing a cause.';
  if(event.probe_version==='target-probes-v1'&&(event.error||[405,501].includes(event.status))){const diagnosis=probeExplanation(event);return [{text:diagnosis.summary,next:diagnosis.next_check}];}
  if(isIncomplete(event))return [{text:'Collector connectivity, DNS, destination delay, and filtering are competing possibilities. This measurement does not distinguish them.',next}];
  if(event.status===429)return [{text:'The endpoint may be applying a request limit. Its scope and reason are not established by this response alone.',next:'Review the endpoint’s authorized server logs and retry policy before increasing any sampling.'}];
  if([401,403].includes(event.status))return [{text:'An access policy may be rejecting this unauthenticated HEAD request. That does not establish whether a signed-in browser is affected.',next:'Compare authorized application access and server logs. This collector does not send credentials.'}];
  if(event.status===404)return [{text:'This path may be absent or intentionally hidden by routing or access rules. HTTP 404 alone does not identify which.',next:'Confirm the intended public path and compare an authorized application request.'}];
  if(event.status>=500)return [{text:'An origin or intermediary returned a server-error response. This observation alone cannot attribute the fault.',next:'Compare later responses with authorized origin and edge logs.'}];
  if(event.target_analysis){
    if(list(event.target_analysis.signals).some(signal=>signal.state==='unusual'))return [{text:'One or more measured connection stages departed from their provisional baseline and passed the measured-deviation gate. The cause remains unverified.',next}];
    if(list(event.target_analysis.changes).length)return [{text:'The recorded endpoint metadata changed. DNS rotation, certificate renewal and dynamic content can produce expected changes.',next}];
    return [];
  }
  if(event.brain?.event_active)return [{text:'The measured latency behavior differs from its provisional baseline. Network-path variation and application-side variation remain competing explanations.',next}];
  return [];
}
function securityMarkup(headers) {
  const keys=[['hsts_present','HSTS'],['csp_present','CSP'],['x_content_type_options','Content type'],['frame_options','Frame policy'],['referrer_policy','Referrer policy'],['cache_private','Private cache'],['cache_no_store','No-store cache']];
  return '<dl class="security-grid">'+keys.map(([key,label])=>{const value=headers?.[key];return `<div><dt>${label}</dt><dd>${value===null||value===undefined?'not captured':value===true?'present':value===false?'absent':esc(value)}</dd></div>`;}).join('')+'</dl>';
}
function renderInspector() {
  renderTargetPanel($('target-probes'),{event:selectedEvent()||state.events[0],events:state.events,capabilities:state.capabilities});
  const event=selectedEvent();if(!event)return;
  const signature=JSON.stringify(event);if(signature===state.inspectorSignature)return;
  const wasOpen=document.querySelector('.technical-details')?.open,previousId=$('inspector-content').dataset.event;
  state.inspectorSignature=signature;$('inspector-content').dataset.event=event.id;
  const brain=event.brain||{},hypotheses=hypothesesFor(event),issued=brain.forecast?.issued;
  $('inspector-kind').textContent=state.followLatest?'Following latest':'Follow latest';
  $('inspector-content').innerHTML=`<h2 id="inspector-title">${esc(labelFor(event))}</h2><p class="evidence-url">${esc(event.url||'URL not supplied')}</p><div class="evidence-readings"><div><span>HTTP status</span><strong class="${isIncomplete(event)?'failed':event.status>=400?'warning':'good'}">${esc(finite(event.status)===null?'No response':String(event.status))}</strong></div><div><span>Probe elapsed</span><strong>${formatNumber(event.probe_version==='target-probes-v1'?probeExplanation(event).elapsed_ms:event.latency_ms)}<small>ms</small></strong></div></div><p class="evidence-timestamp">${esc(dateTime(event.completed_at))} · collector clock<br>Received ${esc(dateTime(event.client_received_at))} · browser clock</p><div class="evidence-block"><h3>What was observed</h3><p>${esc(observedText(event))}</p><p class="evidence-hint">Elapsed time includes this collector’s network path. It is not page-load time or origin processing time.</p>${evidenceLink(event.id,'Original evidence')}</div>${routeMarkup(event)}${networkMarkup(event)}<div class="evidence-block"><h3>Observed response controls</h3>${securityMarkup(event.security_headers)}<p class="evidence-hint">Header presence is not a vulnerability grade. Absent does not mean exploited.</p></div><div class="evidence-block"><h3>Brain / Sentinel <span class="evidence-hint">· experimental</span></h3><span class="signal-state">${esc((brain.state||'waiting').replace(/_/g,' '))}</span>${finite(brain.warmup_remaining)>0?`<p>${brain.warmup_remaining} valid responses remain before provisional calibration.</p>`:''}${brain.reason?`<p>${esc(brain.reason)}</p>`:''}${list(brain.signals).length?'<ul class="signal-list">'+brain.signals.map(signal=>`<li>${esc(signal.label||signal.key)}<strong>${finite(signal.value)!==null?signal.value.toFixed(2):'—'}</strong><small>${esc(signal.meaning)}</small></li>`).join('')+'</ul>':''}${brain.baseline_scope?`<p class="evidence-hint">Baseline scope: ${esc(brain.baseline_scope)}</p>`:''}${brain.calibration_note?`<p class="evidence-hint">${esc(brain.calibration_note)}</p>`:''}<p class="evidence-hint">Portable numerical analysis. Scores are not attack probabilities.</p>${issued?`<p class="evidence-hint">Issued after receiving this evidence: next-response EWMA estimate ${formatNumber(issued.predicted_latency_ms)} ms; persistence ${formatNumber(issued.persistence_latency_ms)} ms. This is not an incident prediction.</p>`:''}</div><div class="evidence-block"><h3>Possible explanations</h3><p class="evidence-hint">Deterministic, conditional notes · no model invoked</p>${hypotheses.length?hypotheses.map(item=>`<div class="hypothesis"><small>Possibility · cause unverified</small><p>${esc(item.text)}</p><p class="next">Next observation: ${esc(item.next)}</p>${evidenceLink(event.id,'Supporting measurement')}</div>`).join(''):'<p>No specific cause is established or needed by this observation. Watch the sequence before assigning meaning to a single response.</p>'}</div><div class="evidence-block"><h3>Still unknown</h3><p>Visitor impact, internal behavior, origin processing, authenticated application state, and the cause of any change remain outside this view.</p></div><details class="technical-details"><summary>Original measurement & provenance</summary><dl><dt>Observation ID</dt><dd>${esc(event.id)}</dd><dt>Request started · collector UTC</dt><dd>${esc(event.observed_at)}</dd><dt>Response completed · collector UTC</dt><dd>${esc(event.completed_at)}</dd><dt>Request started · browser UTC</dt><dd>${esc(event.client_request_started_at||'Not retained')}</dd><dt>Receipt · browser UTC</dt><dd>${esc(event.client_received_at||'Not retained')}</dd><dt>Evidence hash · collector payload</dt><dd>${esc(event.evidence_hash||'Not supplied by this collector; exported JSON is an evidence record, not a signature.')}</dd>${event.evidence_hash?'<dd>Browser timing, Brain signals, and interpretation are added after collection and are outside this collector hash.</dd>':''}<dt>Analysis epoch</dt><dd>${esc(brain.session_epoch||'Not supplied')}</dd><dt>Coverage / vantage</dt><dd>${esc(textValue(event.coverage))}</dd>${Object.entries(event.headers||{}).map(([key,value])=>`<dt>${esc(key)}${key.toLowerCase()==='date'?' · upstream clock':''}</dt><dd>${esc(textValue(value))}</dd>`).join('')}${event.error?`<dt>Original collection error</dt><dd>${esc(textValue(event.error))}</dd>`:''}</dl><button class="quiet-button" id="copy-evidence">Copy this evidence JSON</button></details>`;
  if(event.interpretation){const result=document.createElement('div');result.className='model-inspector-result interpretation-result';result.innerHTML=interpretationMarkup(event,true);$('inspector-content').insertBefore(result,document.querySelector('.technical-details'));}
  if(wasOpen&&previousId===event.id)document.querySelector('.technical-details').open=true;
  bindInterpretationEvidence($('inspector-content'));
  $('copy-evidence').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(JSON.stringify(event,null,2));notify('Original evidence copied.');}catch{notify('Clipboard unavailable. Export the session JSON instead.');}});
}

function filteredEvents() {
  const changes=changeIds();
  return state.events.filter(event=>(!state.sourceFilter||event.source_id===state.sourceFilter)&&(state.filter==='all'||state.filter==='changes'&&changes.has(event.id)||state.filter==='incomplete'&&isIncomplete(event)));
}
function renderRecord() {
  const events=filteredEvents(),changes=changeIds();
  $('view-all').hidden=!state.sourceFilter;
  $('record-summary').textContent=`${state.events.length} retained observations${state.events.length===MAX_EVENTS?' · oldest records leave the 500-event window':''}. ${state.sourceFilter?'Showing one target.':'Pause or remove a target without losing its recorded evidence.'}`;
  $('event-list').innerHTML=events.length?events.slice(0,state.limit).map(event=>`<button class="record-button ${event.id===state.selectedId?'selected':''}" data-event="${esc(event.id)}" aria-label="Inspect ${esc(labelFor(event))}, ${esc(responseText(event))}, ${clock(event.completed_at)}" ${event.id===state.selectedId?'aria-current="true"':''}><span class="record-name"><i class="dot" style="background:${statusColor(event,changes)}"></i><span>${esc(labelFor(event))}</span><small>${isIncomplete(event)?'incomplete':changes.has(event.id)?'status changed':'HEAD observation'}</small></span><span class="record-status" style="color:${statusColor(event,changes)}">${esc(responseText(event))}</span><span class="record-latency">${formatNumber(event.latency_ms)} ms</span><span class="record-time">${clock(event.completed_at)}</span><span class="record-arrow" aria-hidden="true">↗</span></button>`).join(''):`<p class="empty-record">${state.filter==='changes'?'No response changes in this retained window.':state.filter==='incomplete'?'No incomplete observations in this retained window.':'No observations for this selection yet.'}</p>`;
  $('event-list').querySelectorAll('[data-event]').forEach(button=>button.addEventListener('click',()=>selectEvent(button.dataset.event)));
  $('show-more').hidden=events.length<=state.limit;$('show-more').textContent=`Show more (${Math.max(0,events.length-state.limit)} remaining)`;
  renderTimeline(events,changes);
}
function renderTimeline(events,changes) {
  const values=[...events].reverse().filter(event=>Number.isFinite(stamp(event.completed_at))&&finite(event.latency_ms)!==null);
  if(!values.length){$('timeline').innerHTML='<text x="600" y="59" text-anchor="middle" fill="#758a99" font-family="sans-serif" font-size="12">Actual elapsed-time measurements will appear here</text>';$('timeline').setAttribute('aria-label','No timed observations for this selection');return;}
  const first=Math.min(...values.map(event=>stamp(event.completed_at))),last=Math.max(...values.map(event=>stamp(event.completed_at))),peak=Math.max(1,...values.map(event=>event.latency_ms));
  const x=event=>50+(stamp(event.completed_at)-first)/Math.max(1,last-first)*1138,y=event=>88-event.latency_ms/peak*73;
  const parts=[`<path d="M50 15H1188M50 88H1188" stroke="#203340" stroke-width=".7"/><text x="0" y="19" font-family="monospace" font-size="10" fill="#758a99">${Math.round(peak)} ms</text><text x="0" y="92" font-family="monospace" font-size="10" fill="#758a99">0</text>`];
  const ids=[...new Set(values.map(event=>event.source_id))];
  ids.forEach(id=>{const points=values.filter(event=>event.source_id===id);if(points.length>1)parts.push(`<path d="${points.map((event,index)=>(index?'L':'M')+x(event).toFixed(1)+' '+y(event).toFixed(1)).join(' ')}" fill="none" stroke="#80b2bd" stroke-width="1" opacity=".28"/>`);});
  values.forEach(event=>parts.push(`<circle cx="${x(event).toFixed(1)}" cy="${y(event).toFixed(1)}" r="${event.id===state.selectedId?4.5:2.8}" fill="${statusColor(event,changes)}" ${event.id===state.selectedId?'stroke="#fff" stroke-width="1"':''}><title>${esc(labelFor(event))} · ${clock(event.completed_at)} · ${formatNumber(event.latency_ms)} ms · ${esc(responseText(event))}</title></circle>`));
  parts.push(`<text x="50" y="113" fill="#758a99" font-family="monospace" font-size="9">${clock(first)}</text><text x="1188" y="113" text-anchor="end" fill="#758a99" font-family="monospace" font-size="9">${clock(last)}</text>`);
  $('timeline').innerHTML=parts.join('');$('timeline').setAttribute('aria-label',`${values.length} actual elapsed-time measurements. Longest ${Math.round(peak)} milliseconds. Incomplete requests remain labeled. Inspect details in the record below.`);
  $('timeline-period').textContent=`${values.length} actual points · collector clock`;
}
function renderForecast() {
  const retained=(selectedEvent()||state.events[0])?.target_analysis;
  if(retained){
    const streams=retained.signals||[],ready=streams.filter(s=>['observing','unusual'].includes(s.state)).length;
    $('forecast-title').textContent=ready?`${ready} metric streams have a provisional baseline.`:'Learning each measured stream separately.';
    $('forecast-description').textContent='Server-retained Brain / Sentinel baselines survive a browser refresh. Each metric has its own calibration. These are observed change signals; future incidents are not predicted.';
    $('forecast-metrics').hidden=true;return;
  }
  // Current live epochs only: reload never backfills the numerical engine.
  const summaries=state.targets.flatMap(target=>target.brain?.summary().sources||[]);
  const comparisons=summaries.map(source=>source.forecast).filter(comparison=>finite(comparison?.scored_count)>0);
  const count=comparisons.reduce((sum,comparison)=>sum+comparison.scored_count,0);
  const pending=state.targets.some(target=>latest(target.sourceId)?.brain?.session_epoch===target.brain?.epochId&&latest(target.sourceId)?.brain?.forecast?.issued);
  $('forecast-title').textContent=count?'Predictive skill is not established.':pending?'The first forecast is awaiting its outcome.':'Waiting for enough evidence.';
  $('forecast-description').textContent=count?`${count} forecasts have been scored against later requests in the current live calibration. Lower error is better. This dependent, narrow observation window cannot establish general predictive skill or predict security incidents.`:'At least 12 valid responses per target establish provisional calibration. Forecasts are issued on the browser clock, then scored only against a later browser-started request. Reloading or resuming starts a fresh epoch.';
  $('forecast-metrics').hidden=!count;
  if(count){const ewma=comparisons.reduce((sum,comparison)=>sum+comparison.ewma_mae_ms*comparison.scored_count,0)/count,persistence=comparisons.reduce((sum,comparison)=>sum+comparison.persistence_mae_ms*comparison.scored_count,0)/count;$('forecast-metrics').innerHTML=`<div><span>EWMA mean absolute error</span><strong>${formatNumber(ewma)} <small>ms</small></strong></div><div><span>Persistence baseline error</span><strong>${formatNumber(persistence)} <small>ms</small></strong></div>`;}
}

function activeModelStatus() {
  return modelState.engine==='companion'?modelState.companionStatus||{state:'unavailable',connected:false,available:false,reason:'No configured companion model is available.'}:modelState.hostedStatus||state.capabilities?.hosted_interpretation||{state:'unavailable',connected:false,configured:false,mode:null,model_id:'gpt-6.1-sol',reason:'The server’s Sol configuration is unavailable. Original measurements and deterministic notes remain inspectable.'};
}
function hostedEvidenceId(event) {
  if(event?.source==='observatory_edge'&&/^edge_[0-9a-f]{32}$/i.test(event.source_event_id||''))return event.source_event_id;
  return /^evt_[0-9a-f]{32}$/i.test(event?.id||'')?event.id:null;
}
function canRequestHostedInterpretation() {
  const status=modelState.hostedStatus||state.capabilities?.hosted_interpretation;
  return !!status&&status.mode==='hosted_gpt_sol'&&status.storage_attached===true&&status.state!=='storage_unavailable';
}
function interpretationMarkup(event) {
  const analysis=event?.interpretation;if(!analysis)return '';
  const provenance=analysis.provenance||{},ran=provenance.model_ran===true;
  const accepted=ran&&provenance.output_accepted===true&&provenance.execution_verified!==false&&analysis.kind==='model';
  const label=accepted?'Sol-ranked review notes · causes unverified':provenance.rejection_code==='cached_context_changed'?'Saved context changed · current evidence notes':ran?'Model execution verified · output unavailable':provenance.model_ran===false&&provenance.inference_attempted!==true?'Deterministic evidence notes · model not invoked':'Model execution unverified · evidence fallback';
  const detail=String(analysis.summary||analysis.reason||'The evidence does not establish a confirmed cause.');
  const facts=list(analysis.facts).slice(0,8).map(item=>`<li>${esc(item.text||item.statement||'Recorded fact')}${list(item.evidence_ids).map(id=>evidenceLink(id,'Fact evidence')).join('')}</li>`).join('');
  const hypotheses=list(analysis.hypotheses).slice(0,5).map(item=>`<li><strong>${esc(item.text||item.label||'Possible explanation')}</strong>${item.next_observation||item.next?`<small>Next observation: ${esc(item.next_observation||item.next)}</small>`:''}${list(item.support||item.evidence_ids).map(id=>evidenceLink(id,'Supporting evidence')).join('')}${typeof item.support==='string'?`<small>${esc(item.support)}</small>`:''}</li>`).join('');
  return `<span class="model-result-badge ${accepted?'':'fallback'}">${esc(label)}</span><h4>${esc(labelFor(event))}</h4><p>${esc(detail)}</p>${provenance.reason?`<p class="evidence-hint">Interpreter note: ${esc(provenance.reason)}</p>`:''}${facts?'<ul class="interpretation-facts">'+facts+'</ul>':''}${list(analysis.next_checks).length?'<h5>Next check</h5><ul>'+list(analysis.next_checks).map(item=>'<li>'+esc(item.text)+list(item.evidence_ids).map(id=>evidenceLink(id,'Related evidence')).join('')+'</li>').join('')+'</ul>':''}${hypotheses?'<ul>'+hypotheses+'</ul>':''}${list(analysis.unknowns).length?`<p class="evidence-hint">Still unknown: ${esc(analysis.unknowns.join(' · '))}</p>`:''}${evidenceLink(event.id,'Interpretation evidence')}<p class="model-provenance">${esc(provenance.model||provenance.model_requested||'No accepted model output')} · ${esc(provenance.runtime||provenance.engine||provenance.role||'Deterministic evidence notes')}<br>Evidence collected ${esc(dateTime(event.completed_at))}<br>Interpretation received ${esc(dateTime(event.interpreted_at))}${event.interpretation_cached?' · cached server result':''}${finite(provenance.inference_ms)!==null?' · '+formatNumber(provenance.inference_ms)+' ms inference':''}</p>`;
}
function bindInterpretationEvidence(container) {
  container.querySelectorAll('[data-evidence]').forEach(button=>button.addEventListener('click',()=>selectEvent(button.dataset.evidence)));
  container.querySelectorAll('[data-sensor-evidence]').forEach(button=>button.addEventListener('click',()=>{
    if(networkPanel?.selectEvidence?.(button.dataset.sensorEvidence))$('network-panel').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});
  }));
}
function renderInterpreter() {
  const status=activeModelStatus(),hosted=modelState.engine==='hosted';
  const configured=hosted?status.configured===true:status.available===true;
  const usable=hosted?canRequestHostedInterpretation():status.available===true;
  const event=selectedEvent()||state.events[0],known=hosted?!!hostedEvidenceId(event):!!event;
  $('model-shortcut').textContent=modelState.busy?'Interpretation running ↗':hosted?configured?'Sol configured ↗':'Sol configuration ↗':'Companion interpretation ↗';
  $('model-badge').textContent=modelState.busy?'Interpreting selected evidence':hosted?status.state==='storage_unavailable'||status.storage_attached===false?'Evidence storage unavailable':status.connected===true&&configured?'Sol configured · manual':'Sol not configured':status.connected?'Companion model loaded':configured?'Companion configured':'Companion unavailable';
  $('model-title').textContent=hosted?'Interpret the evidence with Sol.':status.name||status.model||'Use your configured companion.';
  $('model-description').textContent=hosted?'Measurement rules explain the selected endpoint and its failure stage. Sol ranks eligible review notes; it does not independently diagnose the cause. Facts cite retained evidence, and calls start only when you choose Interpret.':'Uses the model configured on your local collector. This optional path does not download a model in your browser.';
  $('model-status').textContent=modelState.issue||status.reason||status.message||(configured?'Ready for a manual interpretation request.':'Sol is not configured; original observations and deterministic notes remain available.');
  $('model-status').classList.toggle('issue',!!modelState.issue||!configured);
  $('model-engine').disabled=modelState.busy;
  $('interpret-latest').disabled=modelState.busy||!usable||!known;
  $('interpret-latest').textContent=modelState.busy?'Interpreting…':hosted&&!configured?'Show selected evidence notes':'Interpret selected evidence';
  const limits=status.limits||{},daily=finite(limits.daily_requests),budget=finite(limits.daily_reserved_usd),reservation=finite(limits.reserved_usd_per_call),cooldown=finite(limits.cooldown_ms);
  const newCalls=daily!==null&&budget!==null&&reservation!==null&&reservation>0?Math.min(daily,Math.floor(budget/reservation)):null;
  $('model-requirements').textContent=hosted?`Manual OpenAI API calls. ${newCalls!==null?'Up to '+newCalls+' new provider attempts per day. ':''}${daily!==null?'Daily request limit: '+daily+'. ':''}${budget!==null?'Daily reserved budget: $'+budget.toFixed(2)+'. ':''}${cooldown!==null?'Cooldown: '+Math.round(cooldown/1000)+' s. ':''}No model files are downloaded. Missing configuration returns deterministic evidence notes for known server evidence.`:'Manual companion inference. Original evidence survives unavailable or rejected model output. Eido remains evaluation-only unless a trusted checkpoint is explicitly configured.';
  const container=$('interpretation-result'),markup=interpretationMarkup(event);container.hidden=!markup;
  const signature=event?JSON.stringify([event.id,event.interpreted_at,event.interpretation_cached,event.interpretation]):'empty';
  if(container.dataset.signature!==signature){container.dataset.signature=signature;container.innerHTML=markup;bindInterpretationEvidence(container);}
}
async function refreshCompanionModel() {
  if(state.capabilities?.mode!=='local')return;
  try {
    const response=await fetch('/api/model/status',{cache:'no-store',credentials:'same-origin',signal:AbortSignal.timeout(5000)});
    if(!response.ok)return;
    modelState.companionStatus=await response.json();$('companion-model-option').disabled=modelState.companionStatus?.available!==true;renderInterpreter();
  }catch{/* A missing companion model never stops actual observations. */}
}
async function requestHostedInterpretation(evidenceId) {
  if(!canRequestHostedInterpretation())throw new Error('Server evidence storage is unavailable for Sol interpretation. Recorded facts remain usable.');
  const response=await fetch('/api/interpret',{method:'POST',headers:{'Content-Type':'application/json'},credentials:'same-origin',cache:'no-store',body:JSON.stringify({evidence_id:evidenceId}),signal:AbortSignal.timeout(90000)});
  let payload;try{payload=await response.json();}catch{throw new Error('The interpreter returned an unreadable response. Original evidence remains unchanged.');}
  if(!response.ok)throw new Error(readableError(payload,response.status===429?'The server’s interpretation limit or cooldown is active. No automatic retry will run.':response.status===409?'Another interpretation is in progress. Request this evidence again after it finishes.':'The server could not interpret this evidence.'));
  if(payload.evidence_id!==evidenceId)throw new Error('The interpreter returned a different evidence ID. The result was not attached.');
  if(!/^(edge|evt)_[0-9a-f]{32}$/i.test(payload.source_event_id||'')||evidenceId.startsWith('edge_')&&payload.source_event_id!==evidenceId||payload.source_event_id?.startsWith('evt_')&&payload.source_event_id!==evidenceId)throw new Error('The interpreter returned an invalid source receipt. The result was not attached.');
  if(!payload.analysis||!['model','deterministic'].includes(payload.analysis.kind))throw new Error('The interpreter returned an invalid result. Original evidence remains unchanged.');
  if(payload.model)modelState.hostedStatus=payload.model;
  return payload;
}
async function interpretEvent(event) {
  if(modelState.busy||!event)return;
  modelState.busy=true;modelState.issue=null;modelState.lastAttemptId=event.id;renderInterpreter();
  try {
    let analysis,cached=false,sourceEvidenceId=event.id;
    if(modelState.engine==='companion'){
      const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},credentials:'same-origin',cache:'no-store',body:JSON.stringify({event_id:event.id}),signal:AbortSignal.timeout(120000)});
      const payload=await response.json();if(!response.ok)throw new Error(readableError(payload,'The companion could not interpret this evidence.'));
      if(payload.event_id!==event.id)throw new Error('The companion returned a different observation ID. Nothing was attached.');
      analysis=payload.analysis;cached=payload.cached===true;if(payload.model)modelState.companionStatus=payload.model;
    }else{
      const evidenceId=hostedEvidenceId(event);if(!evidenceId)throw new Error('This browser-only record has no authoritative server evidence ID.');
      const payload=await requestHostedInterpretation(evidenceId);analysis=payload.analysis;cached=payload.cached===true;sourceEvidenceId=payload.source_event_id||evidenceId;
    }
    if(!analysis||!['model','deterministic'].includes(analysis.kind))throw new Error('The interpreter returned an invalid result. Original evidence remains unchanged.');
    event.interpretation=analysis;event.interpreted_at=new Date().toISOString();event.interpretation_cached=cached;event.interpretation_source_evidence_id=sourceEvidenceId;
    persist();state.inspectorSignature=null;renderInspector();
  }catch(error){modelState.issue=String(error.message||'Interpretation failed. Original evidence remains available.').slice(0,500);}
  finally{modelState.busy=false;renderInterpreter();}
}
async function interpretSensorEvent(event) {
  if(modelState.busy){notify('Another interpretation is running. This record remains available for a later manual request.');return;}
  const evidenceId=hostedEvidenceId(event);
  if(event?.source!=='observatory_edge'||!evidenceId){notify('This imported capture stays in your browser and is not registered as server evidence. Its original records and deterministic signals remain available.');return;}
  modelState.engine='hosted';$('model-engine').value='hosted';modelState.busy=true;modelState.issue=null;renderInterpreter();
  try {
    const payload=await requestHostedInterpretation(evidenceId);
    const analysis={...payload.analysis,provenance:{...payload.analysis.provenance,cached_server_result:payload.cached===true,source_event_id:payload.source_event_id,requested_evidence_id:evidenceId}};
    networkPanel?.attachInterpretation(event.id,analysis,new Date().toISOString());
  }catch(error){modelState.issue=String(error.message||'Sol could not interpret this sensor evidence. Original records remain available.').slice(0,500);notify(modelState.issue);}
  finally{modelState.busy=false;renderInterpreter();}
}
function renderAll() {
  renderTargetPanel($('target-probes'),{event:selectedEvent()||state.events[0],events:state.events,capabilities:state.capabilities});renderStatus();renderCards();renderMetrics();renderField();renderInspector();renderRecord();renderForecast();renderInterpreter();}

async function loadCapabilities() {
  try {
    const response=await fetch('/api/capabilities',{cache:'no-store',credentials:'same-origin',signal:AbortSignal.timeout(5000)});
    if(!response.ok)return;
    const capabilities=await response.json();state.capabilities=capabilities;modelState.hostedStatus=capabilities.hosted_interpretation||null;networkPanel?.setCapabilities?.(capabilities);
    const recordedLink=document.querySelector('a[href="/recorded"]');
    if(capabilities.recorded_url===null)recordedLink?.remove();
    if(capabilities.mode==='hosted'&&capabilities.direct_ip===false){
      $('connect-target').placeholder='eidos-works.com or https://your-site.com';
      $('connection-hint').textContent='Authorized public websites · every 15+ seconds · up to four targets. Direct public IPs require the local collector.';
      document.querySelector('.intro>div>p').innerHTML='Connect a public website and watch actual observations arrive.<br>Public IP monitoring is available through the same interface in the local collector.';
    }else if(capabilities.mode==='local')$('connection-hint').textContent='Local collector · authorized public websites or public IPs · every 15+ seconds · up to four targets.';
    for(const target of state.targets)restoreTargetHistory(target);
    $('probe-options').hidden=!capabilities.target_probes?.connected;
    refreshCompanionModel();renderStatus();renderInterpreter();
  }catch{/* Capability lookup is optional; it never starts sampling or blocks Connect. */}
}

$('test-ip').addEventListener('click',()=>{
  $('connect-target').value='1.1.1.1';
  $('connect-form').requestSubmit();
});
$('model-shortcut').addEventListener('click',()=>document.querySelector('.understanding').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'}));
$('model-engine').addEventListener('change',()=>{
  if(modelState.busy){$('model-engine').value=modelState.engine;return;}
  modelState.engine=$('model-engine').value;modelState.issue=null;renderInterpreter();
});
$('interpret-latest').addEventListener('click',()=>{const event=selectedEvent()||state.events[0];if(event)interpretEvent(event);});
$('connect-form').addEventListener('submit',event=>{
  event.preventDefault();formMessage('');
  if(state.targets.length>=MAX_TARGETS){formMessage('Four targets are connected. Remove one before adding another.');return;}
  let url;try{url=normalizeInput($('connect-target').value);}catch(error){formMessage(error.message);return;}
  const existing=state.targets.find(target=>target.url===url);
  if(existing){formMessage(existing.active?'This endpoint is already being observed.':'This endpoint is already listed. Use its Resume control to reconnect.');return;}
  const target={id:uuid(),sourceId:null,url,label:safeLabel(url),createdAt:new Date().toISOString(),active:true,inFlight:false,timer:null,controller:null,nextAt:null,lastIssue:null,blocked:false,brain:newBrain(),currentReceiptId:null,run:0};
  state.targets.push(target);state.followLatest=true;state.restored=false;state.sessionNotice=null;
  $('connect-target').value='';persist();renderAll();schedule(target,0);
});
$('pause-all').addEventListener('click',()=>{for(const target of state.targets){target.active=false;cancelWork(target);}state.restored=false;persist();renderAll();notify('All targets paused. Their observations remain inspectable.');});
$('view-all').addEventListener('click',()=>{state.sourceFilter=null;state.followLatest=true;state.selectedId=state.events[0]?.id||state.selectedId;renderAll();});
$('inspector-kind').addEventListener('click',()=>{state.followLatest=true;state.selectedId=(state.sourceFilter?latest(state.sourceFilter):state.events[0])?.id||state.selectedId;renderAll();});
document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{state.filter=button.dataset.filter;state.limit=8;document.querySelectorAll('[data-filter]').forEach(item=>{item.classList.toggle('active',item===button);item.setAttribute('aria-pressed',String(item===button));});renderRecord();}));
$('show-more').addEventListener('click',()=>{state.limit+=12;renderRecord();});
$('export-button').addEventListener('click',()=>{
  if(!state.events.length)return;
  const exported={schema:'sentinel-live-evidence-v2',exported_at:new Date().toISOString(),mode:'recorded_actual_observations',session_started_at:state.sessionStarted,source:'Measured endpoint evidence; authoritative server analysis is identified per record',engine:ENGINE_PROVENANCE,model:{engine:modelState.engine,manual_only:true,status:activeModelStatus(),role:'Evidence-linked interpretation; causes remain unverified'},capabilities:state.capabilities,scope:{protocols:['HTTP','HTTPS'],method:'HEAD',sampling:'Foreground, >=15 seconds after each result; gaps and failures retained',missing:['Visitor traffic','Internal networks','Packet contents','Origin logs','Authenticated state']},targets:state.targets.map(target=>({source_id:target.sourceId,url:target.url,label:target.label})),events:state.events,network_sensor_evidence:networkPanel?.getEvidence()||null};
  const blob=new Blob([JSON.stringify(exported,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');
  link.href=url;link.download='Sentinel_Evidence_'+new Date().toISOString().replace(/[:.]/g,'-')+'.json';document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);notify('Actual session evidence exported. Review it before sharing.');
});
['coverage-open','coverage-inline'].forEach(id=>$(id).addEventListener('click',()=>$('coverage-dialog').showModal()));
$('coverage-close').addEventListener('click',()=>$('coverage-dialog').close());
$('coverage-dialog').addEventListener('click',event=>{if(event.target!==$('coverage-dialog'))return;const rectangle=event.target.getBoundingClientRect();if(event.clientX<rectangle.left||event.clientX>rectangle.right||event.clientY<rectangle.top||event.clientY>rectangle.bottom)event.target.close();});
document.addEventListener('visibilitychange',()=>{
  if(!visible()){for(const target of state.targets)if(target.active)cancelWork(target);}
  else for(const target of state.targets)if(target.active&&!target.inFlight)schedule(target,MIN_INTERVAL);
  renderAll();
});
window.addEventListener('pagehide',()=>{for(const target of state.targets)cancelWork(target);persist();});
restore();renderAll();loadCapabilities();setInterval(tick,1000);
import('/network-panel.mjs').then(({initializeNetworkPanel})=>{networkPanel=initializeNetworkPanel($('network-panel'),{onNotice:notify,onInterpret:interpretSensorEvent});networkPanel?.setCapabilities?.(state.capabilities);}).catch(()=>{
  $('network-panel').innerHTML='<p class="empty-record">Network sensor import is unavailable in this build. Endpoint evidence collection remains available.</p>';
});
