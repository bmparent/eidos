(() => {
  'use strict';

  // Telemetry and model text are data. Nothing from either is evaluated or inserted unescaped.
  const $ = id => document.getElementById(id);
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number = value => typeof value === 'number' && Number.isFinite(value) ? value : null;
  const stamp = value => { const n = typeof value === 'number' ? value : Date.parse(value); return Number.isFinite(n) ? n : null; };
  const formatNumber = value => number(value) === null ? '—' : Math.round(value).toLocaleString();
  const formatTime = value => stamp(value) === null ? '—' : new Date(value).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
  const formatDate = value => stamp(value) === null ? '—' : new Date(value).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
  const dateISO = value => stamp(value) === null ? 'Not supplied' : new Date(value).toISOString();
  const recordedInput = window.__OBSERVATORY_RECORDING__;
  const recorded = !!recordedInput;
  const state = {snapshot:null,events:[],selectedId:null,sourceId:null,filter:'all',limit:8,connected:false,lastReceived:0,pendingProbe:false,probeExisting:new Set(),seen:new Set(),hasRendered:false,replayIndex:0,replayPlaying:false,replayEvents:[],replayBase:null,requesting:false,renderSignature:null,inspectorSignature:null};
  let replayTimer = null;
  let toastTimer = null;
  let probeTimer = null;
  const positions = {
    home:{x:625,y:132,labelX:677,labelY:129,anchor:'start',path:'M 399 215 C 468 167 535 109 625 132'},
    account:{x:790,y:316,labelX:838,labelY:313,anchor:'start',path:'M 409 246 C 552 218 709 222 790 316'},
    community:{x:601,y:485,labelX:650,labelY:482,anchor:'start',path:'M 389 278 C 450 343 457 471 601 485'},
    playground:{x:227,y:435,labelX:177,labelY:432,anchor:'end',path:'M 329 279 C 273 318 225 358 227 435'},
    probe:{x:140,y:125,labelX:189,labelY:122,anchor:'start',path:'M 300 225 C 245 176 192 112 140 125'}
  };

  function tell(message) {
    $('toast').textContent = message;
    $('toast').hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { $('toast').hidden = true; }, 4200);
  }

  function getSources() {
    const sources = [...(state.snapshot?.sources || [])];
    if (state.events.some(e => e.source_id === 'probe') && !sources.some(s => s.id === 'probe')) sources.push({id:'probe',label:'Diagnostic probe',state:'fresh'});
    return sources;
  }
  function sourceFor(id) { return getSources().find(s => s.id === id) || {id,label:id || 'Unknown source'}; }
  function orderedEvents() { return [...state.events].sort((a,b) => (stamp(a.observed_at) || 0) - (stamp(b.observed_at) || 0)); }
  function sourceEvents(id) { return state.events.filter(e => e.source_id === id); }
  function latestEvent(id) { return sourceEvents(id).sort((a,b) => (stamp(b.observed_at) || 0) - (stamp(a.observed_at) || 0))[0]; }
  function changeIds() {
    const last = new Map(), changes = new Set();
    for (const e of orderedEvents()) {
      if (e.kind === 'controlled_probe') continue;
      const key = e.error?.code || e.status;
      if (last.has(e.source_id) && last.get(e.source_id) !== key) changes.add(e.id);
      last.set(e.source_id,key);
    }
    return changes;
  }
  function isUnsuccessful(e) { return !!e?.error || number(e?.status) === null || e.status < 200 || e.status >= 400; }
  function eventColor(e,changes) { return isUnsuccessful(e) || changes?.has(e.id) ? '#efb779' : '#b4eedc'; }
  function statusText(e) { return e?.error ? e.error.code || 'Error' : number(e?.status) === null ? 'No response' : String(e.status); }
  function eventTitle(e) { return e.kind === 'controlled_probe' ? 'Diagnostic probe' : sourceFor(e.source_id).label; }
  function eventSummary(e) {
    if (e.error) return `The observer could not complete the request: ${e.error.code || 'transport error'}.${number(e.status)!==null ? ` HTTP ${e.status} was observed before the request chain stopped.` : ' No HTTP response was observed.'}`;
    return `The observer received HTTP ${statusText(e)} from ${sourceFor(e.source_id).label}${number(e.latency_ms) === null ? '' : ` after ${Math.round(e.latency_ms)} ms elapsed`}.${e.response_chain_complete===false?' The response chain is incomplete.':''}`;
  }
  function selectedEvent() { return state.events.find(e => e.id === state.selectedId); }

  function renderConnection() {
    const s = state.snapshot;
    const badge = $('mode-badge');
    badge.className = 'mode-badge';
    if (recorded) {
      badge.textContent = 'RECORDED'; badge.classList.add('recorded');
      $('freshness').textContent = 'Real measurements · replay';
      $('capture-time').textContent = `Captured ${formatDate(s?.captured_at)}`;
      return;
    }
    if (!state.connected) {
      badge.textContent = s ? 'DISCONNECTED' : 'CONNECTING';
      badge.classList.add('disconnected');
      $('freshness').textContent = s ? 'Retaining last received evidence' : 'No telemetry received';
    } else {
      if(s.collection_enabled===false){badge.textContent='PAUSED';badge.classList.add('disconnected');$('freshness').textContent='Collector connected · periodic sampling off';$('capture-time').textContent=`Collector time · ${formatTime(s.captured_at)}`;return;}
      const latest = state.events[0];
      const age = latest ? Math.max(0, (stamp(s.captured_at) || Date.now()) - (stamp(latest.observed_at) || 0)) + (Date.now() - state.lastReceived) : null;
      const stale = age !== null && age > Math.max(90000,(s.cadence_seconds || 30)*3000);
      badge.textContent = stale ? 'STALE' : 'LIVE';
      if(stale) badge.classList.add('disconnected');
      $('freshness').textContent = age === null ? 'Connected · awaiting first sample' : `Last observation ${age < 2000 ? 'just received' : `${Math.floor(age/1000)}s ago`}`;
    }
    $('capture-time').textContent = `Collector time · ${formatTime(s?.captured_at)}`;
  }

  function renderMetrics() {
    const events = state.events;
    const latest = events.find(e => number(e.latency_ms) !== null);
    const sources = getSources().filter(s => s.id !== 'probe');
    const measured = sources.filter(s => events.some(e => e.source_id === s.id)).length;
    const first = events[events.length - 1];
    const last = events[0];
    const elapsed = first && last ? Math.max(0, (stamp(last.observed_at) - stamp(first.observed_at)) / 1000) : 0;
    $('metric-count').textContent = events.length.toLocaleString();
    $('metric-period').textContent = first ? `${elapsed < 60 ? Math.round(elapsed) + ' seconds' : (elapsed / 60).toFixed(1) + ' minutes'} of recorded evidence` : 'Awaiting first collection';
    $('metric-latency').innerHTML = `${formatNumber(latest?.latency_ms)}<small>ms</small>`;
    $('metric-latency-source').textContent = latest ? `${eventTitle(latest)} · ${latest.error?'incomplete request':'HTTP '+statusText(latest)}` : 'No timed response yet';
    $('metric-routes').innerHTML = `${measured}<small>/ ${sources.length || 4}</small>`;
    $('metric-changes').textContent = changeIds().size;
    $('boundary-summary').textContent = state.snapshot?.boundary?.title || 'Eidos Works · public HTTP observations';
    const missing = state.snapshot?.boundary?.missing;
    $('scope-note').textContent = 'Public HTTP measurements only. ' + (Array.isArray(missing) && missing.length ? 'Missing coverage: ' + missing.slice(0,2).join(' · ') : 'No visitor traffic, internal host, or Cloudflare security-event visibility.');
  }

  function renderField() {
    const changes = changeIds();
    const sources = getSources();
    $('field-empty').hidden = state.events.length > 0;
    if (!state.events.length) { $('evidence-field').innerHTML = ''; return; }
    const parts = [
      '<desc id="field-desc">One external observer connected to measured public HTTP routes. Each dot is an actual observation; rings arrange the latest 48 observations for each route. Select a route or a dot to inspect evidence. Arcs represent measurement relationships, not visitor traffic.</desc>',
      '<defs><linearGradient id="arc-gradient" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#8acbde" stop-opacity=".12"/><stop offset=".52" stop-color="#92c7d6" stop-opacity=".64"/><stop offset="1" stop-color="#b4eedc" stop-opacity=".32"/></linearGradient><radialGradient id="observer-glow"><stop stop-color="#396474" stop-opacity=".2"/><stop offset="1" stop-color="#123447" stop-opacity="0"/></radialGradient><filter id="point-glow"><feGaussianBlur stdDeviation="2"/></filter></defs>',
      '<g aria-hidden="true"><ellipse cx="440" cy="290" rx="285" ry="191" transform="rotate(17 440 290)" fill="none" stroke="#1b2c3a" stroke-width=".7"/><ellipse cx="442" cy="290" rx="365" ry="251" transform="rotate(17 442 290)" fill="none" stroke="#152431" stroke-width=".7"/><path d="M75 290H917M442 48V556" stroke="#162736" stroke-width=".6" stroke-dasharray="2 9"/><circle cx="355" cy="237" r="130" fill="url(#observer-glow)"/></g>'
    ];
    for (const source of sources) {
      const p = positions[source.id]; if (!p) continue;
      const events = sourceEvents(source.id).slice(0,48);
      const last = latestEvent(source.id);
      const dim = state.sourceId && state.sourceId !== source.id;
      const historical = !recorded && (!state.connected || source.state==='stale' || state.snapshot?.collection_enabled===false);
      const color = last && !historical ? eventColor(last,changes) : '#718694';
      const selected = state.sourceId === source.id;
      parts.push(`<g class="source-connector ${dim ? 'source-dim' : ''}" aria-hidden="true"><path d="${p.path}" fill="none" stroke="${color}" stroke-width="6" opacity=".035"/><path d="${p.path}" fill="none" stroke="url(#arc-gradient)" stroke-width="1.2" ${last ? '' : 'stroke-dasharray="3 7"'}/></g>`);
      parts.push(`<g class="node ${selected ? 'is-selected' : ''} ${dim ? 'source-dim' : ''}" role="button" tabindex="0" data-source="${escape(source.id)}" aria-label="${escape(source.label)}, ${events.length} visible observations, ${last ? 'HTTP '+escape(statusText(last)) : 'waiting for a measurement'}. Select to filter." aria-pressed="${!!selected}">`);
      parts.push(`<circle class="node-hit" cx="${p.x}" cy="${p.y}" r="48"/><circle cx="${p.x}" cy="${p.y}" r="36" fill="#0b1722" stroke="${selected ? '#477166' : '#263844'}" stroke-width=".8"/><circle class="node-core" cx="${p.x}" cy="${p.y}" r="${selected ? 7 : 5}" fill="${color}"/><circle cx="${p.x}" cy="${p.y}" r="17" fill="none" stroke="${color}" stroke-width=".65" opacity=".26"/>`);
      parts.push(`<text class="node-label" x="${p.labelX}" y="${p.labelY}" text-anchor="${p.anchor}">${escape(source.label)}</text><text class="node-detail" x="${p.labelX}" y="${p.labelY+19}" text-anchor="${p.anchor}">${last ? escape(statusText(last))+' · '+formatNumber(last.latency_ms)+' ms' : 'awaiting evidence'}</text><text class="node-sub" x="${p.labelX}" y="${p.labelY+36}" text-anchor="${p.anchor}">${sourceEvents(source.id).length} observations${historical ? ' · cached' : ''}</text></g>`);
      const ringEvents = [...events].reverse();
      ringEvents.forEach((e,index) => {
        const radius = 36 + Math.floor(index/24)*8;
        const theta = ((index%24)/24)*Math.PI*2 - Math.PI/2;
        const x = p.x + Math.cos(theta)*radius, y = p.y + Math.sin(theta)*radius;
        const c = eventColor(e,changes);
        const sel = e.id === state.selectedId;
        const fresh = !state.seen.has(e.id) && state.hasRendered && !recorded;
        parts.push(`<circle class="event-dot ${dim ? 'source-dim' : ''}" cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${sel?4:2.4}" fill="${c}" ${sel?'stroke="#fff" stroke-width="1"':''} role="button" tabindex="0" data-event="${escape(e.id)}" aria-label="${escape(source.label)} observation at ${formatTime(e.observed_at)}, HTTP ${escape(statusText(e))}, ${formatNumber(e.latency_ms)} milliseconds"><title>${escape(source.label)} · ${formatTime(e.observed_at)} · ${escape(statusText(e))} · ${formatNumber(e.latency_ms)} ms</title></circle>`);
        if (fresh) parts.push(`<circle class="event-halo" cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="5" aria-hidden="true"/>`);
      });
    }
    parts.push('<g aria-hidden="true"><circle cx="355" cy="237" r="58" fill="#101f2b" stroke="#365360" stroke-width="1"/><circle cx="355" cy="237" r="51" fill="none" stroke="#1d3442" stroke-width=".7"/><path d="M 343 221 a 17 17 0 1 0 23 26 M 353 225 a 10 10 0 1 0 8 17" fill="none" stroke="#b4eedc" stroke-width="1.2"/><circle cx="357" cy="233" r="2" fill="#b4eedc"/><text class="field-origin-title" x="355" y="327" text-anchor="middle">The observer</text><text class="field-origin-detail" x="355" y="346" text-anchor="middle">EXTERNAL HTTP · READ ONLY</text><text class="field-small-label" x="25" y="577">MEASURED RELATIONSHIPS / NOT INTERNAL NETWORK TOPOLOGY</text></g>');
    const focusId=document.activeElement?.dataset?.event, focusSource=document.activeElement?.dataset?.source;
    $('evidence-field').innerHTML = parts.join('');
    if(focusId||focusSource){const target=[...$('evidence-field').querySelectorAll('[data-event],[data-source]')].find(el=>(focusId&&el.dataset.event===focusId)||(focusSource&&el.dataset.source===focusSource));target?.focus({preventScroll:true});}
    $('evidence-field').querySelectorAll('[data-source]').forEach(el => bindActivation(el,() => selectSource(el.dataset.source)));
    $('evidence-field').querySelectorAll('[data-event]').forEach(el => bindActivation(el,() => selectEvent(el.dataset.event)));
    state.events.forEach(e => state.seen.add(e.id));
    state.hasRendered = true;
  }

  function bindActivation(el,handler) {
    el.addEventListener('click',handler);
    el.addEventListener('keydown',e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handler(); } });
  }
  function selectSource(id) {
    state.sourceId = state.sourceId === id ? null : id;
    state.limit = 8;
    if (state.sourceId) state.selectedId = latestEvent(id)?.id || null;
    renderField(); renderInspector(); renderRecord();
    $('reset-source').hidden = !state.sourceId;
  }
  function selectEvent(id) {
    state.selectedId = id;
    renderField(); renderInspector(); renderRecord();
    if (window.innerWidth < 681) document.querySelector('.inspector').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});
  }
  function safeList(value) { return Array.isArray(value) ? value : []; }
  function plainValue(value) { if (value === null || value === undefined) return 'Not supplied'; if(typeof value === 'object') return JSON.stringify(value); return String(value); }
  function evidenceRefs(ids,label='Evidence') { return safeList(ids).length ? `<span class="evidence-reference">${escape(label)} · ${safeList(ids).map(id=>state.events.some(e=>e.id===id)?`<button data-citation="${escape(id)}" title="Inspect ${escape(id)}">${escape(String(id).slice(0,18))}…</button>`:`<span title="${escape(id)}">${escape(String(id).slice(0,18))}… (outside retained record)</span>`).join(' ')}</span>` : ''; }
  function signalMarkup(brain) {
    if (!brain) return '<p>Brain analysis has not been attached to this observation.</p>';
    const active = brain.event_active === true;
    const signalState = brain.state || brain.status || 'observing';
    const chips = [`<span class="brain-chip ${active?'warn':''}">${escape(signalState.replace(/_/g,' '))}</span>`];
    if (number(brain.warmup_remaining)>0) chips.push(`<p style="margin-top:10px">${brain.warmup_remaining} more measurements needed for provisional calibration.</p>`);
    const signals=safeList(brain.signals);
    if(signals.length) chips.push('<ul class="signal-list">'+signals.slice(0,4).map(sig=>`<li>${escape(sig.label || sig.key)}<strong>${number(sig.value)!==null?Number(sig.value).toFixed(2):'—'}</strong><small>${escape(sig.meaning || 'Experimental signal; not a security severity.')}</small></li>`).join('')+'</ul>');
    chips.push('<p class="evidence-hint">Experimental stream analysis. Signal scores are not attack probabilities.</p>');
    return chips.join('');
  }
  function securityMarkup(headers) {
    const controls=[['hsts_present','HSTS'],['csp_present','CSP'],['x_content_type_options','Content type'],['frame_options','Frame policy'],['referrer_policy','Referrer policy'],['cache_private','Private cache'],['cache_no_store','No-store cache']];
    return '<div class="inspector-block"><h3>Observed response controls</h3><dl class="security-controls">'+controls.map(([key,label])=>{const val=headers?.[key];return `<div><dt>${label}</dt><dd>${val===null||val===undefined?'not captured':val===true?'present':val===false?'absent':escape(val)}</dd></div>`;}).join('')+'</dl><p class="evidence-hint">Header presence is an observation, not a vulnerability grade. Absent does not mean exploited.</p></div>';
  }

  function renderInspector() {
    const e = selectedEvent();
    if (!e) return;
    const signature=JSON.stringify(e);
    if(state.inspectorSignature===signature)return;
    const wasOpen=document.querySelector('.technical-details')?.open;
    const previousId=$('inspector-content').dataset.eventId;
    state.inspectorSignature=signature;$('inspector-content').dataset.eventId=e.id;
    const source = sourceFor(e.source_id);
    const analysis = e.analysis || {};
    const catalogue = analysis.provenance?.role === 'ranked_catalogue_selection';
    const facts = safeList(analysis.facts);
    const hypotheses = safeList(analysis.hypotheses);
    const unknowns = safeList(analysis.unknowns);
    const headers = e.headers || {};
    const url = e.url || source.url || e.path || 'URL not supplied';
    $('inspector-kind').textContent = e.kind === 'controlled_probe' ? 'Controlled observation' : 'Observed fact';
    $('inspector-content').innerHTML = `
      <h2 id="inspector-title">${escape(eventTitle(e))}</h2>
      <p class="event-path">${escape(url)}</p>
      <div class="inspector-reading"><div><span class="reading-label">HTTP response</span><strong class="${isUnsuccessful(e)?'status-change':'status-good'}">${escape(statusText(e))}</strong></div><div><span class="reading-label">Observer elapsed time</span><strong>${formatNumber(e.latency_ms)}<small>ms</small></strong></div></div>
      <div class="inspector-time">${escape(formatDate(e.observed_at))} · collector time</div>
      <div class="inspector-block"><h3>What was observed</h3><p>${escape(eventSummary(e))}</p><p class="evidence-hint">Elapsed time is measured by this observer, through its network path. It is not origin processing time or page-load time.</p></div>
      ${securityMarkup(e.security_headers)}
      <div class="inspector-block"><h3>${analysis.kind==='model'&&!catalogue?'Model interpretation':'Evidence notes'}</h3>${analysis.kind==='model'?`<p class="interpretation-label">${catalogue?'Measurement template · model ranks predefined possibilities':'Unverified model prose · inspect cited measurements'}</p>`:''}<p>${escape(analysis.summary || 'Interpretation is pending. Collection continues independently.')}</p>${facts.length?'<ul style="margin-top:12px">'+facts.map(f=>`<li class="fact">${escape(typeof f==='string'?f:f.text)}${typeof f==='object'?evidenceRefs(f.evidence_ids):''}</li>`).join('')+'</ul>':''}</div>
      <div class="inspector-block"><h3>Brain / Sentinel <span style="color:var(--faint);font-size:8px;font-weight:400">· experimental</span></h3>${signalMarkup(e.brain)}${e.brain?.forecast?.issued?`<p class="evidence-hint">Issued after this observation: next-observation EWMA estimate ${formatNumber(e.brain.forecast.issued.predicted_latency_ms)} ms; persistence ${formatNumber(e.brain.forecast.issued.persistence_latency_ms)} ms. Experimental latency forecast, not an incident prediction.</p>`:''}</div>
      <div class="inspector-block"><h3>${catalogue?'Model-ranked predefined possibilities':'Possible explanations'}</h3>${catalogue?'<p class="interpretation-label">Causes unverified · ranking is not a diagnosis</p>':''}${hypotheses.length?hypotheses.map(h=>`<div class="hypothesis"><small>Hypothesis · not established</small><p>${escape(typeof h==='string'?h:h.text)}</p>${evidenceRefs(h.support,'Linked context')}${h.next_observation?`<p class="next">Next observation: ${escape(h.next_observation)}</p>`:''}${safeList(h.support).some(id=>state.events.find(event=>event.id===id)?.source_id!==e.source_id)?'<p class="next">Other routes provide context; they do not establish this event’s cause.</p>':''}${safeList(h.against).length?`<p class="next">Counterevidence: ${safeList(h.against).map(escape).join(', ')}</p>`:''}</div>`).join(''):'<p>No competing explanation is supported yet. A response code alone does not establish cause.</p>'}</div>
      ${unknowns.length?`<div class="inspector-block"><h3>Still unknown</h3><ul>${unknowns.map(u=>`<li>${escape(typeof u==='string'?u:u.text||plainValue(u))}</li>`).join('')}</ul></div>`:''}
      <details class="technical-details"><summary>Original measurement & provenance</summary><dl><dt>Observation ID</dt><dd>${escape(e.id)}</dd><dt>Observed at (collector UTC)</dt><dd>${escape(dateISO(e.observed_at))}</dd><dt>Evidence hash</dt><dd>${escape(e.evidence_hash || 'Not supplied')}</dd><dt>Analysis source</dt><dd>${escape(analysis.kind || 'No interpretation attached')}</dd><dt>Measurement kind</dt><dd>${escape(e.kind || 'measurement')}</dd>${Object.entries(headers).map(([k,v])=>`<dt>${escape(k)}${k.toLowerCase()==='date'?' (upstream clock)':''}</dt><dd>${escape(plainValue(v))}</dd>`).join('')}${e.error?`<dt>Transport error</dt><dd>${escape(plainValue(e.error))}</dd>`:''}</dl><button class="copy-button" id="copy-evidence">Copy evidence JSON</button></details>`;
    if(wasOpen&&previousId===e.id)document.querySelector('.technical-details').open=true;
    $('inspector-content').querySelectorAll('[data-citation]').forEach(button=>button.addEventListener('click',()=>selectEvent(button.dataset.citation)));
    $('copy-evidence').addEventListener('click',async () => {
      try { await navigator.clipboard.writeText(JSON.stringify(e,null,2)); tell('Original measurement copied.'); }
      catch { tell('Clipboard is unavailable in this browser context.'); }
    });
  }

  function filteredEvents() {
    const changes = changeIds();
    return state.events.filter(e=>(!state.sourceId||e.source_id===state.sourceId)&&(state.filter==='all'||(state.filter==='changes'&&changes.has(e.id))||(state.filter==='probes'&&e.kind==='controlled_probe')));
  }
  function renderRecord() {
    const events = filteredEvents();
    const changes = changeIds();
    const visible = events.slice(0,state.limit);
    $('event-list').innerHTML = visible.length ? visible.map(e=>`<button class="record-button ${e.id===state.selectedId?'selected':''}" data-event="${escape(e.id)}" aria-label="Inspect ${escape(eventTitle(e))}, HTTP ${escape(statusText(e))}, ${formatTime(e.observed_at)}" ${e.id===state.selectedId?'aria-current="true"':''}><span class="record-name"><i class="legend-dot" style="background:${eventColor(e,changes)}"></i>${escape(eventTitle(e))}<small>${e.kind==='controlled_probe'?'controlled probe':e.error||e.response_chain_complete===false?'incomplete request':changes.has(e.id)?'status changed':'HEAD observation'}</small></span><span class="record-status ${isUnsuccessful(e)||changes.has(e.id)?'status-change':'status-good'}">${escape(statusText(e))}</span><span class="record-latency">${formatNumber(e.latency_ms)} ms</span><span class="record-time">${formatTime(e.observed_at)}</span><span class="record-arrow" aria-hidden="true">↗</span></button>`).join('') : `<p class="empty-state">${state.filter==='changes'?'No response-status changes in this measured window.':state.filter==='probes'?'No diagnostic probe has been recorded in this window.':'No observations for this selection.'}</p>`;
    $('event-list').querySelectorAll('[data-event]').forEach(el=>el.addEventListener('click',()=>selectEvent(el.dataset.event)));
    $('show-more').hidden = events.length <= state.limit;
    $('show-more').textContent = `Show more observations (${events.length-state.limit} remaining)`;
    renderTimeline(events,changes);
  }
  function renderTimeline(events,changes) {
    const data = [...events].filter(e=>stamp(e.observed_at)!==null&&number(e.latency_ms)!==null).reverse();
    if (!data.length) { $('timeline').innerHTML='<text x="600" y="65" text-anchor="middle" fill="#596d7c" font-family="sans-serif" font-size="12">No timed observations for this selection</text>'; return; }
    const width=1200,left=50,right=12,top=12,bottom=92;
    const min=Math.min(...data.map(e=>stamp(e.observed_at))),max=Math.max(...data.map(e=>stamp(e.observed_at)));
    const peak=Math.max(1,...data.map(e=>e.latency_ms));
    const x=e=>left+(stamp(e.observed_at)-min)/Math.max(1,max-min)*(width-left-right);
    const y=e=>bottom-(e.latency_ms/peak)*(bottom-top);
    const parts=[`<path d="M${left} ${bottom}H${width-right}M${left} ${top}H${width-right}" stroke="#1b2c3a" stroke-width="1"/><text x="0" y="${top+4}" fill="#596d7c" font-family="monospace" font-size="10">${Math.round(peak)} ms</text><text x="0" y="${bottom+4}" fill="#596d7c" font-family="monospace" font-size="10">0</text>`];
    getSources().forEach(source=>{
      const route=data.filter(e=>e.source_id===source.id);
      if(route.length>1) parts.push(`<path d="${route.map((e,i)=>(i?'L':'M')+x(e).toFixed(1)+' '+y(e).toFixed(1)).join(' ')}" fill="none" stroke="#6f9ba9" stroke-width="1" opacity=".28"/>`);
    });
    data.forEach(e=>parts.push(`<circle cx="${x(e).toFixed(1)}" cy="${y(e).toFixed(1)}" r="${e.id===state.selectedId?4.5:3}" fill="${eventColor(e,changes)}" ${e.id===state.selectedId?'stroke="#fff" stroke-width="1"':''}><title>${escape(eventTitle(e))} · ${formatTime(e.observed_at)} · ${Math.round(e.latency_ms)} ms · HTTP ${escape(statusText(e))}</title></circle>`));
    parts.push(`<text x="${left}" y="115" fill="#596d7c" font-family="monospace" font-size="9">${formatTime(min)}</text><text x="${width-right}" y="115" text-anchor="end" fill="#596d7c" font-family="monospace" font-size="9">${formatTime(max)}</text>`);
    $('timeline').innerHTML=parts.join('');
    $('timeline').setAttribute('aria-label',`${data.length} actual observer-duration measurements, from ${formatTime(min)} to ${formatTime(max)} collector time. Longest elapsed observation ${Math.round(peak)} milliseconds. Durations include incomplete requests where present. Inspect individual measurements in the table below.`);
    $('timeline-period').textContent = `${state.sourceId?sourceFor(state.sourceId).label+' · ':''}${data.length} measured points · collector time`;
  }

  function renderUnderstanding() {
    const model=state.snapshot?.model || {};
    $('model-state').textContent = model.state || 'Unavailable';
    $('model-name').textContent = model.name || 'Deterministic evidence notes';
    $('model-reason').textContent = model.reason || 'No model run has been proven in this environment. Original measurements remain inspectable without model interpretation.';
    const brain=state.snapshot?.brain || {};
    // Use only the latest *visible* event for each route. Replay cannot look ahead.
    const comparisons=getSources().map(source=>({source,comparison:latestEvent(source.id)?.brain?.forecast?.comparison})).filter(row=>number(row.comparison?.scored_count)>0);
    const count=comparisons.reduce((total,row)=>total+row.comparison.scored_count,0);
    const ewma=count?comparisons.reduce((total,row)=>total+(number(row.comparison.ewma_mae_ms)||0)*row.comparison.scored_count,0)/count:null;
    const persistence=count?comparisons.reduce((total,row)=>total+(number(row.comparison.persistence_mae_ms)||0)*row.comparison.scored_count,0)/count:null;
    const issuedPending=getSources().some(source=>latestEvent(source.id)?.brain?.forecast?.issued);
    $('forecast-title').textContent = count ? 'Predictive skill not established' : issuedPending ? 'Awaiting first scored forecast' : 'Forecast withheld';
    $('forecast-reason').textContent = count ? `${count} next-observation forecasts scored against later measurements across ${comparisons.length} routes. Lower error is better. This short, dependent window cannot establish general predictive skill or predict security incidents.` : 'Too few observations to score a forecast against later evidence. The separate latency experiment must beat simple persistence before it can be useful.';
    $('forecast-metric').textContent = count ? `${count} pairs` : '—';
    const comparisonElement=$('forecast-comparison');
    comparisonElement.hidden=!count;
    comparisonElement.innerHTML=count?`<div><span>EWMA mean absolute error</span><strong>${formatNumber(ewma)} <small>ms</small></strong></div><div><span>Persistence baseline error</span><strong>${formatNumber(persistence)} <small>ms</small></strong></div>`:'';
    const notes=[...safeList(brain.validation_notes),...safeList(brain.limitations)];
    $('brain-method').innerHTML = '<p>Brain / Sentinel looks for changes within the measured response stream. Scores are experimental, not calibrated attack probabilities. The model cannot initiate network or defensive actions.</p>'+(notes.length?'<ul>'+notes.map(n=>`<li>${escape(typeof n==='string'?n:plainValue(n))}</li>`).join('')+'</ul>':'')+'<p style="margin-top:10px">A model explanation is separate from an observation. Competing explanations remain hypotheses until supporting evidence arrives.</p>';
  }

  function renderCoverage() {
    const b=state.snapshot?.boundary || {}, s=state.snapshot || {};
    const missing=safeList(b.missing);
    $('coverage-content').innerHTML=`<h3>Monitored boundary</h3><p>${escape(b.description || 'Read-only HTTP HEAD requests to selected Eidos Works public routes from a single external observer.')}</p><h3>Vantage point</h3><p>${escape(plainValue(b.vantage || 'One external collection environment. Not a packet tap or Cloudflare traffic feed.'))}</p><h3>Sampling & freshness</h3><p>${escape(plainValue(b.sampling || `One measurement per selected route approximately every ${s.cadence_seconds || 'unknown'} seconds. No visitor request bodies are collected.`))}</p><p style="margin-top:8px">${recorded?'This artifact replays recorded real observations. It has no live connection and cannot send probes.':'The interface requests the latest collection snapshot every three seconds. A disconnected interface retains evidence with its original timestamps.'}</p><h3>Missing coverage</h3><ul>${(missing.length?missing:['Cloudflare visitor traffic and security events','Internal hosts, network packets, authentication outcomes, and application logs','Visibility from other networks and geographic regions']).map(m=>`<li>${escape(plainValue(m))}</li>`).join('')}</ul><h3>What a diagnostic probe does</h3><p>A single harmless HEAD request to a deliberately missing public path. The expected result is HTTP 404; the actual result is always retained, including redirects or application fallback responses.</p><h3>Time & evidence</h3><p>Observation times belong to the collector. The upstream HTTP Date header is shown separately and may differ; it is never used to infer freshness. Evidence hashes allow a saved measurement to be checked for accidental changes.</p><h3>Interpretation is untrusted</h3><p>Incoming headers and model output are displayed as inert text. They cannot issue instructions, run code, or take defensive action.</p>`;
  }

  function acceptSnapshot(snapshot) {
    if (!snapshot || typeof snapshot!=='object' || !Array.isArray(snapshot.events)) throw new Error('The collector returned an invalid snapshot.');
    state.snapshot=snapshot;
    state.events=[...snapshot.events].sort((a,b)=>(stamp(b.observed_at)||0)-(stamp(a.observed_at)||0));
    const recovered=!state.connected;
    state.connected=true; state.lastReceived=Date.now();
    $('connection-note').hidden=true;
    if(state.pendingProbe) {
      const probe=state.events.find(e=>e.kind==='controlled_probe'&&!state.probeExisting.has(e.id));
      if(probe){state.selectedId=probe.id;state.pendingProbe=false;clearTimeout(probeTimer);$('probe-button').disabled=false;$('probe-button').innerHTML='<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 3v14M3 10h14"/></svg>Run diagnostic probe';tell(`Probe recorded: ${statusText(probe)}. Select its evidence to inspect.`);}
    }
    if(!state.selectedId||!state.events.some(e=>e.id===state.selectedId)) state.selectedId=state.events[0]?.id || null;
    renderConnection(); renderMetrics();
    const signature=JSON.stringify([state.events,snapshot.sources,snapshot.brain,snapshot.model,snapshot.collection_enabled]);
    if(recovered||signature!==state.renderSignature){state.renderSignature=signature;renderField();renderInspector();renderRecord();renderUnderstanding();renderCoverage();}
  }

  async function poll() {
    if(recorded||state.requesting) return;
    state.requesting=true;
    try{
      const response=await fetch('/api/snapshot',{cache:'no-store',signal:AbortSignal.timeout(8000)});
      if(!response.ok) throw new Error(`Collector returned HTTP ${response.status}.`);
      acceptSnapshot(await response.json());
    }catch(error){const wasConnected=state.connected;state.connected=false;renderConnection();if(wasConnected)renderField();$('connection-note').textContent=state.snapshot?'Connection to the collector is interrupted. These are the last received measurements, not live traffic.':'The collector is not reachable. No measurements will be invented; waiting for a real connection.';$('connection-note').hidden=false;}
    finally{state.requesting=false;}
  }

  async function runProbe() {
    if(recorded||state.pendingProbe) return;
    state.probeExisting=new Set(state.events.map(e=>e.id));state.pendingProbe=true;
    $('probe-button').disabled=true;$('probe-button').textContent='Awaiting real observation…';
    try{
      const response=await fetch('/api/probe',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}',signal:AbortSignal.timeout(20000)});
      if(!response.ok) throw new Error(`Probe request returned HTTP ${response.status}.`);
      const result=await response.json();
      if(result.event?.id){state.selectedId=result.event.id;}
      tell('Diagnostic probe queued. Waiting for the actual response.');
      poll();
      probeTimer=setTimeout(()=>{if(!state.pendingProbe)return;state.pendingProbe=false;$('probe-button').disabled=false;$('probe-button').textContent='Run diagnostic probe';tell('No probe result received yet. The observation record remains authoritative.');},60000);
    }catch(error){state.pendingProbe=false;$('probe-button').disabled=false;$('probe-button').textContent='Run diagnostic probe';tell(error.name==='TimeoutError'?'Probe request timed out. Check the record before retrying.':error.message);}
  }

  function prepareReplay() {
    const input=recordedInput;
    const snapshots=Array.isArray(input)?input:Array.isArray(input.snapshots)?input.snapshots:null;
    const base=snapshots?snapshots[snapshots.length-1]:(input.snapshot||input);
    if(!base||!Array.isArray(base.events)){ $('connection-note').textContent='This recorded artifact does not contain a valid evidence snapshot.';$('connection-note').hidden=false;return; }
    state.replayBase=base;
    const all=snapshots?snapshots.flatMap(s=>s.events||[]):base.events;
    state.replayEvents=[...new Map(all.map(e=>[e.id,e])).values()].sort((a,b)=>(stamp(a.observed_at)||0)-(stamp(b.observed_at)||0));
    state.replayIndex=state.replayEvents.length;
    $('replay-bar').hidden=false;
    $('replay-scrub').max=Math.max(1,state.replayEvents.length);
    $('replay-scrub').value=state.replayIndex;
    $('probe-button').disabled=true;$('probe-button').textContent='Probes unavailable in replay';$('probe-button').title='Recorded artifacts have no live connection';
    renderReplay();
  }
  function renderReplay() {
    const events=state.replayEvents.slice(0,state.replayIndex).reverse();
    const sources=(state.replayBase.sources||[]).filter(s=>s.id!=='probe'||events.some(e=>e.source_id==='probe')).map(s=>({...s,state:events.some(e=>e.source_id===s.id)?'recorded':'waiting',latest:events.find(e=>e.source_id===s.id)||null,event_count:events.filter(e=>e.source_id===s.id).length}));
    acceptSnapshot({...state.replayBase,mode:'recorded',events,sources});
    $('replay-scrub').value=state.replayIndex;
    $('replay-position').textContent=`${state.replayIndex} / ${state.replayEvents.length}`;
  }
  function pauseReplay() {clearInterval(replayTimer);state.replayPlaying=false;$('replay-play').textContent='Play';$('replay-play').setAttribute('aria-label','Play recorded observations');}
  function toggleReplay() {
    if(state.replayPlaying){pauseReplay();return;}
    if(state.replayIndex>=state.replayEvents.length)state.replayIndex=0;
    state.replayPlaying=true;$('replay-play').textContent='Pause';$('replay-play').setAttribute('aria-label','Pause recorded observations');
    replayTimer=setInterval(()=>{state.replayIndex=Math.min(state.replayEvents.length,state.replayIndex+1);state.selectedId=state.replayEvents[state.replayIndex-1]?.id;renderReplay();if(state.replayIndex>=state.replayEvents.length)pauseReplay();},900);
  }

  $('probe-button').addEventListener('click',runProbe);
  $('reset-source').addEventListener('click',()=>{state.sourceId=null;$('reset-source').hidden=true;renderField();renderRecord();});
  document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{state.filter=button.dataset.filter;state.limit=8;document.querySelectorAll('[data-filter]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});renderRecord();}));
  $('show-more').addEventListener('click',()=>{state.limit+=12;renderRecord();});
  ['coverage-open','coverage-inline'].forEach(id=>$(id).addEventListener('click',()=>{renderCoverage();$('coverage-dialog').showModal();}));
  $('coverage-close').addEventListener('click',()=>$('coverage-dialog').close());
  $('coverage-dialog').addEventListener('click',e=>{if(e.target===$('coverage-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
  $('replay-play').addEventListener('click',toggleReplay);
  $('replay-scrub').addEventListener('input',e=>{pauseReplay();state.replayIndex=Number(e.target.value);state.selectedId=state.replayEvents[state.replayIndex-1]?.id;renderReplay();});
  if(recorded)prepareReplay();else{poll();setInterval(poll,3000);setInterval(renderConnection,1000);}
})();
