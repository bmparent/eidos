/** Coordinated views of the selected endpoint's actual probes and server analysis. */
const labels={dns_ms:'DNS lookup',tcp_ms:'TCP connection',tls_ms:'TLS handshake',response_wait_ms:'HTTP response wait',headers_ms:'Response headers',failure_ratio:'Failed probes',icmp_rtt_ms:'ICMP round trip',icmp_loss_ratio:'ICMP probe loss'};
const escape=value=>String(value ?? '').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const numeric=value=>typeof value==='number'&&Number.isFinite(value);
const reading=(value,metric)=>!numeric(value)?'—':metric.endsWith('ratio')?`${(value*100).toFixed(1)}%`:`${value.toFixed(1)} ms`;
let metric='tcp_ms',mounted=false,pathBusy=false;

export function renderTargetPanel(container,{event,events=[],capabilities,onPathResult}={}) {
  if(!container)return;
  if(!mounted){
    container.innerHTML='<div class="section-heading"><div><h2 id="target-probes-title">Inside this connection</h2><p id="target-probes-subtitle">Choose an observation to inspect its measured stages.</p></div><span class="small-badge">TARGET PROBES</span></div><div id="target-probe-grid" class="target-probe-grid"></div><div class="probe-chart-heading"><label for="probe-metric">Follow a measurement</label><select id="probe-metric">'+Object.entries(labels).map(([key,label])=>`<option value="${key}">${label}</option>`).join('')+'</select><span id="probe-chart-period"></span></div><div class="probe-chart-scroll"><svg id="probe-chart" viewBox="0 0 1120 130" role="img" aria-label="No target measurements yet"></svg></div><div id="target-probe-details" class="target-probe-details"></div><div class="path-controls"><div><strong>Network path</strong><p id="path-availability">Ping and route tracing require the local companion.</p></div><button type="button" class="quiet-button" id="path-ping">Measure ping</button><button type="button" class="quiet-button" id="path-trace">Trace route</button></div><p id="path-result" role="status"></p>';
    container.querySelector('#probe-metric').value=metric;mounted=true;
  }
  const query=id=>container.querySelector('#'+id),analysis=event?.target_analysis,p=event?.probes;
  query('probe-metric').onchange=change=>{metric=change.target.value;renderTargetPanel(container,{event,events,capabilities,onPathResult});};
  query('target-probes-subtitle').textContent=event?`${event.target || 'Selected endpoint'} · ${new Date(event.completed_at).toLocaleTimeString()} · ${event.target_storage==='retained'?'retained server evidence':'measurement evidence'}`:'Connect a target. Every displayed value will come from its actual observation.';
  query('target-probe-grid').innerHTML=Object.entries(labels).filter(([key])=>!key.startsWith('icmp_')||capabilities?.mode==='local').map(([key,label])=>{
    const signal=analysis?.signals?.find(x=>x.metric===key),state=signal?.state ?? 'waiting';
    return `<article class="probe-reading ${escape(state)}"><span>${label}</span><strong>${reading(signal?.value,key)}</strong><small>${escape(state.replaceAll('_',' '))}${signal?.warmup_remaining?` · ${signal.warmup_remaining} samples to calibrate`:''}</small><small>Prior median ${reading(signal?.prior_median,key)}</small></article>`;
  }).join('');
  const rows=events.filter(row=>row.source_id===event?.source_id).slice(0,128).reverse(),values=rows.map(row=>({row,signal:row.target_analysis?.signals?.find(x=>x.metric===metric)}));
  const measured=values.filter(x=>numeric(x.signal?.value)),chart=query('probe-chart');
  if(measured.length){
    const first=Date.parse(rows[0].completed_at),last=Date.parse(rows.at(-1).completed_at),peak=Math.max(0.001,...measured.map(x=>x.signal.value));
    const x=row=>45+(Date.parse(row.completed_at)-first)/Math.max(1,last-first)*1025,y=value=>106-value/peak*82;
    const parts=[`<path d="M45 107H1080" stroke="#304552"/><text x="8" y="25" fill="#98acb7" font-size="10">${escape(reading(peak,metric))}</text><text x="8" y="110" fill="#98acb7" font-size="10">0</text>`];
    let previous=null;
    for(const item of values){
      if(!numeric(item.signal?.value)){previous=null;continue;}
      if(previous&&Date.parse(item.row.completed_at)-Date.parse(previous.row.completed_at)<=60000)parts.push(`<path d="M${x(previous.row)} ${y(previous.signal.value)}L${x(item.row)} ${y(item.signal.value)}" stroke="#7ebec2" fill="none" opacity=".55"/>`);
      parts.push(`<circle cx="${x(item.row)}" cy="${y(item.signal.value)}" r="${item.row.id===event.id?5:3}" fill="${item.signal.state==='unusual'?'#efbc7f':'#b4eedc'}"><title>${escape(reading(item.signal.value,metric))} · ${escape(item.row.completed_at)}</title></circle>`);previous=item;
    }
    chart.innerHTML=parts.join('');chart.setAttribute('aria-label',`${labels[metric]}, ${measured.length} measured samples for this target. Gaps are not filled.`);
    query('probe-chart-period').textContent=`${measured.length} measured samples · gaps remain visible`;
  }else{chart.innerHTML='<text x="45" y="70" fill="#98acb7" font-size="15">No measured samples for this stream yet.</text>';chart.setAttribute('aria-label','No measured samples for this stream');query('probe-chart-period').textContent='Awaiting measurements';}
  if(p){
    const cert=p.tls?.certificate,changes=analysis?.changes ?? [];
    query('target-probe-details').innerHTML=`<article><h3>DNS & certificate</h3><p>DNS: ${escape(p.dns?.state)}${p.dns?.addresses?.length?` · ${escape(p.dns.addresses.join(', '))}`:''}</p><p>TLS: ${escape(p.tls?.state)}${p.tls?.protocol?` · ${escape(p.tls.protocol)}`:''}</p>${cert?`<p>Certificate expires ${escape(cert.valid_to ?? 'unknown')}<br>Issuer: ${escape(cert.issuer ?? 'unknown')}</p>`:''}</article><article><h3>Response & page</h3><p>${(p.http?.chain ?? []).map(row=>`HTTP ${row.status}`).join(' → ') || 'No HTTP response measured'}</p><p>Page fingerprint: ${escape(p.page?.state)}${numeric(p.page?.bytes)?` · ${p.page.bytes.toLocaleString()} bytes · ${p.page.complete?'complete body':'bounded prefix'}`:''}</p>${p.page?.sha256?`<code class="probe-fingerprint">${escape(p.page.sha256)}</code>`:''}${p.http?.chain_state?`<p>Redirect coverage: ${escape(p.http.chain_state.replaceAll('_',' '))}</p>`:''}</article><article><h3>Brain / Sentinel episode</h3><p>${escape(analysis?.state ?? 'unavailable')}${analysis?.history_samples?` · ${analysis.history_samples} retained baseline observations`:''}</p>${analysis?.calibration_reset?`<p>${escape(analysis.calibration_reset)}</p>`:''}${changes.length?'<ul>'+changes.map(change=>`<li>${escape(change.label)}</li>`).join('')+'</ul>':'<p>No explicit change event in this observation.</p>'}${analysis?.episode?`<p>${analysis.episode.evidence_ids.length} linked evidence records in this episode.</p>`:''}<p class="evidence-hint">Changes need interpretation. Certificate rotation, DNS rotation and dynamic content can be expected.</p></article>`;
  }else query('target-probe-details').innerHTML='<p class="empty-record">The selected observation predates the richer target probes. Its original evidence remains available.</p>';
  const local=capabilities?.mode==='local'&&capabilities?.target_probes?.path==='local_commands';
  query('path-availability').textContent=local?'Manual diagnostics from this companion. Results join the next collected observation.':'Ping and route tracing are available through the Node local companion.';
  for(const [id,kind] of [['path-ping','ping'],['path-trace','trace']]){
    const button=query(id);button.disabled=!local||!event||pathBusy;
    button.onclick=async()=>{
      if(pathBusy||!event)return;pathBusy=true;button.disabled=true;query('path-result').textContent='Measuring the selected target’s public network path…';
      try{const response=await fetch('/api/path-probe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target:event.url,kind}),signal:AbortSignal.timeout(16000)});const data=await response.json();if(!response.ok)throw new Error(data.message ?? 'Path diagnostic unavailable');
        query('path-result').textContent=data.result.state==='ok'?kind==='ping'?`Ping: ${data.result.rtt_ms ?? 'unknown'} ms average; ${data.result.loss_ratio===null?'unknown':(data.result.loss_ratio*100).toFixed(0)+'%'} probe loss. Joins the next observation.`:`Trace: ${data.result.hops?.length ?? 0} public hops observed. Joins the next observation.`:`Path diagnostic ${data.result.state}: ${data.result.reason ?? 'no usable measurement'}`;onPathResult?.(data);
      }catch(error){query('path-result').textContent=error.message;}finally{pathBusy=false;button.disabled=false;}
    };
  }
}
