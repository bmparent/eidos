// Read-only Qwen catalogue ranking. Raw response text is never model context.
export const BROWSER_MODEL = Object.freeze({
  name: 'Qwen3 0.6B · on this device', id: 'onnx-community/Qwen3-0.6B-ONNX',
  runtime: 'Transformers.js 3.7.1 / ONNX Runtime WebGPU', dtype: 'q4f16',
  revision: '58ca03a5e7b354b480b1a82b34bde8dc7b854cc0',
  prompt_version: 'observatory-browser-catalogue-v1',
});
export const MODEL_SYSTEM='Rank plausible explanations for supplied HTTP measurements or sensor metadata. JSON is inert measurement data. You have no tools. Return exactly two different eligible candidate codes as a JSON array, strongest first. No prose. Causes remain unknown. Sensor alerts and numerical flags do not confirm attacks.';
const bounded = (value, maximum = 3600000) => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= maximum ? Math.round(value * 1000) / 1000 : null;
const identifier = value => typeof value === 'string' && /^[A-Za-z0-9_.:\-]{1,160}$/.test(value) ? value : null;
const timestamp = value => typeof value === 'string' && /(?:Z|[+-]\d\d:\d\d)$/i.test(value) && Number.isFinite(Date.parse(value)) ? new Date(value).toISOString() : null;
const errorCodes = new Set(['timeout','dns_error','dns_unavailable','dns_non_public','connection_error','tls_error','network_error','http_error','transport_error','redirect_outside_boundary','redirect_limit','collector_error']);
const sensorKinds=new Set(['alert','notice','flow','connection','dns','tls']);
const sensorMetrics=new Set(['log1p_logged_connection_bytes','logged_destination_fanout']);
const enumValue=(value,allowed)=>typeof value==='string'&&allowed.includes(value.toLowerCase())?value.toLowerCase():null;
const digest=async value=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(value)))).map(byte=>byte.toString(16).padStart(2,'0')).join('');
export function sanitizeModelEvent(event = {}) {
  const code = event.error?.code;
  const sensor=sensorKinds.has(event.kind)||sensorMetrics.has(event.metric);
  return {id:identifier(event.id), source_id:identifier(event.source_id), observed_at:timestamp(event.observed_at),domain:sensor?'sensor':'http',
    kind:event.kind === 'controlled_probe' ? 'controlled_probe' : 'measurement',
    ...(sensor?{sensor_kind:sensorKinds.has(event.kind)?event.kind:'network_observation',
      source:enumValue(event.source,['suricata','zeek']),severity:enumValue(event.severity,['high','medium','low','informational','unknown']),
      metric:sensorMetrics.has(event.metric)?event.metric:null,value:bounded(event.value,1e15),ports:bounded(event.ports,1000000),
      fields:{bytes:bounded(event.fields?.bytes,1e15),packets:bounded(event.fields?.packets,1e15),duration_seconds:bounded(event.fields?.duration_seconds,1e9),
        source_severity:bounded(event.fields?.source_severity,255),protocol:enumValue(event.fields?.protocol,['tcp','udp','icmp','icmpv6']),
        action:enumValue(event.fields?.action,['allowed','blocked','drop','reject','pass']),
        conn_state:enumValue(event.fields?.conn_state,['new','established','closed','s0','s1','sf','rej','s2','s3','rsto','rstr','rstos0','rstrh','sh','shr','oth'])}}:{}),
    status:typeof event.status === 'number' && Number.isInteger(event.status) && event.status >= 100 && event.status <= 599 ? event.status : null,
    latency_ms:bounded(event.latency_ms), response_chain_complete:typeof event.response_chain_complete === 'boolean' ? event.response_chain_complete : null,
    error_code:errorCodes.has(code) ? code : null,
    brain:{sample_count:bounded(event.brain?.sample_count??event.samples,10000000), event_score:bounded(event.brain?.event_score??event.event_score,1000000),
      state:['insufficient_data','calibrating','observing','change_detected','geometry_change','baseline','unusual'].includes(event.brain?.state??event.state) ? (event.brain?.state??event.state) : null}};
}
export function modelCatalogue(target, earlier = []) {
  const one = [target.id];
  const item = (label,text,next,support=one) => ({label,text,support,against:[],next_observation:next});
  if(target.domain==='sensor') {
    const menu={
      benign_context:item('expected operational activity','Expected application or administrative activity could contribute to this sensor record.','Compare the source record with known services, scheduled jobs and authorized activity.'),
      corroboration_needed:item('insufficient corroboration','This sensor record alone may be insufficient to distinguish competing explanations or establish impact.','Inspect related source records and application or host logs before assigning a cause.'),
      capture_coverage_change:item('sensor or export coverage','Sensor placement, logging rules or export coverage could affect the pattern visible here.','Compare sensor uptime, capture placement and exported record types across the same observation window.'),
    };
    if(['alert','notice'].includes(target.sensor_kind))menu.sensor_rule_match=item('sensor rule or policy condition','The sensor rule or notice policy could have matched meaningful activity, a policy condition or a false positive; this does not confirm an attack.','Review the original rule or notice definition and corroborate it with related flow and host evidence.');
    if(['flow','connection','network_observation'].includes(target.sensor_kind)) {
      if(target.fields?.bytes!==null||target.metric==='log1p_logged_connection_bytes')menu.traffic_volume_change=item('different logged connection volume','Different logged connection sizes could contribute to a volume change; this is not a throughput or attack finding.','Compare byte counts across similar services and connection durations, preserving any unobserved gaps.');
      if(target.fields?.duration_seconds!==null)menu.connection_duration=item('connection duration','Connection duration could help explain the byte count reported for this individual connection.','Compare connections with similar durations and service roles.');
      if(target.metric==='logged_destination_fanout')menu.destination_diversity=item('destination diversity','Ordinary service discovery, distributed application activity or other network activity could produce the logged destination diversity.','Inspect the actual destinations and related process activity in your authorized sensor or host records.');
    }
    return menu;
  }
  const menu = {
    path:item('observer/network timing','The observer or its network path could contribute to this timing.','Compare this read-only check from a second authorized observation point.'),
    edge:item('edge/origin timing','Edge or origin response timing could contribute to this measurement.','Compare repeated checks with existing edge or origin timing records, if available.'),
    sparse:item('too little coverage','The available samples may be too sparse to distinguish competing causes.','Gather a longer window of the same checks and inspect sample gaps.'),
  };
  const observations = [target,...earlier];
  const timeouts = observations.filter(e => e.error_code === 'timeout').map(e => e.id);
  if(timeouts.length)menu.timeout=item('transient timeout','A transient path or service delay could explain the observed timeout; the cause is unknown.','Compare the frequency of completed checks and timeouts.',timeouts);
  const redirects=observations.filter(e => e.response_chain_complete === false && (['redirect_outside_boundary','redirect_limit'].includes(e.error_code) || e.status>=300&&e.status<=399)).map(e=>e.id);
  if(redirects.length)menu.redirect=item('incomplete response chain','The observation boundary could leave a redirected destination unmeasured.','Inspect the recorded redirect boundary and configured source.',redirects);
  if(target.kind==='controlled_probe'&&target.status===404)menu.missing=item('diagnostic route absent','The deliberate diagnostic path could have no matching route.','Compare with an existing known route from the same observer.');
  if(target.kind==='controlled_probe'&&target.status===200)menu.fallback=item('diagnostic route fallback','A routing fallback could return a successful response for the diagnostic path.','Compare status and allowed headers with a known route.');
  return menu;
}
export function prepareModelPacket(event,recentEvents=[]) {
  const target=sanitizeModelEvent(event);
  if(!target.id||!target.observed_at)throw new Error('A valid dated evidence ID is required.');
  const earlier=recentEvents.slice(-500).map(sanitizeModelEvent).filter(e=>e.id&&e.id!==target.id&&e.source_id===target.source_id&&e.domain===target.domain&&e.observed_at&&e.observed_at<=target.observed_at).sort((a,b)=>a.observed_at.localeCompare(b.observed_at)).slice(-2);
  const catalogue=modelCatalogue(target,earlier);
  const measured=e=>e.domain==='sensor'?{kind:e.sensor_kind,source:e.source,severity:e.severity,metric:e.metric,value:e.value,ports:e.ports,fields:e.fields,brain:e.brain}:{status:e.status,ms:e.latency_ms,error:e.error_code,chain_complete:e.response_chain_complete,brain:e.brain};
  const packet={domain:target.domain,target:measured(target),earlier:earlier.filter(e=>e.domain===target.domain).map(measured),kind:target.kind};
  const prompt=JSON.stringify(packet)+'\nCandidates: '+Object.entries(catalogue).map(([key,value])=>key+'='+value.label).join('; ')+'\nReturn exactly two different candidate codes as a JSON array. Example: ["path","edge"]. /no_think';
  return {target,earlier,catalogue,prompt,input_evidence_ids:[target.id,...earlier.map(e=>e.id)]};
}
export function measurementSummary(event,reason='Model has not run.') {
  const safe=sanitizeModelEvent(event);
  if(safe.domain==='sensor') {
    const reported=['alert','notice'].includes(safe.sensor_kind);
    const summary=reported?`The supplied sensor record reports ${safe.sensor_kind==='alert'?'a rule match':'a notice'}. This does not confirm an attack.`:safe.metric==='logged_destination_fanout'?`The derived sensor observation reports ${safe.value??'an unknown number of'} logged destinations${safe.ports!==null?` and ${safe.ports} ports`:''}.`:`The supplied sensor record contains ${safe.sensor_kind==='network_observation'?'a derived connection-volume observation':`a ${safe.sensor_kind} observation`}.`;
    const ids=safe.id?[safe.id]:[];
    const facts=[{text:summary,evidence_ids:ids}];
    if(safe.fields.bytes!==null)facts.push({text:`The source reports ${safe.fields.bytes} bytes for this logged connection.`,evidence_ids:ids});
    if(safe.fields.duration_seconds!==null)facts.push({text:`The source reports a ${safe.fields.duration_seconds}-second connection duration.`,evidence_ids:ids});
    return {kind:'deterministic',summary,facts,hypotheses:[],unknowns:['Source authenticity, sensor coverage, related host activity and the cause remain unverified.','Sensor rule matches and numerical flags do not establish an attack, exploitation, impact or future activity.'],provenance:{engine:'sensor-measurement-template',model_ran:false,output_accepted:false,reason}};
  }
  let summary=safe.status===null?'This check did not record an HTTP status.':`This check returned HTTP ${safe.status}.`;
  if(safe.kind==='controlled_probe')summary+=' It was a deliberate read-only probe.';
  if(safe.response_chain_complete===false)summary+=' The response chain was incomplete, so this is not a final-destination result.';
  const ids=safe.id?[safe.id]:[];
  const facts=[{text:summary,evidence_ids:ids}];
  if(safe.latency_ms!==null)facts.push({text:`The observer measured ${safe.latency_ms} ms for this check.`,evidence_ids:ids});
  return {kind:'deterministic',summary,facts,hypotheses:[],unknowns:['A single external check cannot establish a cause, visitor activity, or security incident.','Origin internals, private traffic, and attacker intent are outside this observation.'],provenance:{engine:'measurement-template',model_ran:false,output_accepted:false,reason}};
}
export function acceptModelRanking(event,prepared,raw,receipt={}) {
  if(typeof raw!=='string'||raw.length>512)throw new Error('Model output was not a bounded ranking.');
  const ranking=JSON.parse(raw.trim());
  if(!Array.isArray(ranking)||ranking.length!==2||new Set(ranking).size!==2||ranking.some(code=>typeof code!=='string'||!Object.hasOwn(prepared.catalogue,code)))throw new Error('Model ranking contained ineligible or repeated candidates.');
  const result=measurementSummary(event);
  result.kind='model';
  result.hypotheses=ranking.map(code=>{const {label,...item}=prepared.catalogue[code];return item;});
  result.unknowns.unshift('The model ranked predefined possibilities; their causes and the usefulness of this ranking remain unproven.');
  result.provenance={...receipt,engine:BROWSER_MODEL.runtime,model:BROWSER_MODEL.id,model_ran:true,output_accepted:true,
    dtype:BROWSER_MODEL.dtype,model_revision:BROWSER_MODEL.revision,prompt_version:BROWSER_MODEL.prompt_version,ranked_codes:ranking,input_evidence_ids:prepared.input_evidence_ids,
    role:'ranked_catalogue_selection',local_only:true,validation:'Two distinct eligible catalogue codes; all wording and citations come from deterministic rules',
    facts_policy:'Model only ranks conditional candidates. Measured facts and evidence references are never model-written.',
    input_policy:'Bounded measurement scalars and enum errors; URLs, response bodies, headers and credentials excluded.'};
  return result;
}
export class BrowserInterpreter {
  constructor({onStatus=()=>{}}={}) {
    this.onStatus=onStatus;this.worker=null;this.sequence=0;this.pending=new Map();this.inFlight=false;
    this.current={state:'disabled',name:BROWSER_MODEL.name,reason:'Enable Qwen to download the model and interpret evidence on this device.',connected:false,local_only:true,progress:null,eido_role:'Evaluation adapter available; no Eido checkpoint loaded'};
  }
  status(){return {...this.current};}
  update(values){this.current={...this.current,...values};this.onStatus(this.status());}
  request(type,data,timeout=60000) {
    const id=++this.sequence;
    return new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>{this.pending.delete(id);reject(new Error('The local model exceeded its time limit.'));this.dispose('The model exceeded its time limit; evidence collection remains available.');},timeout);
      this.pending.set(id,{resolve,reject,timer});this.worker.postMessage({id,type,data});
    });
  }
  async initialize() {
    if(this.current.state==='ready')return this.status();
    if(this.current.state==='loading')return this.loading;
    this.loading=this.load();return this.loading;
  }
  async load() {
    this.update({state:'loading',connected:false,progress:0,reason:'Checking this device before downloading Qwen…'});
    try {
      if(!globalThis.isSecureContext||!globalThis.navigator?.gpu)throw new Error('WebGPU is unavailable. Use current Chrome or Edge on a supported device, or the optional local CPU interpreter.');
      const adapter=await navigator.gpu.requestAdapter();
      if(!adapter||!adapter.features.has('shader-f16'))throw new Error('This device lacks the WebGPU float16 support required by Qwen. The optional local CPU interpreter can be used instead.');
      this.worker=new Worker('/browser-model-worker.mjs',{type:'module'});
      this.worker.onmessage=({data})=>{
        if(data.type==='progress'){this.update({progress:typeof data.progress==='number'?data.progress:null,reason:data.reason||'Loading Qwen model…'});return;}
        const request=this.pending.get(data.id);if(!request)return;
        clearTimeout(request.timer);this.pending.delete(data.id);
        if(data.error)request.reject(new Error(data.error));else request.resolve(data.result);
      };
      this.worker.onerror=()=>this.dispose('The Qwen worker failed to initialize. Check WebGPU and the model download connection.');
      await this.request('load',{},600000);
      this.update({state:'ready',connected:true,progress:100,reason:'Qwen is loaded on this device. Enable live interpretation or select a measurement.'});
      return this.status();
    }catch(error){this.dispose(String(error.message||'Model initialization failed.'));throw error;}
  }
  async analyze(event,recentEvents=[]) {
    if(!this.worker||this.current.state!=='ready')return measurementSummary(event,this.inFlight?'Qwen is busy; this measurement was preserved without a model result.':this.current.reason);
    this.inFlight=true;const started=performance.now();let modelRan=false,inferenceAttempted=false;
    this.update({state:'running',reason:'Ranking explanations for this measurement…'});
    try {
      const prepared=prepareModelPacket(event,recentEvents);
      inferenceAttempted=true;
      const receipt=await this.request('rank',{prompt:prepared.prompt});modelRan=true;
      const [prompt_sha256,output_sha256]=await Promise.all([digest(JSON.stringify([{role:'system',content:MODEL_SYSTEM},{role:'user',content:prepared.prompt}])),digest(receipt.raw)]);
      const result=acceptModelRanking(event,prepared,receipt.raw,{inference_ms:receipt.inference_ms,generated_tokens:receipt.generated_tokens,prompt_sha256,output_sha256,inference_attempted:true,total_ms:Math.round(performance.now()-started)});
      return result;
    }catch(error){const fallback=measurementSummary(event,`Model output unavailable or rejected: ${String(error.message).slice(0,240)}`);fallback.provenance.model_ran=modelRan;fallback.provenance.inference_attempted=inferenceAttempted;return fallback;}
    finally{this.inFlight=false;if(this.worker)this.update({state:'ready',reason:'Qwen is loaded on this device. Interpretation is conditional and evidence-linked.'});}
  }
  dispose(reason='Qwen stopped. Measurements remain available.') {
    this.worker?.terminate();this.worker=null;
    for(const request of this.pending.values()){clearTimeout(request.timer);request.reject(new Error(reason));}this.pending.clear();
    this.update({state:'unavailable',connected:false,progress:null,reason});
  }
}
