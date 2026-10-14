/** Per-target/per-metric integration. SentinelPort's numerical core is unchanged. */
import {SentinelPort, ENGINE_PROVENANCE} from './brain.mjs';

export const TARGET_PROFILE = 'target-metrics-v1';
export const TARGET_METRICS = Object.freeze({
  dns_ms:{label:'DNS lookup',unit:'ms'},tcp_ms:{label:'TCP connection',unit:'ms'},tls_ms:{label:'TLS handshake',unit:'ms'},
  response_wait_ms:{label:'HTTP response wait',unit:'ms'},headers_ms:{label:'First-hop response headers',unit:'ms'},
  failure_ratio:{label:'Failed probes · last 20',unit:'ratio'},
  icmp_rtt_ms:{label:'ICMP round trip',unit:'ms'},icmp_loss_ratio:{label:'ICMP probe loss',unit:'ratio'},
});
const number=value=>typeof value==='number'&&Number.isFinite(value)&&value>=0&&value<=86400000?value:null;
const date=value=>typeof value==='string'&&/(?:Z|[+-]\d\d:\d\d)$/.test(value)&&Number.isFinite(Date.parse(value))?Date.parse(value):null;
const clone=value=>JSON.parse(JSON.stringify(value));
const median=values=>{const sorted=[...values].sort((a,b)=>a-b),n=sorted.length;return n?(n%2?sorted[(n-1)/2]:(sorted[n/2-1]+sorted[n/2])/2):null;};

function restoreDetector(raw) {
  const detector=new SentinelPort();
  if(!raw||!Number.isSafeInteger(raw.t)||raw.t<0||raw.t>10000000)throw new Error('invalid_detector_checkpoint');
  for(const key of ['state','wOut','pDiag'])if(!Array.isArray(raw[key])||raw[key].length!==32||raw[key].some(x=>typeof x!=='number'||!Number.isFinite(x)||Math.abs(x)>1000000))throw new Error('invalid_detector_checkpoint');
  for(const key of ['residuals','rolling'])if(!Array.isArray(raw[key])||raw[key].length>(key==='rolling'?8:64)||raw[key].some(x=>typeof x!=='number'||!Number.isFinite(x)||Math.abs(x)>1000000))throw new Error('invalid_detector_checkpoint');
  for(const key of ['t','ema','eventScore','multiplier','collapse','freeze','episode','activeEpisode','episodeAge','cooldown','recoveries'])if(typeof raw[key]!=='number'||!Number.isFinite(raw[key])||Math.abs(raw[key])>10000000)throw new Error('invalid_detector_checkpoint');
  for(const key of Object.keys(detector.stats)){
    const stats=raw.stats?.[key];if(!stats||!Number.isSafeInteger(stats.count)||stats.count<0||![stats.mean,stats.m2].every(Number.isFinite))throw new Error('invalid_detector_checkpoint');
    Object.assign(detector.stats[key],{count:stats.count,mean:stats.mean,m2:stats.m2});
  }
  for(const key of ['state','wOut','pDiag','residuals','rolling','t','ema','eventScore','multiplier','collapse','freeze','episode','activeEpisode','episodeAge','cooldown','recoveries','previous'])detector[key]=clone(raw[key]);
  return detector;
}

export function targetMetricValues(event, failures=[]) {
  const p=event.probes ?? {},value=(probe,key)=>probe?.state==='ok'?number(probe[key]):null;
  return {dns_ms:value(p.dns,'duration_ms'),tcp_ms:value(p.tcp,'duration_ms'),tls_ms:value(p.tls,'duration_ms'),
    response_wait_ms:value(p.http,'response_wait_ms'),headers_ms:value(p.http,'headers_ms'),
    failure_ratio:failures.length?failures.reduce((a,b)=>a+b,0)/failures.length:null,
    icmp_rtt_ms:value(p.path,'rtt_ms'),icmp_loss_ratio:value(p.path,'loss_ratio')};
}

export class TargetBrain {
  constructor(saved=null) {
    this.data={profile:TARGET_PROFILE,collector_id:null,probe_options:null,last_at:null,last_id:null,samples:0,generation:1,
      series:{},last_facts:{},failure_window:[],seen:[],recent_ids:[],episode:null};
    this.checkpointIssue=null;
    if(saved){
      try{
        if(saved.profile!==TARGET_PROFILE||JSON.stringify(saved).length>90000||!Number.isSafeInteger(saved.samples)||saved.samples<0||!Number.isSafeInteger(saved.generation)||saved.generation<1||!Array.isArray(saved.seen)||saved.seen.length>512||!Array.isArray(saved.failure_window)||saved.failure_window.length>20||saved.failure_window.some(x=>x!==0&&x!==1))throw new Error('invalid_checkpoint');
        for(const [key,series] of Object.entries(saved.series ?? {})){
          if(!Object.hasOwn(TARGET_METRICS,key)||!Array.isArray(series.values)||series.values.length>32||series.values.some(x=>number(x)===null))throw new Error('invalid_checkpoint');
          restoreDetector(series.detector);
        }
        this.data=clone(saved);
      }catch{this.checkpointIssue='The retained baseline was incompatible or invalid; calibration restarted.';}
    }
  }
  observe(event) {
    if(!/^evt_[a-f0-9]{32}$/.test(event?.id ?? '')||date(event.completed_at)===null||event.probe_version!=='target-probes-v1')throw new Error('bounded_native_probe_required');
    const d=this.data,time=date(event.completed_at);
    const base={profile:TARGET_PROFILE,engine:ENGINE_PROVENANCE.engine,source:'authoritative_server',source_id:event.source_id,evidence_id:event.id,observed_at:event.completed_at,
      experimental:true,security_verdict:'not_assessed',forecast_claim:'Current-frame change detection; no incident prediction',signals:[],changes:[]};
    if(d.seen.includes(event.id))return {...base,state:'duplicate',reason:'Duplicate excluded; baseline unchanged.'};
    if(d.last_at!==null&&time<=date(d.last_at))return {...base,state:'late_excluded',reason:'Out-of-order evidence retained but excluded from baseline learning.'};
    const identity=JSON.stringify({collector:event.collector_id,options:event.probe_options});
    const previousIdentity=d.collector_id===null?null:JSON.stringify({collector:d.collector_id,options:d.probe_options});
    const gap=d.last_at===null?null:time-date(d.last_at);
    let reset=this.checkpointIssue;this.checkpointIssue=null;
    if(previousIdentity!==null&&(identity!==previousIdentity||gap>300000)){
      reset=identity!==previousIdentity?'Collector or probe settings changed; separate calibration started.':'A coverage gap exceeded five minutes; calibration restarted.';
      d.series={};d.last_facts={};d.failure_window=[];d.generation++;d.episode=null;
    }
    d.collector_id=event.collector_id;d.probe_options=clone(event.probe_options);d.samples++;
    d.failure_window.push(event.error?1:0);d.failure_window=d.failure_window.slice(-20);
    const values=targetMetricValues(event,d.failure_window),signals=[];
    for(const [key,value] of Object.entries(values)) {
      const definition=TARGET_METRICS[key];
      if(value===null){signals.push({metric:key,...definition,value:null,state:'unavailable',event_score:null,sample_count:d.series[key]?.detector?.t ?? 0,evidence_ids:[event.id]});continue;}
      const prior=d.series[key],detector=prior?restoreDetector(prior.detector):new SentinelPort(),history=prior?.values ?? [];
      const expected=median(history),result=detector.step(Math.log1p(value)),change=!result.warmup&&result.status!=='GREEN';
      signals.push({metric:key,...definition,value,prior_median:expected,delta:expected===null?null:value-expected,
        state:result.warmup?'calibrating':change?'unusual':'observing',sample_count:detector.t,warmup_remaining:Math.max(0,12-detector.t),
        event_score:result.warmup?null:result.event_score,evidence_ids:[...(history.length&&d.last_id?[d.last_id]:[]),event.id],baseline_evidence_ids:prior?.evidence_ids ?? [],
        meaning:'Experimental within-stream numerical change; not an attack probability'});
      d.series[key]={detector:clone(detector),values:[...history,value].slice(-16),evidence_ids:[...(prior?.evidence_ids ?? []),event.id].slice(-16)};
    }
    const p=event.probes,last=d.last_facts,changes=[];
    const fact=(name,current,label)=>{if(current!==null&&current!==undefined){if(last[name]!==undefined&&last[name]!==current)changes.push({code:name,label,evidence_ids:[d.last_id,event.id].filter(Boolean)});last[name]=current;}};
    if(p.dns.state==='ok')fact('dns_changed',JSON.stringify(p.dns.addresses),'DNS answer set changed');
    if(p.tls.state==='ok')fact('certificate_changed',p.tls.certificate?.fingerprint256,'TLS certificate changed; rotation can be expected');
    if(p.http.state==='ok'){fact('http_status_changed',event.status,'HTTP response status changed');fact('response_metadata_changed',p.http.header_signature,'Response header controls or server metadata changed');}
    if(p.page.state==='ok'&&p.page.sha256)fact('page_changed',JSON.stringify([p.page.sha256,p.page.bytes,p.page.complete]),'Measured page fingerprint changed; dynamic content may explain it');
    if(p.path?.state==='ok'&&Array.isArray(p.path.hops))fact('path_changed',JSON.stringify(p.path.hops),'Observed public network path changed; route visibility can vary');
    const expiry=date(p.tls.certificate?.valid_to);
    if(p.tls.state==='ok'&&expiry!==null&&(expiry-time)/86400000<=14)changes.push({code:'certificate_expiring',label:'Certificate expires within 14 days',days_remaining:Math.max(0,(expiry-time)/86400000),evidence_ids:[event.id]});
    if(event.error)changes.push({code:'probe_failed',label:'This observation failed; target-wide availability is unconfirmed',evidence_ids:[event.id]});
    const unusual=signals.filter(x=>x.state==='unusual'),codes=[...changes.map(x=>x.code),...unusual.map(x=>`${x.metric}_change`)];
    if(codes.length){
      if(!d.episode||time-date(d.episode.last_at)>60000)d.episode={id:`episode_${event.id.slice(4)}`,started_at:event.completed_at,last_at:event.completed_at,evidence_ids:[],codes:[]};
      d.episode.last_at=event.completed_at;d.episode.evidence_ids=[...new Set([...d.episode.evidence_ids,...changes.flatMap(x=>x.evidence_ids),...unusual.flatMap(x=>x.evidence_ids)])].slice(-24);
      d.episode.codes=[...new Set([...d.episode.codes,...codes])].slice(-16);
    } else if(d.episode&&time-date(d.episode.last_at)>60000)d.episode=null;
    d.last_at=event.completed_at;d.last_id=event.id;d.seen=[...d.seen,event.id].slice(-512);d.recent_ids=[...d.recent_ids,event.id].slice(-20);
    return {...base,state:codes.length?'change_detected':signals.some(x=>x.state==='calibrating')?'calibrating':'observing',signals,changes,
      episode:clone(d.episode),generation:d.generation,history_samples:d.samples,calibration_reset:reset,
      coverage_gap_ms:gap,collector_id:d.collector_id,baseline_scope:'Separate metric, target, probe settings and collector identity',
      missing_data_policy:'Unmeasured values excluded; failed observations counted separately; no gap filled with zero'};
  }
  checkpoint(){return clone(this.data);}
}
