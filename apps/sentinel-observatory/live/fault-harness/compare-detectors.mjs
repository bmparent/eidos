import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve,join} from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {TargetBrain} from '../target-brain.mjs';
const configBytes=readFileSync(new URL('detector-protocol.json',import.meta.url)),config=JSON.parse(configBytes),sha=x=>createHash('sha256').update(x).digest('hex');
const lock=JSON.parse(readFileSync(new URL('detector-lock.json',import.meta.url)));assert.equal(sha(configBytes),lock.sha256,'Detector protocol changed');
const out=resolve(process.argv[process.argv.indexOf('--out')+1]??'artifacts/proof_runs/2026-10-18/observatory-m1/detectors');mkdirSync(out,{recursive:true});
const rng=seed=>()=>{seed=(1664525*seed+1013904223)>>>0;return seed/4294967296;};
const events=flags=>{const list=[];let start=null;for(let i=0;i<=flags.length;i++){if(flags[i]===true&&start===null)start=i;if(flags[i]!==true&&start!==null){list.push([start,i]);start=null;}}return list;};
const percent=(a,b)=>b?a/b:null;
function score(flags,labels){
  const predicted=events(flags),actual=events(labels),matched=new Set(),delays=[],recoveries=[];let tp=0,fp=0,duplicates=0;
  for(const [a,b] of predicted){const index=actual.findIndex(([c,d],i)=>!matched.has(i)&&a<d&&b>c);if(index>=0){matched.add(index);tp++;delays.push(Math.max(0,a-actual[index][0])*config.cadence_seconds);recoveries.push(Math.max(0,b-actual[index][1])*config.cadence_seconds);}else{fp++;if(actual.some(([c,d])=>a<d&&b>c))duplicates++;}}
  const benign=labels.filter((v,i)=>!v&&flags[i]!==null).length,unknown=flags.filter(v=>v===null).length;
  return {tp,fp,fn:actual.length-tp,duplicate_incidents:duplicates,fault_events:actual.length,benign_frames:benign,unknown_frames:unknown,measured_frames:flags.length-unknown,detection_delay_seconds:delays,recovery_delay_seconds:recoveries};
}
const mean=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:null;
function aggregate(rows){const total={tp:0,fp:0,fn:0,duplicate_incidents:0,fault_events:0,benign_frames:0,unknown_frames:0,measured_frames:0};for(const row of rows)for(const key in total)total[key]+=row[key];
  return {...total,precision:percent(total.tp,total.tp+total.fp),recall:percent(total.tp,total.tp+total.fn),false_incidents_per_benign_target_day:percent(total.fp,total.benign_frames*config.cadence_seconds/86400),missing_fraction:percent(total.unknown_frames,total.measured_frames+total.unknown_frames),mean_detection_delay_seconds:mean(rows.flatMap(x=>x.detection_delay_seconds)),mean_recovery_delay_seconds:mean(rows.flatMap(x=>x.recovery_delay_seconds))};}
const rows=[],streams=[],timings=Object.fromEntries(config.methods.map(m=>[m,0]));
for(const family of config.families)for(const seed of config.seeds){
  const random=rng(seed),values=[],labels=[];
  for(let i=0;i<config.frames;i++){
    let value=20+(random()-0.5)*(family==='jitter'?4:1),label=false;
    const active=config.fault_windows.some(([a,b])=>i>=a&&i<b);
    if(['step_up','recovery'].includes(family)&&active){value+=30;label=true;}
    if(family==='step_down'&&active){value-=15;label=true;}
    if(family==='spike'&&[60,110].includes(i)){value+=50;label=true;}
    if(family==='ramp'&&active){value+=(i%50)*1.5;label=true;}
    if(family==='drift'&&i>=60){value+=(i-59)*0.4;label=true;}
    if(family==='burst'&&active){value+=i%2?35:20;label=true;}
    if(family==='seasonal')value+=3*Math.sin(i/4);
    if(family==='missing'&&active)value=null;
    values.push(value);labels.push(label);
  }
  const flags=Object.fromEntries(config.methods.map(m=>[m,[]]));let brain=new TargetBrain(),cal=[],gateHistory=[],mu=null,sigma=null,ewma=null,positive=0,negative=0,n=0;
  const id=`${family}-${seed}`,split=config.evaluation_seeds.includes(seed)?'locked_evaluation':'development';
  for(let i=0;i<values.length;i++){
    if(family==='reset'&&i===80){brain=new TargetBrain();cal=[];gateHistory=[];mu=null;sigma=null;ewma=null;positive=negative=n=0;}
    const value=values[i];if(value===null){for(const method of config.methods)flags[method].push(null);continue;}
    const before=performance.now(),time=new Date(Date.UTC(2026,9,18)+i*config.cadence_seconds*1000).toISOString();
    const event={id:'evt_'+(i+1).toString(16).padStart(32,'0'),source_id:'target_0123456789abcdef',probe_version:'target-probes-v1',collector_id:'node-pinned-target-v1:synthetic',probe_options:{},completed_at:time,status:200,error:null,
      probes:{dns:{state:'unavailable',addresses:[]},tcp:{state:'ok',duration_ms:value},tls:{state:'not_applicable',certificate:null},http:{state:'unavailable'},page:{state:'disabled'}}};
    const signal=brain.observe(event).signals.find(x=>x.metric==='tcp_ms');timings.sentinel_plus_gate+=performance.now()-before;
    const warm=cal.length<config.warmup;
    const gateStart=performance.now(),median=a=>{const s=[...a].sort((a,b)=>a-b),n=s.length;return n?n%2?s[(n-1)/2]:(s[n/2-1]+s[n/2])/2:null;};
    const expected=median(gateHistory),mad=expected===null?null:median(gateHistory.map(x=>Math.abs(x-expected))),threshold=mad===null?null:Math.max(5,Math.abs(expected)*0.25,6*1.4826*mad);
    const gateAlarm=expected!==null&&Math.abs(value-expected)>threshold;gateHistory=[...gateHistory,value].slice(-16);timings.gate+=performance.now()-gateStart;
    assert.equal(gateAlarm,signal.measured_deviation,'Standalone gate must equal the actual operational gate');
    flags.gate.push(warm?null:gateAlarm);flags.sentinel_plus_gate.push(warm?null:signal.state==='unusual');
    if(warm){cal.push(value);flags.ewma.push(null);flags.cusum.push(null);if(cal.length===config.warmup){mu=mean(cal);sigma=Math.max(0.5,Math.sqrt(cal.reduce((s,v)=>s+(v-mu)**2,0)/(cal.length-1)));ewma=mu;}continue;}
    const ewmaStart=performance.now();n++;ewma=config.ewma.alpha*value+(1-config.ewma.alpha)*ewma;
    const limit=config.ewma.z*sigma*Math.sqrt(config.ewma.alpha/(2-config.ewma.alpha)*(1-(1-config.ewma.alpha)**(2*n)));
    flags.ewma.push(Math.abs(ewma-mu)>limit);timings.ewma+=performance.now()-ewmaStart;
    const cusumStart=performance.now(),z=(value-mu)/sigma;positive=Math.max(0,positive+z-config.cusum.k);negative=Math.max(0,negative-z-config.cusum.k);
    const alarm=Math.max(positive,negative)>config.cusum.h;flags.cusum.push(alarm);if(alarm)positive=negative=0;timings.cusum+=performance.now()-cusumStart;
  }
  streams.push({id,family,seed,split,values,labels,flags});for(const method of config.methods)rows.push({id,family,seed,split,method,...score(flags[method],labels)});
}
const result={schema:'detector-results-v1',evidence_class:'synthetic_engineering',config_sha256:sha(configBytes),streams:streams.length,frames:streams.length*config.frames,methods:{},limits:['Synthetic timing labels, not attack labels or pilot outcomes.','Sentinel timings include the multi-metric adapter; simple gate is standalone median/MAD and excludes integration overhead.','Recovery delay is event end beyond known fault end; alarms stopping early can report zero. An event at end of stream is right-censored, not a verified recovery.','Bootstrap units are whole streams; repeated related generators limit external validity.','No threshold tuning after this lock; core and production default unchanged.']};
for(const method of config.methods){result.methods[method]={observed_adapter_ms:timings[method],splits:{}};for(const split of ['development','locked_evaluation']){
  const subset=rows.filter(x=>x.method===method&&x.split===split),point=aggregate(subset),random=rng(42),samples=[];
  for(let n=0;n<config.confidence.resamples;n++)samples.push(aggregate(subset.map(()=>subset[Math.floor(random()*subset.length)])));
  const ci={};for(const key of ['precision','recall','false_incidents_per_benign_target_day','mean_detection_delay_seconds','mean_recovery_delay_seconds']){const list=samples.map(x=>x[key]).filter(x=>x!==null).sort((a,b)=>a-b);ci[key]=list.length?[list[Math.floor(list.length*0.025)],list[Math.min(list.length-1,Math.floor(list.length*0.975))]]:null;}
  result.methods[method].splits[split]={...point,stream_cluster_ci95:ci};
}}
const evalRows=result.methods,gate=evalRows.gate.splits.locked_evaluation,sentinel=evalRows.sentinel_plus_gate.splits.locked_evaluation;
result.conclusion={sentinel_precision_minus_gate:sentinel.precision===null||gate.precision===null?null:sentinel.precision-gate.precision,sentinel_recall_minus_gate:sentinel.recall===null||gate.recall===null?null:sentinel.recall-gate.recall,superiority_established:false,production_default_changed:false};
writeFileSync(join(out,'streams.jsonl'),streams.map(x=>JSON.stringify(x)).join('\n')+'\n');writeFileSync(join(out,'stream_scores.jsonl'),rows.map(x=>JSON.stringify(x)).join('\n')+'\n');
writeFileSync(join(out,'comparison.json'),JSON.stringify(result,null,2)+'\n');writeFileSync(join(out,'config.json'),configBytes);
writeFileSync(join(out,'manifest.json'),JSON.stringify({schema:'detector-run-manifest-v1',command:`node apps/sentinel-observatory/live/fault-harness/compare-detectors.mjs --out ${out}`,timestamp:new Date().toISOString(),config_sha256:sha(configBytes),input_sha256:sha(readFileSync(join(out,'streams.jsonl'))),runner_sha256:sha(readFileSync(fileURLToPath(import.meta.url))),engine_sha256:sha(readFileSync(new URL('../brain.mjs',import.meta.url))),peak_rss_bytes:process.resourceUsage().maxRSS*1024,node:process.version},null,2)+'\n');
console.log(JSON.stringify(result.conclusion));
