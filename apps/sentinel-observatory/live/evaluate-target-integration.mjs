/** Deterministic synthetic integration evaluation; never writes live evidence. */
import {TargetBrain} from './target-brain.mjs';
const scenarios=[
  {name:'steady',metric:'tcp_ms',value:i=>12},
  {name:'small_jitter',metric:'tcp_ms',value:i=>12+(i%5-2)*0.1},
  {name:'TCP_step',metric:'tcp_ms',value:i=>i<80?12:120},
  {name:'DNS_step',metric:'dns_ms',value:i=>i<80?5:90},
  {name:'response_wait_step',metric:'response_wait_ms',value:i=>i<80?20:300},
];
const cases=[];
for(const scenario of scenarios){
  const brain=new TargetBrain(),counts={quiet_samples:0,changed_samples:0,quiet_raw_flags:0,quiet_operational_flags:0,changed_raw_flags:0,changed_operational_flags:0,quiet_simple_gate_flags:0,changed_simple_gate_flags:0};
  for(let i=0;i<100;i++){
    const time=new Date(Date.UTC(2026,9,14)+i*15000).toISOString();
    const event={id:'evt_'+(i+1).toString(16).padStart(32,'0'),source_id:'target_0123456789abcdef',method:'HEAD',probe_version:'target-probes-v1',collector_id:'node-pinned-target-v1:synthetic',probe_options:{},completed_at:time,status:200,error:null,
      probes:{dns:{state:'ok',duration_ms:5,addresses:['1.1.1.1']},tcp:{state:'ok',duration_ms:12},tls:{state:'ok',duration_ms:15,certificate:null},http:{state:'ok',response_wait_ms:20,headers_ms:50,header_signature:'a'.repeat(64)},page:{state:'disabled'},path:{state:'unavailable'}}};
    if(scenario.metric==='tcp_ms')event.probes.tcp.duration_ms=scenario.value(i);
    if(scenario.metric==='dns_ms')event.probes.dns.duration_ms=scenario.value(i);
    if(scenario.metric==='response_wait_ms')event.probes.http.response_wait_ms=scenario.value(i);
    const signal=brain.observe(event).signals.find(x=>x.metric===scenario.metric);if(i<12)continue;
    const phase=i>=80&&scenario.name.endsWith('step')?'changed':'quiet';counts[phase+'_samples']++;counts[phase+'_raw_flags']+=Number(signal.raw_engine_change);counts[phase+'_operational_flags']+=Number(signal.state==='unusual');counts[phase+'_simple_gate_flags']+=Number(signal.measured_deviation);
  }
  cases.push({scenario:scenario.name,metric:scenario.metric,...counts});
}
console.log(JSON.stringify({kind:'synthetic_integration_evaluation',live_records_created:0,profile:'target-metrics-v2',cases,
  limits:['Finite controlled numeric sequences, not a labeled security dataset.','A simple measured-deviation baseline receives the same comparisons.','This does not establish Sentinel advantage, root-cause accuracy, attack detection or forecast skill.']},null,2));
