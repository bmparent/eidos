/** Facts and eligible explanations rebuilt from retained server analysis only. */
import {probeExplanation} from './probe-explanation.mjs';
export function prepareTargetInterpretation(event) {
  if(!/^evt_[a-f0-9]{32}$/.test(event?.id ?? '')||event.probe_version!=='target-probes-v1'||event.target_analysis?.source!=='authoritative_server')throw new Error('invalid_target_evidence');
  const analysis=event.target_analysis,refs=[event.id],facts=[],diagnosis=probeExplanation(event);
  const finite=value=>typeof value==='number'&&Number.isFinite(value)?value:null;
  const metricNames=['dns_ms','tcp_ms','tls_ms','response_wait_ms','headers_ms','failure_ratio','icmp_rtt_ms','icmp_loss_ratio'];
  const states=['calibrating','observing','unusual','unavailable'];
  const metrics=(analysis.signals ?? []).filter(x=>metricNames.includes(x.metric)).map(x=>({metric:x.metric,state:states.includes(x.state)?x.state:'unavailable',value:finite(x.value),prior_median:finite(x.prior_median),event_score:finite(x.event_score),measured_deviation:x.measured_deviation===true,deviation_threshold:finite(x.deviation_threshold),raw_engine_change:x.raw_engine_change===true,samples:Number.isSafeInteger(x.sample_count)?x.sample_count:0}));
  const changeCodes=['dns_changed','certificate_changed','http_status_changed','response_metadata_changed','page_changed','certificate_expiring','probe_failed','path_changed'];
  const changes=(analysis.changes ?? []).map(x=>x.code).filter(x=>changeCodes.includes(x));
  const safe={kind:'target_probe',method:'HEAD',status:Number.isInteger(event.status)?event.status:null,
    outcome:event.error?['timeout','cancelled','dns_failure','tls_error','transport_error'].includes(event.error.code)?event.error.code:'unavailable':'response_observed',
    metrics,changes,stage:diagnosis.stage,stages:diagnosis.stages,elapsed_ms:diagnosis.elapsed_ms,
    analysis_state:['calibrating','observing','change_detected','late_excluded','unavailable'].includes(analysis.state)?analysis.state:'unavailable',
    baseline_generation:analysis.generation ?? null,page_complete:event.probes?.page?.complete===true,redirect_hops:event.redirect_count ?? 0,
    final_status:Number.isInteger(event.probes?.http?.final_status)?event.probes.http.final_status:null,response_chain_complete:event.response_chain_complete===true,
    method_coverage:event.probes?.http?.method_coverage==='head_rejected'?'head_rejected':null,
    get_fallback_status:Number.isInteger(event.probes?.http?.get_fallback?.status)?event.probes.http.get_fallback.status:null};
  const add=(text,evidence=refs)=>facts.push({text,evidence_ids:[...new Set(evidence)].filter(x=>/^evt_[a-f0-9]{32}$/.test(x)).slice(-33)});
  const summary=diagnosis.summary;
  add(summary);
  if(event.error)add(`Recorded failure: ${safe.outcome}. Failure stage: ${diagnosis.stage}. A failed probe does not establish a target-wide outage or its cause.`);
  if(analysis.calibration_reset)add('The baseline restarted for this observation; it is not continuous with the preceding calibration.');
  if(analysis.state==='calibrating')add('At least one measured stream is still calibrating. An established normal range is not yet available for every stream.');
  if(analysis.state==='observing'&&!changes.length)add('No configured change threshold was crossed in this observation’s calibrated streams. This does not establish application health or security.');
  for(const signal of analysis.signals ?? [])if(metricNames.includes(signal.metric)&&finite(signal.value)!==null){
    const value=signal.metric.endsWith('ratio')?`${(signal.value*100).toFixed(1)}%`:`${signal.value.toFixed(2)} ms`;
    const baseline=finite(signal.prior_median)===null?'':signal.metric.endsWith('ratio')?`; prior window median ${(signal.prior_median*100).toFixed(1)}%`:`; prior window median ${signal.prior_median.toFixed(2)} ms`;
    add(`${signal.label}: ${value}${baseline}. Analysis state: ${signal.state}.`,[...(signal.baseline_evidence_ids ?? []),event.id]);
  }
  for(const change of analysis.changes ?? [])if(changeCodes.includes(change.code))add(change.label,change.evidence_ids);
  const item=(label,text,next,support=refs)=>({label,text,support,against:[],next_observation:next});
  const catalogue={
    corroboration_needed:item('limited corroboration','These endpoint measurements leave the cause and target-wide impact unresolved.','Compare subsequent samples and another observation point.'),
    observation_path:item('observer path','The observer’s resolver, connection path or runtime could contribute to the observed behavior.','Compare the separate timing stages across subsequent probes.'),
  };
  if(!event.error&&(changes.length||metrics.some(x=>x.state==='unusual')))catalogue.normal_variation=item('ordinary endpoint variation','Normal routing, endpoint load, deployment or content variation could contribute to the observed changes.','Check whether the same changes persist across multiple observations.');
  if(!event.error&&!changes.length&&!metrics.some(x=>x.state==='unusual'))catalogue.response_scope=item('response coverage','A response to this HEAD request establishes only the measured endpoint behavior at this observation time.',diagnosis.next_check);
  for(const metric of metrics.filter(x=>x.state==='unusual'))catalogue[metric.metric]=item(`${metric.metric} changed`,`${metric.metric.replaceAll('_',' ')} differs from its own provisional Sentinel baseline; this does not establish its cause.`,`Compare the next ${metric.metric.replaceAll('_',' ')} samples with the other measured stages.`);
  if(changes.includes('dns_changed'))catalogue.dns_changed=item('DNS answer change','DNS answer rotation or a configuration change could contribute to the observed address-set change.','Compare later DNS answer sets and connection outcomes.');
  if(changes.includes('certificate_changed'))catalogue.certificate_changed=item('certificate rotation','Expected certificate renewal or an endpoint change could contribute to the certificate fingerprint change.','Compare identity validation and expiry metadata.');
  if(changes.includes('certificate_expiring'))catalogue.certificate_expiring=item('certificate expiry window','The measured certificate expiry is within the configured 14-day review window.','Check subsequent certificate validity and renewal evidence.');
  if(changes.includes('page_changed'))catalogue.page_changed=item('page fingerprint change','Dynamic content, an ordinary deployment or another response change could explain the changed bounded page fingerprint.','Compare repeated fingerprints and whether the whole body fit within the limit.');
  if(event.error)catalogue.probe_failed=item('probe failure',diagnosis.summary,diagnosis.next_check);
  if(!event.error&&event.status>=400)catalogue.http_response=item([405,501].includes(event.status)?'HTTP method coverage':'HTTP error response',diagnosis.summary,diagnosis.next_check);
  return {evidence_id:event.id,linked_evidence_id:null,observed_at:event.completed_at,safe,catalogue,summary,facts,next_checks:[{text:diagnosis.next_check,evidence_ids:refs}],
    input_evidence_ids:[...new Set(facts.flatMap(x=>x.evidence_ids))].slice(-65),
    unknowns:['Only this probe’s public endpoint behavior was measured.','A Sentinel score is experimental change evidence, not an attack probability.','Target internals, visitor traffic and root cause remain unobserved.']};
}
