/** Versioned authoritative endpoint history and optimistic per-target checkpoints. */
import {TargetBrain, TARGET_PROFILE} from './target-brain.mjs';

const targetDatabase=env=>env?.OBSERVATORY_DB?.prepare&&env.OBSERVATORY_DB?.batch?env.OBSERVATORY_DB:null;
const targetRetention=7*86400000;
const targetBounded=async promise=>{let timer;try{return await Promise.race([promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('target_storage_deadline')),2500);})]);}finally{clearTimeout(timer);}};

export async function retainTargetObservation(env,event) {
  const db=targetDatabase(env);
  const unavailable=()=>({...event,target_analysis:{source:'authoritative_server',state:'unavailable',signals:[],changes:[],reason:'Retained target analysis is unavailable; measured evidence is preserved.'},target_storage:'unavailable'});
  if(!db)return unavailable();
  try{
    const existing=await targetBounded(db.prepare('SELECT event_json FROM observatory_target_records WHERE id = ?').bind(event.id).first());
    if(existing?.event_json)return {...JSON.parse(existing.event_json),target_storage:'retained',duplicate:true};
    for(let attempt=0;attempt<4;attempt++){
      const row=await targetBounded(db.prepare('SELECT revision, state_json FROM observatory_target_states WHERE target_id = ?').bind(event.source_id).first());
      const revision=Number.isSafeInteger(row?.revision)?row.revision:0;
      let saved=null;try{saved=row?.state_json?JSON.parse(row.state_json):null;}catch{saved={profile:'invalid'};}
      const brain=new TargetBrain(saved),analysis=brain.observe(event),retained={...event,target_analysis:analysis,target_storage:'retained'};
      const raw=JSON.stringify(retained),checkpoint=JSON.stringify(brain.checkpoint());
      if(raw.length>65536||checkpoint.length>90000)throw new Error('target_record_bound');
      const statements=[
        db.prepare('INSERT INTO observatory_target_states (target_id, revision, profile, last_event_id, updated_at, state_json) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(target_id) DO UPDATE SET revision=excluded.revision, profile=excluded.profile, last_event_id=excluded.last_event_id, updated_at=excluded.updated_at, state_json=excluded.state_json WHERE observatory_target_states.revision = ?')
          .bind(event.source_id,revision+1,TARGET_PROFILE,event.id,event.completed_at,checkpoint,revision),
        db.prepare('INSERT OR IGNORE INTO observatory_target_records (id, target_id, observed_at, event_json) SELECT ?, ?, ?, ? WHERE EXISTS (SELECT 1 FROM observatory_target_states WHERE target_id=? AND revision=? AND last_event_id=?)')
          .bind(event.id,event.source_id,event.completed_at,raw,event.source_id,revision+1,event.id),
      ];
      const result=await targetBounded(db.batch(statements));
      if(result?.some(x=>x?.success===false))throw new Error('target_storage_error');
      if(result?.[0]?.meta?.changes&&result?.[1]?.meta?.changes){
        const cutoff=new Date(Date.now()-targetRetention).toISOString();
        await targetBounded(db.batch([
          db.prepare('DELETE FROM observatory_target_records WHERE observed_at < ? OR id IN (SELECT id FROM observatory_target_records ORDER BY observed_at DESC, id DESC LIMIT -1 OFFSET 2048)').bind(cutoff),
          db.prepare('DELETE FROM observatory_target_states WHERE updated_at < ? OR target_id IN (SELECT target_id FROM observatory_target_states ORDER BY updated_at DESC LIMIT -1 OFFSET 64)').bind(cutoff),
        ])).catch(()=>{});
        return retained;
      }
      const duplicate=await targetBounded(db.prepare('SELECT event_json FROM observatory_target_records WHERE id = ?').bind(event.id).first());
      if(duplicate?.event_json)return {...JSON.parse(duplicate.event_json),target_storage:'retained',duplicate:true};
    }
  }catch{/* Storage failure cannot turn an actual measurement into a healthy baseline. */}
  return unavailable();
}

export async function resolveTargetEvidence(env,id) {
  if(!/^evt_[a-f0-9]{32}$/.test(id ?? ''))return {found:false,reason:'not_found'};
  const db=targetDatabase(env);if(!db)return {found:false,reason:'unavailable'};
  try{
    const cutoff=new Date(Date.now()-targetRetention).toISOString();
    const row=await targetBounded(db.prepare('SELECT event_json FROM observatory_target_records WHERE id = ? AND observed_at >= ?').bind(id,cutoff).first());
    if(!row?.event_json)return {found:false,reason:'not_found'};
    const event=JSON.parse(row.event_json);
    if(event.id!==id||event.probe_version!=='target-probes-v1'||event.target_analysis?.source!=='authoritative_server')return {found:false,reason:'invalid'};
    return {found:true,event};
  }catch{return {found:false,reason:'unavailable'};}
}

export async function targetHistory(env,targetId) {
  if(!/^target_[a-f0-9]{16}$/.test(targetId ?? ''))return {available:false,events:[]};
  const db=targetDatabase(env);if(!db)return {available:false,events:[]};
  try{
    const cutoff=new Date(Date.now()-targetRetention).toISOString();
    const result=await targetBounded(db.prepare('SELECT event_json FROM observatory_target_records WHERE target_id = ? AND observed_at >= ? ORDER BY observed_at DESC, id DESC LIMIT 128').bind(targetId,cutoff).all());
    if(result.success===false)throw new Error('history_unavailable');
    return {available:true,source:'authoritative_server',events:(result.results ?? []).map(row=>JSON.parse(row.event_json)),retention_days:7,limit:128};
  }catch{return {available:false,events:[]};}
}
