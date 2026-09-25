import { createRemoteJWKSet, jwtVerify } from 'jose';
import type { Database } from './vendor/functions/_shared/platform/core';
import { platformDatabase } from './database';
import { observeStripeTest } from './stripeObservation';

type Source = Record<string, string | undefined>;
type Work = { id:string; title:string; detail:string; status:string; severity:string; evidence_url:string|null; due_at:string|null; source:string; created_at:string; updated_at:string; version:number };
const response = (body:unknown,status=200) => new Response(JSON.stringify(body), {status,headers:{'content-type':'application/json; charset=utf-8','cache-control':'no-store','x-content-type-options':'nosniff'}});
const id = () => crypto.randomUUID();
const text = (value:unknown,max:number) => typeof value === 'string' ? value.trim().slice(0,max) : '';
const allowedUrl = (value:unknown) => {
  if (!value) return null;
  try { const url = new URL(String(value)); return url.protocol === 'https:' && !url.username && !url.password ? url.toString() : null; } catch { return null; }
};
const ownerKeys = new Map<string,ReturnType<typeof createRemoteJWKSet>>();
export async function verifyOwner(request:Request, source:Source) {
  const team = source.EIDOS_OPS_ACCESS_TEAM;
  const audience = source.EIDOS_OPS_ACCESS_AUD;
  const owner = source.EIDOS_OPS_OWNER_SUB;
  const jwt = request.headers.get('cf-access-jwt-assertion') || '';
  if (!team || !/^[a-z0-9-]{3,80}$/.test(team) || !audience || !owner || !jwt) return null;
  const issuer = `https://${team}.cloudflareaccess.com`;
  let keys = ownerKeys.get(issuer);
  if (!keys) { keys = createRemoteJWKSet(new URL(`${issuer}/cdn-cgi/access/certs`)); ownerKeys.set(issuer,keys); }
  try {
    const {payload} = await jwtVerify(jwt,keys,{issuer,audience,algorithms:['RS256']});
    return payload.sub === owner && payload.email === source.EIDOS_OPS_OWNER_EMAIL ? payload.sub : null;
  } catch { return null; }
}
async function audit(database:Database, actor:string, action:string, targetType:string, targetId:string, before:unknown, after:unknown, reason:string, outcome:string, correlation:string, revision:string) {
  await database.prepare('INSERT INTO eidos_ops_audit(id,actor_sub,action,target_type,target_id,before_safe,after_safe,reason,outcome,correlation_id,source_revision,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)')
    .bind(id(),actor,action,targetType,targetId,JSON.stringify(before),JSON.stringify(after),reason,outcome,correlation,revision,new Date().toISOString()).run();
}
async function all(database:Database,sql:string,...params:unknown[]) { return (await database.prepare(sql).bind(...params).all()).results; }
async function claim(database:Database,key:string,actor:string,action:string) {
  const result=await database.prepare("INSERT OR IGNORE INTO eidos_ops_idempotency(key,actor_sub,action,state,response_safe,created_at) VALUES(?,?,?,'pending',NULL,?)").bind(key,actor,action,new Date().toISOString()).run();
  return result.meta.changes===1;
}
async function complete(database:Database,key:string,data:unknown) {
  await database.prepare("UPDATE eidos_ops_idempotency SET state='completed',response_safe=? WHERE key=? AND state='pending'").bind(JSON.stringify(data),key).run();
}
function envelope(data:unknown,source:string,environment:string,observedAt=new Date().toISOString(),status='healthy',errorCode:string|null=null) {
  return {status,observedAt,source,environment,data,staleAfter:new Date(Date.now()+300000).toISOString(),errorCode,correlationId:id()};
}
const workSelect = 'SELECT id,title,detail,status,severity,evidence_url,due_at,source,created_at,updated_at,version FROM eidos_ops_work';
export async function operations(request:Request, source:Source, database=platformDatabase(source)) {
  const correlation = id();
  const actor = await verifyOwner(request,source);
  if (!actor) return response({error:'Owner access required.',correlationId:correlation},403);
  if (!database) return response(envelope(null,'works-database','preview',undefined,'unavailable','database_not_configured'),503);
  const environment = source.EIDOS_OPS_ENVIRONMENT || 'unknown';
  const revision = source.EIDOS_SOURCE_REVISION || 'unknown';
  try {
    if (request.method === 'GET') {
      const [work,accounts,orders,agents,outcomes,auditRows,checkpoints] = await Promise.all([
        all(database,`${workSelect} WHERE status!='done' ORDER BY CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END,updated_at DESC LIMIT 50`),
        database.prepare('SELECT COUNT(*) AS count FROM eidos_email_members').first(),
        all(database,'SELECT status,COUNT(*) AS count FROM eidos_orders GROUP BY status'),
        database.prepare('SELECT COUNT(*) AS count FROM eidos_agents WHERE revoked=0').first(),
        all(database,'SELECT day,feature,outcome,count,updated FROM eidos_outcomes WHERE day>=? ORDER BY updated DESC LIMIT 80',new Date(Date.now()-7*86400000).toISOString().slice(0,10)),
        all(database,'SELECT action,target_type,target_id,outcome,created_at,correlation_id FROM eidos_ops_audit ORDER BY created_at DESC LIMIT 30'),
        all(database,'SELECT source,environment,observed_at,last_success_at,stale_after,status,error_code FROM eidos_ops_connector_checkpoints'),
      ]);
      return response(envelope({work,accounts:accounts?.count ?? null,orders,agents:agents?.count ?? null,outcomes,audit:auditRows,checkpoints,sourceRevision:revision,
        connectors:{inquiries:'not_configured',stripeProvider:'not_configured',cloudflare:'not_configured',github:'not_configured',ga4:'not_configured',drive:'not_configured'},
        readiness:{database:'ready',assistantEnabled:source.EIDOS_AI_ENABLED==='true',accountsEnabled:source.EIDOS_ACCOUNTS_ENABLED==='true',googleConfigured:Boolean(source.EIDOS_GOOGLE_CLIENT_ID && source.EIDOS_GOOGLE_CLIENT_SECRET && source.EIDOS_GOOGLE_REDIRECT_URI),kitMode:source.STRIPE_SECRET_KEY?.startsWith('sk_test_')?'test':source.STRIPE_SECRET_KEY?.startsWith('sk_live_')?'live':'unknown',snapshot:'not accepted for sale'},
        release:{status:'blocked',siteCandidate:'f2d98bc1d1de8a7b01509b580558c39858949684',backendCandidate:'a50647dddb8a270563062b378033155afe935753'}},'works-database',environment));
    }
    if (request.method !== 'POST' || request.headers.get('content-type')?.split(';')[0] !== 'application/json') return response({error:'Unsupported request.'},405);
    const bodyText = await request.text();
    if (bodyText.length > 8000) return response({error:'Request too large.'},413);
    const input = JSON.parse(bodyText) as Record<string,unknown>;
    const command = text(input.command,40);
    if (command === 'accounts') {
      const term = text(input.search,120);
      const page = Math.max(0,Math.min(100,Number(input.page)||0));
      if (term.length < 2) return response(envelope({rows:[],page,total:null},'works-members',environment));
      const pattern = `%${term.replace(/[\\%_]/g,'\\$&')}%`;
      const rows = await all(database,`SELECT id,substr(email,1,2)||'***@'||substr(email,instr(email,'@')+1) AS email_masked,username,kind,created_at,disabled FROM eidos_email_members WHERE email LIKE ? ESCAPE '\\' OR username LIKE ? ESCAPE '\\' OR id=? ORDER BY created_at DESC LIMIT 26 OFFSET ?`,pattern,pattern,term,page*25);
      return response(envelope({rows:rows.slice(0,25),page,hasMore:rows.length>25},'works-members',environment));
    }
    if (command === 'account') {
      const memberId = text(input.memberId,100);
      const member = await database.prepare('SELECT id,email,username,kind,created_at,disabled FROM eidos_email_members WHERE id=?').bind(memberId).first();
      if (!member) return response({error:'Member not found.'},404);
      const [methods,projects,purchases,sessions,history] = await Promise.all([
        database.prepare('SELECT EXISTS(SELECT 1 FROM eidos_member_passwords WHERE member_id=?) AS password,EXISTS(SELECT 1 FROM eidos_member_google WHERE member_id=?) AS google').bind(memberId,memberId).first(),
        database.prepare('SELECT COUNT(*) AS count FROM eidos_pg_projects WHERE owner_id=?').bind(memberId).first(),
        database.prepare("SELECT COUNT(*) AS count FROM eidos_pg_orders WHERE owner_id=? AND status='paid'").bind(memberId).first(),
        database.prepare('SELECT COUNT(*) AS count FROM eidos_member_sessions WHERE member_id=? AND expires>?').bind(memberId,Date.now()).first(),
        all(database,"SELECT action,outcome,created_at FROM eidos_ops_audit WHERE target_type='member' AND target_id=? ORDER BY created_at DESC LIMIT 30",memberId),
      ]);
      return response(envelope({member,methods,projects:projects?.count ?? null,purchases:purchases?.count ?? null,activeSessions:sessions?.count ?? null,history,emailVerified:'unknown',lastAuthActivity:'unknown'},'works-members',environment));
    }
    if (command === 'payments') {
      const orders = await all(database,'SELECT id,status,created_at,paid_at,payment_intent FROM eidos_orders ORDER BY created_at DESC LIMIT 50');
      const playground = await all(database,'SELECT id,owner_id,project_id,status,mode,amount,created_at,paid_at,payment_intent,CASE WHEN length(archive)>0 THEN 1 ELSE 0 END AS archive_present FROM eidos_pg_orders ORDER BY created_at DESC LIMIT 50');
      const provider = await observeStripeTest(source.STRIPE_SECRET_KEY,orders as {id:string;status:string;payment_intent?:string|null}[],playground as {id:string;status:string;payment_intent?:string|null;archive_present?:number|null}[]);
      await database.prepare("INSERT INTO eidos_ops_connector_checkpoints(source,environment,observed_at,last_success_at,stale_after,status,error_code,cursor) VALUES('stripe','test',?,?,?,?,?,NULL) ON CONFLICT(source,environment) DO UPDATE SET observed_at=excluded.observed_at,last_success_at=CASE WHEN excluded.status IN ('healthy','partial') THEN excluded.observed_at ELSE eidos_ops_connector_checkpoints.last_success_at END,stale_after=excluded.stale_after,status=excluded.status,error_code=excluded.error_code")
        .bind(provider.observedAt,['healthy','partial'].includes(provider.status)?provider.observedAt:null,provider.staleAfter,provider.status,provider.errorCode).run();
      return response(envelope({orders:orders.map(({payment_intent: _payment_intent,...safe})=>safe),playground:playground.map(({payment_intent: _payment_intent,...safe})=>safe),provider,revenue:null},'works-order-ledger',environment));
    }
    if (command === 'agents') return response(envelope(await all(database,'SELECT id,name,profile_url,revoked,created_at FROM eidos_agents ORDER BY created_at DESC LIMIT 50'),'works-community-agents',environment));
    if (command === 'artifacts') return response(envelope(await all(database,"SELECT id,owner_id,project_id,name,status,mode,created_at FROM eidos_pg_orders ORDER BY created_at DESC LIMIT 50"),'works-purchased-archives-index',environment));
    if (command === 'work') return response(envelope(await all(database,`${workSelect} ORDER BY updated_at DESC LIMIT 100`),'works-operations',environment));
    if (command === 'work_notes') return response(envelope(await all(database,'SELECT id,work_id,body,created_at FROM eidos_ops_work_notes WHERE work_id=? ORDER BY created_at DESC LIMIT 100',text(input.id,100)),'works-operations',environment));
    if (command !== 'work_create' && command !== 'work_update' && command !== 'work_note' && command !== 'account_action') return response({error:'Unknown command.'},400);
    const origin = request.headers.get('origin');
    if (origin !== source.EIDOS_OPS_ORIGIN || request.headers.get('x-ops-csrf') !== '1') {
      await audit(database,actor,command,'request','-',null,null,'','denied_csrf',correlation,revision);
      return response({error:'Origin or CSRF check failed.'},403);
    }
    const reason = text(input.reason,500);
    const key = text(input.idempotencyKey,100);
    if (!reason || !/^[a-f0-9-]{36}$/.test(key)) {
      await audit(database,actor,command,'request','-',null,null,'','denied_validation',correlation,revision);
      return response({error:'Reason and idempotency key required.'},400);
    }
    const prior = await database.prepare('SELECT after_safe,outcome FROM eidos_ops_audit WHERE correlation_id=?').bind(key).first<{after_safe:string;outcome:string}>();
    if (prior) return response({replayed:true,outcome:prior.outcome,data:JSON.parse(prior.after_safe)});
    if (command === 'work_create') {
      const title=text(input.title,180),detail=text(input.detail,2000),severity=text(input.severity,20)||'normal';
      if (!title || !['critical','high','normal','low'].includes(severity)) return response({error:'Invalid work item.'},400);
      if (!await claim(database,key,actor,command)) return response({error:'Action with this key is still pending.'},409);
      const workId=id(), now=new Date().toISOString(), evidence=allowedUrl(input.evidenceUrl);
      await database.prepare('INSERT INTO eidos_ops_work(id,title,detail,status,severity,evidence_url,due_at,source,created_at,updated_at,version) VALUES(?,?,?,?,?,?,?,?,?,?,1)').bind(workId,title,detail,'open',severity,evidence,text(input.dueAt,40)||null,'manual',now,now).run();
      const after={id:workId,title,status:'open',severity,version:1};
      await audit(database,actor,command,'work',workId,null,after,reason,'success',key,revision);
      await complete(database,key,after);
      return response({data:after},201);
    }
    if (command === 'work_update') {
      const workId=text(input.id,100),current=await database.prepare(`${workSelect} WHERE id=?`).bind(workId).first<Work>();
      if (!current) return response({error:'Work item not found.'},404);
      const version=Number(input.version),status=text(input.status,30);
      if (version!==current.version || !['open','in_progress','blocked','done'].includes(status)) {
        await audit(database,actor,command,'work',workId,{status:current.status,version:current.version},null,reason,'denied_stale_or_invalid',key,revision);
        return response({error:'Stale or invalid work update.'},409);
      }
      if (!await claim(database,key,actor,command)) return response({error:'Action with this key is still pending.'},409);
      const nextVersion=version+1,now=new Date().toISOString();
      const change=await database.prepare('UPDATE eidos_ops_work SET status=?,updated_at=?,version=? WHERE id=? AND version=?').bind(status,now,nextVersion,workId,version).run();
      if (!change.meta.changes) {
        await audit(database,actor,command,'work',workId,{status:current.status,version},null,reason,'denied_stale',key,revision);
        await database.prepare("UPDATE eidos_ops_idempotency SET state='denied' WHERE key=?").bind(key).run();
        return response({error:'Work item changed.'},409);
      }
      const after={id:workId,status,version:nextVersion};
      await audit(database,actor,command,'work',workId,{status:current.status,version},after,reason,'success',key,revision);
      await complete(database,key,after);
      return response({data:after});
    }
    if (command === 'work_note') {
      const workId=text(input.id,100),note=text(input.note,2000);
      if (!note || !(await database.prepare('SELECT id FROM eidos_ops_work WHERE id=?').bind(workId).first())) return response({error:'Invalid note or work item.'},400);
      if (!await claim(database,key,actor,command)) return response({error:'Action with this key is still pending.'},409);
      const noteId=id(),now=new Date().toISOString();
      await database.prepare('INSERT INTO eidos_ops_work_notes(id,work_id,actor_sub,body,created_at) VALUES(?,?,?,?,?)').bind(noteId,workId,actor,note,now).run();
      const after={id:noteId,workId,createdAt:now};
      await audit(database,actor,command,'work',workId,null,after,reason,'success',key,revision);
      await complete(database,key,after);
      return response({data:after},201);
    }
    const memberId=text(input.memberId,100),action=text(input.action,30),confirmation=text(input.confirmation,100);
    const member=await database.prepare('SELECT id,username,disabled FROM eidos_email_members WHERE id=?').bind(memberId).first<{id:string;username:string;disabled:number}>();
    if (!member || confirmation!==memberId || !['revoke_sessions','suspend','restore','username'].includes(action)) return response({error:'Invalid member action or confirmation.'},400);
    if (input.expectedUsername!==member.username || Number(input.expectedDisabled)!==member.disabled || (action==='suspend' && member.disabled===1) || (action==='restore' && member.disabled===0)) {
      await audit(database,actor,`account_${action}`,'member',memberId,{username:member.username,disabled:member.disabled},null,reason,'denied_stale',key,revision);
      return response({error:'Member state changed; reopen detail.'},409);
    }
    const username=action==='username'?text(input.username,40):'';
    if (action==='username' && (!/^[a-zA-Z0-9_]{3,40}$/.test(username) || await database.prepare('SELECT id FROM eidos_email_members WHERE username=? COLLATE NOCASE AND id!=?').bind(username,memberId).first())) return response({error:'Username unavailable.'},409);
    if (!await claim(database,key,actor,command)) return response({error:'Action with this key is still pending.'},409);
    if (action==='revoke_sessions') await database.prepare('DELETE FROM eidos_member_sessions WHERE member_id=?').bind(memberId).run();
    let accountChange: {meta:{changes:number}}|null=null;
    if (action==='suspend' || action==='restore') accountChange=await database.prepare('UPDATE eidos_email_members SET disabled=? WHERE id=? AND username=? AND disabled=?').bind(action==='suspend'?1:0,memberId,member.username,member.disabled).run();
    if (action==='username') {
      accountChange=await database.prepare('UPDATE eidos_email_members SET username=? WHERE id=? AND username=? AND disabled=?').bind(username,memberId,member.username,member.disabled).run();
    }
    if (accountChange && !accountChange.meta.changes) {
      await audit(database,actor,`account_${action}`,'member',memberId,{username:member.username,disabled:member.disabled},null,reason,'denied_stale',key,revision);
      await database.prepare("UPDATE eidos_ops_idempotency SET state='denied' WHERE key=?").bind(key).run();
      return response({error:'Member state changed; reopen detail.'},409);
    }
    const after={id:memberId,action,username:action==='username'?text(input.username,40):member.username,disabled:action==='suspend'?1:action==='restore'?0:member.disabled};
    await audit(database,actor,`account_${action}`,'member',memberId,{username:member.username,disabled:member.disabled},after,reason,'success',key,revision);
    await complete(database,key,after);
    return response({data:after});
  } catch {
    return response({error:'Operations source unavailable.',correlationId:correlation},503);
  }
}
