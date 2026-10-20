import { assertProjectSaveBoundary } from '../../_shared/platform/studioBoundary';
import { body, db, guarded, HttpError, json, origin, reserve } from '../../_shared/platform/core';
import { requireMember } from '../../_shared/platform/memberAuth';
import { ensurePlayground, ownedProject, projectRevision, saveProject } from '../../_shared/platform/playground';
export const onRequestGet = guarded(async ({request,env}) => {
  const member = await requireMember(request,env,false), url = new URL(request.url);
  await ensurePlayground(env);
  const id = url.searchParams.get('id');
  if (!id) {
    const images = await db(env).prepare('SELECT COALESCE(SUM(length(data)),0) AS bytes FROM eidos_pg_assets WHERE owner_id=?').bind(member.id).first<{bytes:number}>();
    const documents = await db(env).prepare('SELECT COALESCE(SUM(length(CAST(r.document AS BLOB))),0) AS bytes FROM eidos_pg_revisions r JOIN eidos_pg_projects p ON p.id=r.project_id WHERE p.owner_id=?').bind(member.id).first<{bytes:number}>();
    return json({ownerId:member.id,newFormatSaveEnabled:env.EIDOS_PLAYGROUND_AUTHORING_ENABLED==='true',resumeProtocol:1,resumeSaveEnabled:env.EIDOS_PLAYGROUND_RESUME_ENABLED==='true',storage:{imageBytes:images?.bytes ?? 0,imageLimit:40_000_000,documentBytes:documents?.bytes ?? 0,documentLimit:20_000_000,projectLimit:20,revisionLimit:100},projects:(await db(env).prepare('SELECT id,name,head,updated_at FROM eidos_pg_projects WHERE owner_id=? ORDER BY updated_at DESC LIMIT 20').bind(member.id).all()).results});
  }
  if (url.searchParams.has('revisions')) {
    await ownedProject(env,member.id,id);
    return json({revisions:(await db(env).prepare('SELECT id,parent,created_at FROM eidos_pg_revisions WHERE project_id=? ORDER BY created_at DESC LIMIT 100').bind(id).all()).results});
  }
  const result = await projectRevision(env,member.id,id,url.searchParams.get('revision'));
  return json({ownerId:member.id,id:result.project.id,revision:result.revision.id,head:result.project.head,document:result.document});
});
export const onRequestPost = guarded(async ({request,env}) => {
  origin(request); const input = await body(request,2_020_000);
  assertProjectSaveBoundary(env,input);
  const member = await requireMember(request,env,false);
  await ensurePlayground(env);
  if (input.expectedOwner && input.expectedOwner !== member.id) throw new HttpError(409, 'Your account changed. Refresh account projects before saving; local work is preserved.');
  if (!await reserve(db(env), 'project-save:'+member.id, 1, 60, 3600))
    throw new HttpError(429, 'Too many project saves. Your local work is preserved; try again shortly.');
  return json({ownerId:member.id,...await saveProject(env,member.id,input)});
});
