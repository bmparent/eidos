import { body, db, guarded, HttpError, json, origin } from '../../_shared/platform/core';
import { requireMember } from '../../_shared/platform/memberAuth';
import { ensurePlayground, ownedProject, projectRevision, saveProject } from '../../_shared/platform/playground';
export const onRequestGet = guarded(async ({request,env}) => {
  const member = await requireMember(request,env,false), url = new URL(request.url);
  await ensurePlayground(env);
  const id = url.searchParams.get('id');
  if (!id) return json({ownerId:member.id,projects:(await db(env).prepare('SELECT id,name,head,updated_at FROM eidos_pg_projects WHERE owner_id=? ORDER BY updated_at DESC LIMIT 20').bind(member.id).all()).results});
  if (url.searchParams.has('revisions')) {
    await ownedProject(env,member.id,id);
    return json({revisions:(await db(env).prepare('SELECT id,parent,created_at FROM eidos_pg_revisions WHERE project_id=? ORDER BY created_at DESC LIMIT 100').bind(id).all()).results});
  }
  const result = await projectRevision(env,member.id,id,url.searchParams.get('revision'));
  return json({ownerId:member.id,id:result.project.id,revision:result.revision.id,head:result.project.head,document:result.document});
});
export const onRequestPost = guarded(async ({request,env}) => {
  origin(request); const member = await requireMember(request,env,false);
  await ensurePlayground(env);
  const input = await body(request,2_020_000);
  if (input.expectedOwner && input.expectedOwner !== member.id) throw new HttpError(409, 'Your account changed. Refresh account projects before saving; local work is preserved.');
  return json({ownerId:member.id,...await saveProject(env,member.id,input)});
});
