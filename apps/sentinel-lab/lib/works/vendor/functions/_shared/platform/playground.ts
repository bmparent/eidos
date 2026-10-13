import {supportsProductionCloud,CLOUD_FORMAT_NOTICE} from '../../../src/playground/releaseBoundary';
import { clean, db, hash, HttpError, type PlatformEnv } from './core';
import { validateProject, mediaSlots, type Project } from '../../../src/playground/model';
import schema, { assetQuota } from './playgroundSchema';
import { cloudPreflight } from '../../../src/playground/limits';
import type { Database } from './core';
const initialized = new WeakMap<Database,Promise<unknown>>();
// Additive, transactional feature initialization on the existing database. No earlier schema is altered.
export async function ensurePlayground(env: PlatformEnv) {
  const database=db(env);
  let ready=initialized.get(database);
  if(!ready) {ready=database.batch([...schema.split(';').map(s=>s.trim()).filter(Boolean),assetQuota].map(s=>database.prepare(s)));initialized.set(database,ready);}
  try{await ready;}catch(error){initialized.delete(database);throw error;}
}

export interface CloudProject { id: string; owner_id: string; name: string; head: string; created_at: string; updated_at: string }
export interface Revision { id: string; project_id: string; parent: string | null; document: string; created_at: string }
export function identifier(value: unknown) {
  const id = clean(value, 80);
  if (!/^[a-f0-9-]{36}$/.test(id)) throw new HttpError(400, 'Invalid project reference.');
  return id;
}
export async function ownedProject(env: PlatformEnv, owner: string, id: unknown) {
  const project = await db(env).prepare('SELECT * FROM eidos_pg_projects WHERE id=? AND owner_id=?').bind(identifier(id), owner).first<CloudProject>();
  if (!project) throw new HttpError(404, 'Project not found.');
  return project;
}
export async function projectRevision(env: PlatformEnv, owner: string, projectId: unknown, revisionId?: unknown) {
  const project = await ownedProject(env, owner, projectId);
  const revision = await db(env).prepare('SELECT * FROM eidos_pg_revisions WHERE id=? AND project_id=?').bind(revisionId ? identifier(revisionId) : project.head, project.id).first<Revision>();
  if (!revision) throw new HttpError(404, 'Revision not found.');
  const document = JSON.parse(revision.document) as Project;
  for (const section of mediaSlots(document)) if (section.image.startsWith('asset:')) {
    const asset = await db(env).prepare('SELECT data FROM eidos_pg_assets WHERE owner_id=? AND hash=?').bind(owner, section.image.slice(6)).first<{data:string}>();
    if (!asset) throw new HttpError(503, 'An image is unavailable. Your local copy is unchanged.');
    section.image = asset.data;
  }
  return { project, revision, document: validateProject(document) };
}
export async function saveProject(env: PlatformEnv, owner: string, input: Record<string, unknown>) {
  let document: Project;
  try { document = validateProject(input.document); } catch (error) { throw new HttpError(400, (error as Error).message); }
  if(!supportsProductionCloud(document) && env.EIDOS_PLAYGROUND_AUTHORING_ENABLED!=='true')throw new HttpError(409,CLOUD_FORMAT_NOTICE);
  const database=db(env);
  const preflight = cloudPreflight(document);
  if (!preflight.allowed) throw new HttpError(413, preflight.message);
  const mediaBytes = mediaSlots(document).reduce((n,s)=>n+s.image.length,0);
  if(mediaBytes > 1_800_000) throw new HttpError(413, 'This page exceeds its 1.8 MB embedded-image allowance. Local content and exports are preserved.');
  for(const section of mediaSlots(document)) if(section.image) {
    const digest = await hash(section.image);
    const exists = await database.prepare('SELECT hash FROM eidos_pg_assets WHERE owner_id=? AND hash=?').bind(owner,digest).first();
    if(!exists) {
      if(!env.EIDOS_VALIDATE_PLAYGROUND_IMAGE) throw new HttpError(503, 'Account image validation is unavailable. Keep editing locally and retry Save later.');
      try { await env.EIDOS_VALIDATE_PLAYGROUND_IMAGE(section.image); } catch { throw new HttpError(400, 'This image could not be decoded safely. Choose a smaller PNG, JPEG or WebP. Your local design is preserved.'); }
    }
  }
  const now = new Date().toISOString(), revisionId = crypto.randomUUID();
  const project = input.id ? await ownedProject(env, owner, input.id) : null;
  if (project && project.head !== input.expectedRevision) throw new HttpError(409, 'A newer revision exists. Reopen it or save your local work as a new project.');
  const count = await database.prepare('SELECT COUNT(*) AS n FROM eidos_pg_projects WHERE owner_id=?').bind(owner).first<{n:number}>();
  if (!project && (count?.n || 0) >= 20) throw new HttpError(409, 'This account has reached its 20-project limit.');
  if (project) {
    const revisions = await database.prepare('SELECT COUNT(*) AS n FROM eidos_pg_revisions WHERE project_id=?').bind(project.id).first<{n:number}>();
    if ((revisions?.n || 0) >= 100) throw new HttpError(409, 'This project has 100 revisions. Duplicate it to continue.');
  }
  const id = project?.id || crypto.randomUUID();
  const stored = structuredClone(document);
  const statements = [];
  // Recheck admission inside the write transaction: concurrent requests can both
  // observe the same last slot during the earlier, user-friendly preflight.
  if (!project) statements.push(database.prepare('INSERT INTO eidos_pg_projects(id,owner_id,name,head,created_at,updated_at) SELECT ?,?,?,?,?,? WHERE (SELECT COUNT(*) FROM eidos_pg_projects WHERE owner_id=?)<20').bind(id,owner,document.name,revisionId,now,now,owner));
  for (const section of mediaSlots(stored)) if (section.image) {
    const digest = await hash(section.image);
    // A rejected project or stale head must not leave newly uploaded orphan assets.
    statements.push(database.prepare('INSERT OR IGNORE INTO eidos_pg_assets(owner_id,hash,data) SELECT ?,?,? WHERE EXISTS(SELECT 1 FROM eidos_pg_projects WHERE id=? AND owner_id=? AND head=?)').bind(owner,digest,section.image,id,owner,project?.head || revisionId));
    section.image = 'asset:' + digest;
  }
  statements.push(database.prepare('INSERT INTO eidos_pg_revisions(id,project_id,parent,document,created_at) SELECT ?,id,?,?,? FROM eidos_pg_projects WHERE id=? AND owner_id=? AND head=?').bind(revisionId,project?.head || null,JSON.stringify(stored),now,id,owner,project?.head || revisionId));
  statements.push(database.prepare('UPDATE eidos_pg_projects SET name=?,head=?,updated_at=? WHERE id=? AND owner_id=? AND head=? AND EXISTS(SELECT 1 FROM eidos_pg_revisions WHERE id=?)').bind(document.name,revisionId,now,id,owner,project?.head || revisionId,revisionId));
  try { await database.batch(statements); } catch(error) { if(String(error).includes('Playground owner image quota')) throw new HttpError(413,'Your account has reached its 40 MB image storage limit. Your local work and earlier revisions are preserved.'); throw error; }
  const saved = await database.prepare('SELECT id FROM eidos_pg_revisions WHERE id=?').bind(revisionId).first();
  if (!saved && !project) throw new HttpError(409, 'This account has reached its 20-project limit.');
  if (!saved) throw new HttpError(409, 'Another tab saved first. Your edits are still on this device. Save them as a new project or reopen the latest revision.');
  return { id, revision: revisionId, name: document.name, updatedAt: now };
}
