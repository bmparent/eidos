import { clean, db, hash, HttpError, type PlatformEnv } from './core';
import { validateProject, type Project } from '../../../src/playground/model';
import schema from './playgroundSchema';
import type { Database } from './core';
const initialized = new WeakMap<Database,Promise<unknown>>();
// Additive, transactional feature initialization on the existing database. No earlier schema is altered.
export async function ensurePlayground(env: PlatformEnv) {
  const database=db(env);
  let ready=initialized.get(database);
  if(!ready) {ready=database.batch(schema.split(';').map(s=>s.trim()).filter(Boolean).map(s=>database.prepare(s)));initialized.set(database,ready);}
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
  for (const section of document.sections) if (section.image.startsWith('asset:')) {
    const asset = await db(env).prepare('SELECT data FROM eidos_pg_assets WHERE owner_id=? AND hash=?').bind(owner, section.image.slice(6)).first<{data:string}>();
    if (!asset) throw new HttpError(503, 'An image is unavailable. Your local copy is unchanged.');
    section.image = asset.data;
  }
  return { project, revision, document: validateProject(document) };
}
export async function saveProject(env: PlatformEnv, owner: string, input: Record<string, unknown>) {
  const database = db(env);
  let document: Project;
  try { document = validateProject(input.document); } catch (error) { throw new HttpError(400, (error as Error).message); }
  if (new TextEncoder().encode(JSON.stringify(document)).length > 2_000_000) throw new HttpError(413, 'Cloud projects must be under 2 MB. Use smaller images; your local copy is safe.');
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
  for (const section of stored.sections) if (section.image) {
    const digest = await hash(section.image);
    statements.push(database.prepare('INSERT OR IGNORE INTO eidos_pg_assets(owner_id,hash,data) VALUES(?,?,?)').bind(owner,digest,section.image));
    section.image = 'asset:' + digest;
  }
  if (!project) statements.push(database.prepare('INSERT INTO eidos_pg_projects(id,owner_id,name,head,created_at,updated_at) VALUES(?,?,?,?,?,?)').bind(id,owner,document.name,revisionId,now,now));
  statements.push(database.prepare('INSERT INTO eidos_pg_revisions(id,project_id,parent,document,created_at) SELECT ?,id,?,?,? FROM eidos_pg_projects WHERE id=? AND owner_id=? AND head=?').bind(revisionId,project?.head || null,JSON.stringify(stored),now,id,owner,project?.head || revisionId));
  statements.push(database.prepare('UPDATE eidos_pg_projects SET name=?,head=?,updated_at=? WHERE id=? AND owner_id=? AND head=? AND EXISTS(SELECT 1 FROM eidos_pg_revisions WHERE id=?)').bind(document.name,revisionId,now,id,owner,project?.head || revisionId,revisionId));
  await database.batch(statements);
  const saved = await database.prepare('SELECT id FROM eidos_pg_revisions WHERE id=?').bind(revisionId).first();
  if (!saved) throw new HttpError(409, 'Another tab saved first. Your edits are still on this device. Save them as a new project or reopen the latest revision.');
  return { id, revision: revisionId, name: document.name, updatedAt: now };
}
