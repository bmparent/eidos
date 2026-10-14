import { db, HttpError, record, type Database, type PlatformEnv } from './core';
import { buildFrozenPackage, canonicalJson, hashBytes, validateProjectSnapshot, type ProjectSnapshot, type PackageManifest } from './projectDeliveryRenderer';
import schema from './projectDeliverySchema';

export type ProjectDeliveryEnv = PlatformEnv & { EIDOS_PLAYGROUND_DELIVERY_MODE?: string };
export const DELIVERY_LIMITS = Object.freeze({ snapshotBytes: 1_500_000, requestBytes: 1_520_000, archiveBytes: 6_000_000, ownerCount: 20, ownerArchiveBytes: 30_000_000, chunkBytes: 65_536 });
const initialized = new WeakMap<Database, Promise<unknown>>();
const encoder = new TextEncoder();
interface DeliveryRow {
  id: string; owner_id: string; request_id: string; mode: 'test'; document_id: string; revision: number; name: string;
  snapshot_hash: string; project_hash: string; snapshot_json: string; manifest_json: string;
  archive_sha256: string; archive_bytes: number; chunk_count: number; created_at: string;
}
/** Separate opt-in TEST storage. This does not enable checkout, entitlement, or the legacy cloud format. */
export function requireProjectDelivery(env: ProjectDeliveryEnv) {
  let site: URL;
  try { site = new URL(env.PUBLIC_SITE_URL || ''); } catch { throw new HttpError(503, 'Project delivery is not configured.'); }
  const local = ['localhost', '127.0.0.1', '[::1]'].includes(site.hostname) && env.EIDOS_LOCAL_TEST === 'true';
  if (env.EIDOS_PLAYGROUND_DELIVERY_MODE !== 'test' || /(^|\.)eidos-works\.com$/i.test(site.hostname) || site.hostname.toLowerCase() === 'eidosworks.pages.dev' || (!local && site.protocol !== 'https:'))
    throw new HttpError(503, 'Project delivery is not enabled. Local review and downloads remain available.');
}
export async function ensureProjectDelivery(env: ProjectDeliveryEnv) {
  requireProjectDelivery(env);
  const database = db(env);
  let ready = initialized.get(database);
  if (!ready) { ready = database.batch(schema.split(';').map(value => value.trim()).filter(Boolean).map(value => database.prepare(value))); initialized.set(database, ready); }
  try { await ready; } catch (error) { initialized.delete(database); throw error; }
}
function requestIdentifier(value: unknown): string {
  if (typeof value !== 'string' || !/^[a-zA-Z0-9_-]{8,100}$/.test(value)) throw new HttpError(400, 'Provide a stable snapshot request reference.');
  return value;
}
function snapshotIdentifier(value: unknown): string {
  if (typeof value !== 'string' || !/^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/.test(value)) throw new HttpError(400, 'Invalid frozen project reference.');
  return value;
}
function publicDelivery(row: DeliveryRow) {
  return { id: row.id, ownerId: row.owner_id, requestId: row.request_id, mode: row.mode, documentId: row.document_id, revision: row.revision,
    name: row.name, snapshotHash: row.snapshot_hash, projectHash: row.project_hash, archiveSha256: row.archive_sha256,
    archiveBytes: row.archive_bytes, createdAt: row.created_at, state: 'frozen' as const,
    entitlement: 'not-verified' as const, checkout: 'disabled' as const, limits: DELIVERY_LIMITS };
}
async function ownedRow(env: ProjectDeliveryEnv, owner: string, id: unknown) {
  await ensureProjectDelivery(env);
  const row = await db(env).prepare('SELECT * FROM eidos_pg_delivery_snapshots WHERE id=? AND owner_id=? AND mode=?').bind(snapshotIdentifier(id), owner, 'test').first<DeliveryRow>();
  if (!row) throw new HttpError(404, 'Frozen project not found.');
  return row;
}
function base64(bytes: Uint8Array) {
  let binary = '';
  for (let offset = 0; offset < bytes.length; offset += 8192) binary += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
  return btoa(binary);
}
/** Revalidate and render on the server. Browser HTML, archives, ownership, and claimed hashes are never authoritative. */
export async function freezeProjectDelivery(env: ProjectDeliveryEnv, owner: string, input: Record<string, unknown>) {
  requireProjectDelivery(env);
  if (Object.keys(input).some(key => !['requestId', 'snapshot'].includes(key))) throw new HttpError(400, 'Only a project snapshot and request reference are accepted.');
  const requestId = requestIdentifier(input.requestId);
  if (!record(input.snapshot) || !record(input.snapshot.project) || input.snapshot.project.schemaVersion !== 5)
    throw new HttpError(400, 'Frozen delivery accepts the current schema 5 page project only.');
  if (encoder.encode(JSON.stringify(input.snapshot)).length > DELIVERY_LIMITS.snapshotBytes) throw new HttpError(413, 'Server snapshots are limited to 1.5 MB. Reduce image sizes; your local package is unchanged.');
  let snapshot: ProjectSnapshot;
  try { snapshot = await validateProjectSnapshot(input.snapshot); }
  catch (error) { throw new HttpError(400, error instanceof Error ? error.message : 'Invalid frozen project.'); }
  const snapshotJson = canonicalJson(snapshot);
  if (encoder.encode(snapshotJson).length > DELIVERY_LIMITS.snapshotBytes) throw new HttpError(413, 'The frozen source exceeds the server snapshot limit.');
  await ensureProjectDelivery(env);
  const database = db(env);
  const existing = await database.prepare('SELECT * FROM eidos_pg_delivery_snapshots WHERE owner_id=? AND request_id=?').bind(owner, requestId).first<DeliveryRow>();
  if (existing) {
    if (existing.snapshot_hash !== snapshot.snapshotHash || existing.snapshot_json !== snapshotJson) throw new HttpError(409, 'This request reference belongs to a different frozen project.');
    return publicDelivery(existing);
  }
  let frozen;
  try { frozen = await buildFrozenPackage(snapshot); }
  catch (error) { throw new HttpError(400, error instanceof Error ? error.message : 'This project cannot be packaged.'); }
  if (frozen.archive.length > DELIVERY_LIMITS.archiveBytes) throw new HttpError(413, 'Server packages are limited to 6 MB. Reduce image sizes; your local package is unchanged.');
  const id = crypto.randomUUID(), createdAt = new Date().toISOString();
  const chunks: string[] = [];
  for (let offset = 0; offset < frozen.archive.length; offset += DELIVERY_LIMITS.chunkBytes) chunks.push(base64(frozen.archive.subarray(offset, offset + DELIVERY_LIMITS.chunkBytes)));
  // The owner quota and idempotent insert are one SQLite write; chunk writes commit in the same transaction.
  const insert = database.prepare(`INSERT OR IGNORE INTO eidos_pg_delivery_snapshots(id,owner_id,request_id,mode,document_id,revision,name,snapshot_hash,project_hash,snapshot_json,manifest_json,archive_sha256,archive_bytes,chunk_count,created_at)
    SELECT ?,?,?,'test',?,?,?,?,?,?,?,?,?,?,?
    WHERE (SELECT COUNT(*) FROM eidos_pg_delivery_snapshots WHERE owner_id=?)<?
      AND (SELECT COALESCE(SUM(archive_bytes),0) FROM eidos_pg_delivery_snapshots WHERE owner_id=?)+?<=?`)
    .bind(id, owner, requestId, snapshot.documentId, snapshot.revision, snapshot.project.name, snapshot.snapshotHash, snapshot.projectHash,
      snapshotJson, canonicalJson(frozen.manifest), frozen.receipt.archiveSha256, frozen.archive.length, chunks.length, createdAt,
      owner, DELIVERY_LIMITS.ownerCount, owner, frozen.archive.length, DELIVERY_LIMITS.ownerArchiveBytes);
  await database.batch([insert, ...chunks.map((chunk, ordinal) => database.prepare(`INSERT INTO eidos_pg_delivery_chunks(snapshot_id,ordinal,data_base64)
    SELECT id,?,? FROM eidos_pg_delivery_snapshots WHERE id=? AND owner_id=?`).bind(ordinal, chunk, id, owner))]);
  const saved = await database.prepare('SELECT * FROM eidos_pg_delivery_snapshots WHERE owner_id=? AND request_id=?').bind(owner, requestId).first<DeliveryRow>();
  if (!saved) throw new HttpError(409, 'This account has reached the TEST snapshot count or storage limit. Existing frozen projects remain available.');
  if (saved.snapshot_hash !== snapshot.snapshotHash || saved.snapshot_json !== snapshotJson) throw new HttpError(409, 'Another request used this reference for a different frozen project.');
  return publicDelivery(saved);
}
/** Source reads are owner-only and do not imply payment or a commercial license. */
export async function readProjectDelivery(env: ProjectDeliveryEnv, owner: string, id: unknown) {
  const row = await ownedRow(env, owner, id);
  const snapshot = JSON.parse(row.snapshot_json) as ProjectSnapshot, manifest = JSON.parse(row.manifest_json) as PackageManifest;
  return { ...publicDelivery(row), snapshot, manifest };
}
/** Internal integration seam: a future signed-order download handler must verify entitlement before calling this. */
export async function readProjectDeliveryArchive(env: ProjectDeliveryEnv, owner: string, id: unknown) {
  const row = await ownedRow(env, owner, id);
  const chunks = (await db(env).prepare('SELECT ordinal,data_base64 FROM eidos_pg_delivery_chunks WHERE snapshot_id=? ORDER BY ordinal').bind(row.id).all<{ ordinal: number; data_base64: string }>()).results;
  if (chunks.length !== row.chunk_count) throw new HttpError(503, 'The frozen package is incomplete. Do not repurchase.');
  const archive = new Uint8Array(row.archive_bytes); let offset = 0;
  try {
    for (let index = 0; index < chunks.length; index++) {
      const chunk = chunks[index], binary = atob(chunk.data_base64);
      if (chunk.ordinal !== index || btoa(binary) !== chunk.data_base64 || binary.length > DELIVERY_LIMITS.chunkBytes || offset + binary.length > archive.length) throw Error();
      archive.set(Uint8Array.from(binary, char => char.charCodeAt(0)), offset); offset += binary.length;
    }
  } catch { throw new HttpError(503, 'The frozen package could not be verified. Do not repurchase.'); }
  if (offset !== row.archive_bytes || await hashBytes(archive) !== row.archive_sha256) throw new HttpError(503, 'The frozen package hash does not match. Do not repurchase.');
  return { ...publicDelivery(row), archive };
}
