import { db, hash, HttpError, type PlatformEnv } from './core';
import { kitArchiveBase64 } from './kitArchive';
import { legacyKitArchiveBase64 } from './legacyKitArchive';
import { memberFromRequest, requireMember } from './memberAuth';
import type { Order } from './shop';

// Additive tables; old orders, receipts and refund tombstones remain authoritative.
export async function ensureKitDelivery(env: PlatformEnv) {
  await db(env).batch([
    db(env).prepare('CREATE TABLE IF NOT EXISTS eidos_kit_archives (digest TEXT PRIMARY KEY, version TEXT NOT NULL, data TEXT NOT NULL)'),
    db(env).prepare('CREATE TABLE IF NOT EXISTS eidos_kit_attempts (token_hash TEXT PRIMARY KEY, order_id TEXT NOT NULL UNIQUE, owner_id TEXT, use_case TEXT NOT NULL, expires INTEGER NOT NULL, session_url TEXT, archive_digest TEXT NOT NULL, site_origin TEXT NOT NULL)'),
    db(env).prepare('CREATE TABLE IF NOT EXISTS eidos_kit_recovery (order_id TEXT PRIMARY KEY, email_hash TEXT NOT NULL)'),
  ]);
}
export async function kitAttempt(request: Request, env: PlatformEnv, receipt: string, useCase: string, siteOrigin: string) {
  await ensureKitDelivery(env);
  const database = db(env), tokenHash = await hash(receipt);
  const member = await memberFromRequest(request, env, false);
  const id = crypto.randomUUID(), expires = Math.floor(Date.now() / 1000) + 3600;
  const digest = await archiveDigest(kitArchiveBase64);
  await database.batch([
    database.prepare('INSERT OR IGNORE INTO eidos_kit_archives(digest,version,data) VALUES(?,?,?)').bind(digest, 'cinematic-starter-v1', kitArchiveBase64),
    database.prepare('INSERT OR IGNORE INTO eidos_kit_attempts(token_hash,order_id,owner_id,use_case,expires,archive_digest,site_origin) VALUES(?,?,?,?,?,?,?)').bind(tokenHash, id, member?.id || null, useCase, expires, digest, siteOrigin),
    database.prepare('INSERT OR IGNORE INTO eidos_orders(id,receipt_hash,created_at) SELECT order_id,?,? FROM eidos_kit_attempts WHERE token_hash=?').bind(tokenHash, new Date().toISOString(), tokenHash),
  ]);
  const attempt = await database.prepare('SELECT a.*,o.status FROM eidos_kit_attempts a JOIN eidos_orders o ON o.id=a.order_id WHERE token_hash=?').bind(tokenHash).first<{
    order_id: string; owner_id: string | null; use_case: string; expires: number; session_url: string | null; status: string; site_origin: string;
  }>();
  if (!attempt || (attempt.owner_id && attempt.owner_id !== member?.id)) throw new HttpError(403, 'Sign in to the account that started this purchase.');
  if (attempt.use_case !== useCase || attempt.site_origin !== siteOrigin) throw new HttpError(409, 'This purchase attempt has different options. Resume its original options or deliberately start a new purchase.');
  if (attempt.status !== 'pending') throw new HttpError(409, 'This purchase already has a result. Open your receipt or account purchases.');
  if (attempt.expires <= Math.floor(Date.now() / 1000) + 60) throw new HttpError(409, 'This checkout attempt has expired. Check your purchases before starting a new attempt.');
  return attempt;
}
export async function archiveDigest(data: string) {
  const bytes = Uint8Array.from(atob(data), c => c.charCodeAt(0));
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
}
export async function kitDownload(env: PlatformEnv, order: Order) {
  if (order.status !== 'paid') throw new HttpError(403, 'A verified, completed purchase is required to download this package.');
  await ensureKitDelivery(env);
  const archive = await db(env).prepare('SELECT r.data,r.digest FROM eidos_kit_attempts a JOIN eidos_kit_archives r ON r.digest=a.archive_digest WHERE a.order_id=?').bind(order.id).first<{ data: string; digest: string }>();
  // Historical v1 orders retain the original v1 artifact; never substitute a new version.
  const data = archive?.data || legacyKitArchiveBase64, digest = await archiveDigest(data);
  if (archive && digest !== archive.digest) throw new HttpError(503, 'The purchased archive needs a support check. Please try again later.');
  return new Response(Uint8Array.from(atob(data), c => c.charCodeAt(0)), { headers: {
    'content-type': 'application/zip', 'content-disposition': 'attachment; filename="eidos-cinematic-starter-v1.zip"',
    'cache-control': 'private, no-store', 'referrer-policy': 'no-referrer', 'x-content-type-options': 'nosniff', 'x-artifact-sha256': digest,
  } });
}
export async function kitOwner(request: Request, env: PlatformEnv) {
  const member = await requireMember(request, env, false);
  await ensureKitDelivery(env);
  return { id: member.id, emailHash: await hash(member.email.trim().toLowerCase()) };
}
