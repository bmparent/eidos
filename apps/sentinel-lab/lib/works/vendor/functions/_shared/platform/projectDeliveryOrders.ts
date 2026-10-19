import { db, HttpError, type Database, type Statement } from './core';
import { deliveryTables, requireNewProjectDelivery, ensureProjectDelivery, readProjectDelivery, readProjectDeliveryArchive, requireProjectDelivery, type ProjectDeliveryEnv } from './projectDelivery';
import { hashBytes, parentEditions } from './projectDeliveryRenderer';
import type { TemplateProduct } from './templateCatalog';
import type { TemplateOrder } from './templateShop';
import schema from './projectDeliveryOrdersSchema';

const initialized = new WeakMap<Database, Promise<unknown>>();
type Archive = { bytes: Uint8Array; sha256: string };
export interface DeliveryOrderBinding {
  order_id: string; owner_id: string; delivery_id: string; mode: 'test' | 'live'; document_id: string; revision: number;
  snapshot_hash: string; project_hash: string; archive_sha256: string; archive_bytes: number; output: 'eidos-local-page-package-v1';
  download_name: string; parent_product_id: string; parent_edition_id: 'developer'; parent_version: string;
  parent_archive_sha256: string; parent_archive_bytes: number; parent_license_id: string; merchant_id: string; created_at: string;
}
export async function ensureDeliveryOrders(env: ProjectDeliveryEnv) {
  await ensureProjectDelivery(env);
  if (requireProjectDelivery(env) === 'live') return; // The additive LIVE schema includes order bindings.
  const database = db(env); let ready = initialized.get(database);
  if (!ready) { ready = database.batch(schema.split(';').map(value => value.trim()).filter(Boolean).map(value => database.prepare(value))); initialized.set(database, ready); }
  try { await ready; } catch (error) { initialized.delete(database); throw error; }
}
/** Existing original/guest orders do not acquire a new schema requirement or an invented attachment. */
export async function deliveryOrderBinding(env: ProjectDeliveryEnv, orderId: string) {
  const database = db(env);
  const bindings: DeliveryOrderBinding[] = [];
  for (const table of ['eidos_pg_delivery_orders', 'eidos_pg_live_delivery_orders']) {
    if (!await database.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name=?").bind(table).first()) continue;
    const binding = await database.prepare(`SELECT * FROM ${table} WHERE order_id=?`).bind(orderId).first<DeliveryOrderBinding>();
    if (binding) bindings.push(binding);
  }
  if (bindings.length > 1) throw new HttpError(503, 'Ambiguous purchase attachment. Contact support; do not repurchase.');
  return bindings[0] || null;
}
export async function verifiedDeliveryParent(product: TemplateProduct, archive: Archive) {
  const parent = parentEditions.find(value => value.editionId === product.id);
  if (!parent || product.editionId !== 'developer' || parent.version !== product.version || parent.platform !== 'developer' || parent.sha256 !== archive.sha256 || parent.bytes !== archive.bytes.length || await hashBytes(archive.bytes) !== archive.sha256)
    throw new HttpError(409, 'The selected parent edition does not match its verified immutable archive.');
  return parent;
}
export async function prepareDeliveryOrder(env: ProjectDeliveryEnv, owner: string | undefined, deliveryId: unknown, product: TemplateProduct, archive: Archive) {
  await requireNewProjectDelivery(env);
  if (!owner) throw new HttpError(401, 'Sign in to attach your approved page to this purchase.');
  const parent = await verifiedDeliveryParent(product, archive);
  const delivery = await readProjectDelivery(env, owner, deliveryId);
  const selected = delivery.snapshot.parent;
  if (!selected || selected.editionId !== parent.editionId || selected.version !== parent.version || selected.sha256 !== parent.sha256 || selected.bytes !== parent.bytes)
    throw new HttpError(409, 'Freeze this page with the selected parent edition before checkout.');
  // The exact bytes must already be intact before any external checkout can start.
  const stored = await readProjectDeliveryArchive(env, owner, delivery.id);
  if (stored.archiveSha256 !== delivery.archiveSha256 || stored.archiveBytes !== delivery.archiveBytes) throw new HttpError(503, 'The approved package is unavailable. Do not repurchase.');
  await ensureDeliveryOrders(env);
  return { delivery, parent, product };
}
export type PreparedDeliveryOrder = Awaited<ReturnType<typeof prepareDeliveryOrder>>;
/** Included in the transaction that first creates the order, before session dispatch. There is no later attachment mutation. */
export function insertDeliveryOrder(env: ProjectDeliveryEnv, orderId: string, prepared: PreparedDeliveryOrder, merchantId: string): Statement {
  const { delivery, parent, product } = prepared, tables = deliveryTables(env);
  if (delivery.mode !== requireProjectDelivery(env)) throw new HttpError(409, 'The approved page belongs to a different payment mode.');
  if (!/^acct_[A-Za-z0-9]+$/.test(merchantId)) throw new HttpError(503, 'The payment merchant must be verified before attaching a page.');
  return db(env).prepare(`INSERT INTO ${tables.orders}(order_id,owner_id,delivery_id,mode,document_id,revision,snapshot_hash,project_hash,archive_sha256,archive_bytes,output,download_name,parent_product_id,parent_edition_id,parent_version,parent_archive_sha256,parent_archive_bytes,parent_license_id,merchant_id,created_at)
    SELECT t.order_id,s.owner_id,s.id,s.mode,s.document_id,s.revision,s.snapshot_hash,s.project_hash,s.archive_sha256,s.archive_bytes,?,?,?,?,?,?,?,?,?,?
    FROM eidos_template_orders t JOIN eidos_orders o ON o.id=t.order_id JOIN ${tables.snapshots} s ON s.id=? AND s.owner_id=t.member_id
    WHERE t.order_id=? AND o.status='pending' AND o.session_id IS NULL AND t.product_id=? AND t.version=? AND t.archive_sha256=?`)
    .bind(delivery.snapshot.output, `eidos-approved-page-${delivery.id}.zip`, product.id, product.editionId, parent.version, parent.sha256, parent.bytes, parent.license.id, merchantId, new Date().toISOString(),
      delivery.id, orderId, product.id, parent.version, parent.sha256);
}
export function assertDeliveryAttempt(binding: DeliveryOrderBinding | null, deliveryId: unknown) {
  if ((binding?.delivery_id ?? undefined) !== deliveryId) throw new HttpError(409, 'This checkout attempt belongs to a different approved page or original package.');
}
export function assertDeliveryOwner(binding: DeliveryOrderBinding, owner: string | undefined) {
  if (!owner) throw new HttpError(401, 'Sign in to the account that approved this page.');
  if (binding.owner_id !== owner) throw new HttpError(403, 'This approved page belongs to another account.');
}
export function deliveryOrderMetadata(binding: DeliveryOrderBinding, state: 'pending' | 'ready' | 'revoked') {
  return { state, deliveryId: binding.delivery_id, ownerId: binding.owner_id, mode: binding.mode, documentId: binding.document_id, revision: binding.revision,
    snapshotHash: binding.snapshot_hash, projectHash: binding.project_hash, archiveSha256: binding.archive_sha256, archiveBytes: binding.archive_bytes,
    output: binding.output, downloadName: binding.download_name,
    parent: { productId: binding.parent_product_id, editionId: binding.parent_edition_id, version: binding.parent_version,
      archiveSha256: binding.parent_archive_sha256, archiveBytes: binding.parent_archive_bytes, licenseId: binding.parent_license_id } };
}
export async function deliveryOrderRevoked(env: ProjectDeliveryEnv, order: TemplateOrder) {
  return order.status === 'refunded' || order.fulfillment_status === 'revoked' || Boolean(order.payment_intent && await db(env).prepare('SELECT payment_intent FROM eidos_revoked_payments WHERE payment_intent=?').bind(order.payment_intent).first());
}
export async function approvedOrderArchive(env: ProjectDeliveryEnv, owner: string | undefined, order: TemplateOrder) {
  const binding = await deliveryOrderBinding(env, order.id);
  if (!binding) throw new HttpError(404, 'This purchase has no approved page attachment.');
  assertDeliveryOwner(binding, owner);
  if (order.member_id !== binding.owner_id || binding.mode !== requireProjectDelivery(env) || !order.session_id?.startsWith('cs_' + binding.mode + '_') || order.status !== 'paid' || order.fulfillment_status !== 'ready' || await deliveryOrderRevoked(env, order))
    throw new HttpError(403, 'A matching verified, unrevoked matching-mode purchase is required for this approved page.');
  if (order.product_id !== binding.parent_product_id || order.edition_id !== binding.parent_edition_id || order.version !== binding.parent_version || order.archive_sha256 !== binding.parent_archive_sha256)
    throw new HttpError(503, 'The approved page does not match the immutable purchase. Contact support.');
  const frozen = await readProjectDeliveryArchive(env, binding.owner_id, binding.delivery_id);
  if (frozen.snapshotHash !== binding.snapshot_hash || frozen.projectHash !== binding.project_hash || frozen.archiveSha256 !== binding.archive_sha256 || frozen.archiveBytes !== binding.archive_bytes || frozen.documentId !== binding.document_id || frozen.revision !== binding.revision)
    throw new HttpError(503, 'The approved page differs from its purchase receipt. Do not repurchase.');
  return { bytes: frozen.archive, sha256: binding.archive_sha256, downloadName: binding.download_name };
}
