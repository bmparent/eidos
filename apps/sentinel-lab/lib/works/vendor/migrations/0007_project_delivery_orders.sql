-- Immutable attachment beside the existing template order. Original archive fields are unchanged.
CREATE TABLE IF NOT EXISTS eidos_pg_delivery_orders (
 order_id TEXT PRIMARY KEY REFERENCES eidos_template_orders(order_id), owner_id TEXT NOT NULL REFERENCES eidos_email_members(id),
 delivery_id TEXT NOT NULL REFERENCES eidos_pg_delivery_snapshots(id), mode TEXT NOT NULL CHECK(mode='test'),
 document_id TEXT NOT NULL, revision INTEGER NOT NULL CHECK(revision>=0), snapshot_hash TEXT NOT NULL CHECK(length(snapshot_hash)=64),
 project_hash TEXT NOT NULL CHECK(length(project_hash)=64), archive_sha256 TEXT NOT NULL CHECK(length(archive_sha256)=64),
 archive_bytes INTEGER NOT NULL CHECK(archive_bytes>0 AND archive_bytes<=6000000), output TEXT NOT NULL CHECK(output='eidos-local-page-package-v1'),
 download_name TEXT NOT NULL, parent_product_id TEXT NOT NULL, parent_edition_id TEXT NOT NULL CHECK(parent_edition_id='developer'),
 parent_version TEXT NOT NULL, parent_archive_sha256 TEXT NOT NULL CHECK(length(parent_archive_sha256)=64),
 parent_archive_bytes INTEGER NOT NULL CHECK(parent_archive_bytes>0), parent_license_id TEXT NOT NULL,
 merchant_id TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS pg_delivery_order_owner ON eidos_pg_delivery_orders(owner_id,order_id);
