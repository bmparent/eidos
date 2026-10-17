-- Additive LIVE storage. Existing TEST tables and immutable rows remain unchanged.
-- Immutable LIVE delivery source is separate from the legacy schema-1 cloud project protocol.
CREATE TABLE IF NOT EXISTS eidos_pg_live_delivery_snapshots (
 id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES eidos_email_members(id), request_id TEXT NOT NULL,
 mode TEXT NOT NULL CHECK(mode='live'), document_id TEXT NOT NULL, revision INTEGER NOT NULL CHECK(revision>=0),
 name TEXT NOT NULL, snapshot_hash TEXT NOT NULL CHECK(length(snapshot_hash)=64), project_hash TEXT NOT NULL CHECK(length(project_hash)=64),
 snapshot_json TEXT NOT NULL CHECK(length(CAST(snapshot_json AS BLOB))<=1500000), manifest_json TEXT NOT NULL,
 archive_sha256 TEXT NOT NULL CHECK(length(archive_sha256)=64), archive_bytes INTEGER NOT NULL CHECK(archive_bytes>0 AND archive_bytes<=6000000),
 chunk_count INTEGER NOT NULL CHECK(chunk_count>0 AND chunk_count<=92), created_at TEXT NOT NULL,
 UNIQUE(owner_id,request_id)
);
CREATE INDEX IF NOT EXISTS pg_live_delivery_owner ON eidos_pg_live_delivery_snapshots(owner_id,created_at);
CREATE TABLE IF NOT EXISTS eidos_pg_live_delivery_chunks (
 snapshot_id TEXT NOT NULL REFERENCES eidos_pg_live_delivery_snapshots(id), ordinal INTEGER NOT NULL CHECK(ordinal>=0 AND ordinal<92),
 data_base64 TEXT NOT NULL CHECK(length(data_base64)>0 AND length(data_base64)<=87384),
 PRIMARY KEY(snapshot_id,ordinal)
);
-- Immutable attachment beside the existing template order. Original archive fields are unchanged.
CREATE TABLE IF NOT EXISTS eidos_pg_live_delivery_orders (
 order_id TEXT PRIMARY KEY REFERENCES eidos_template_orders(order_id), owner_id TEXT NOT NULL REFERENCES eidos_email_members(id),
 delivery_id TEXT NOT NULL REFERENCES eidos_pg_live_delivery_snapshots(id), mode TEXT NOT NULL CHECK(mode='live'),
 document_id TEXT NOT NULL, revision INTEGER NOT NULL CHECK(revision>=0), snapshot_hash TEXT NOT NULL CHECK(length(snapshot_hash)=64),
 project_hash TEXT NOT NULL CHECK(length(project_hash)=64), archive_sha256 TEXT NOT NULL CHECK(length(archive_sha256)=64),
 archive_bytes INTEGER NOT NULL CHECK(archive_bytes>0 AND archive_bytes<=6000000), output TEXT NOT NULL CHECK(output='eidos-local-page-package-v1'),
 download_name TEXT NOT NULL, parent_product_id TEXT NOT NULL, parent_edition_id TEXT NOT NULL CHECK(parent_edition_id='developer'),
 parent_version TEXT NOT NULL, parent_archive_sha256 TEXT NOT NULL CHECK(length(parent_archive_sha256)=64),
 parent_archive_bytes INTEGER NOT NULL CHECK(parent_archive_bytes>0), parent_license_id TEXT NOT NULL,
 merchant_id TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS pg_live_delivery_order_owner ON eidos_pg_live_delivery_orders(owner_id,order_id);
