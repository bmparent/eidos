-- Additive: existing eidos_orders and Cinematic Starter purchases are preserved.
-- This table stores immutable fulfillment snapshots beside the existing ledger.
CREATE TABLE IF NOT EXISTS eidos_template_orders(order_id TEXT PRIMARY KEY REFERENCES eidos_orders(id), product_id TEXT NOT NULL, edition_id TEXT NOT NULL, version TEXT NOT NULL, price_cents INTEGER NOT NULL, archive_key TEXT NOT NULL, archive_sha256 TEXT NOT NULL, download_name TEXT NOT NULL, guide_path TEXT NOT NULL, member_id TEXT, email TEXT, email_hash TEXT, receipt_expires INTEGER NOT NULL, session_url TEXT, checkout_expires INTEGER NOT NULL, mail_status TEXT NOT NULL DEFAULT 'pending', mail_provider_id TEXT, fulfillment_type TEXT NOT NULL DEFAULT 'download' CHECK(fulfillment_type IN ('download','manual_setup')), fulfillment_status TEXT NOT NULL DEFAULT 'awaiting_payment' CHECK(fulfillment_status IN ('awaiting_payment','ready','setup_pending','revoked')), created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS template_order_owner ON eidos_template_orders(member_id,created_at);
CREATE INDEX IF NOT EXISTS template_order_email ON eidos_template_orders(email_hash,created_at);
CREATE TABLE IF NOT EXISTS eidos_template_recovery(token_hash TEXT PRIMARY KEY, order_id TEXT NOT NULL REFERENCES eidos_orders(id), expires INTEGER NOT NULL, used INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS eidos_template_downloads(token_hash TEXT PRIMARY KEY, order_id TEXT NOT NULL REFERENCES eidos_orders(id), expires INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS template_download_expiry ON eidos_template_downloads(expires);
CREATE TABLE IF NOT EXISTS eidos_template_payment_failures(order_id TEXT PRIMARY KEY REFERENCES eidos_orders(id), failed_at TEXT NOT NULL, event_id TEXT NOT NULL);
