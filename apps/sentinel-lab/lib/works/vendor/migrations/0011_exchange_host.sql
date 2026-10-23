-- Resident host: bounded private operational receipts without a model or payment ledger.
CREATE TABLE IF NOT EXISTS eidos_exchange_host_events (
 id TEXT PRIMARY KEY,
 request_hash TEXT NOT NULL UNIQUE,
 payload_hash TEXT NOT NULL,
 subject TEXT NOT NULL,
 identity_id TEXT REFERENCES eidos_exchange_identities(id),
 action TEXT NOT NULL CHECK(action IN ('guide','enroll','request-review')),
 topic TEXT NOT NULL,
 outcome TEXT NOT NULL CHECK(outcome IN ('running','completed','pending','closed','declined','failed')),
 note TEXT NOT NULL DEFAULT '',
 response TEXT,
 created_at INTEGER NOT NULL,
 resolved_at INTEGER
);
CREATE INDEX IF NOT EXISTS exchange_host_identity ON eidos_exchange_host_events(identity_id,created_at DESC);
CREATE INDEX IF NOT EXISTS exchange_host_queue ON eidos_exchange_host_events(outcome,created_at);
