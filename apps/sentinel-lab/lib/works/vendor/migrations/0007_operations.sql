-- Owner console state only. Customer, payment, and provider records remain in their authoritative tables.
CREATE TABLE IF NOT EXISTS eidos_ops_work (
 id TEXT PRIMARY KEY, title TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '',
 status TEXT NOT NULL CHECK(status IN ('open','in_progress','blocked','done')),
 severity TEXT NOT NULL CHECK(severity IN ('critical','high','normal','low')),
 evidence_url TEXT, due_at TEXT, source TEXT NOT NULL, created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS eidos_ops_work_queue ON eidos_ops_work(status,severity,updated_at DESC);
CREATE TABLE IF NOT EXISTS eidos_ops_work_notes (
 id TEXT PRIMARY KEY, work_id TEXT NOT NULL REFERENCES eidos_ops_work(id),
 actor_sub TEXT NOT NULL, body TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS eidos_ops_work_notes_recent ON eidos_ops_work_notes(work_id,created_at DESC);
CREATE TABLE IF NOT EXISTS eidos_ops_audit (
 id TEXT PRIMARY KEY, actor_sub TEXT NOT NULL, action TEXT NOT NULL, target_type TEXT NOT NULL,
 target_id TEXT NOT NULL, before_safe TEXT, after_safe TEXT, reason TEXT,
 outcome TEXT NOT NULL, correlation_id TEXT NOT NULL, source_revision TEXT NOT NULL,
 created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS eidos_ops_audit_recent ON eidos_ops_audit(created_at DESC);
CREATE TABLE IF NOT EXISTS eidos_ops_connector_checkpoints (
 source TEXT NOT NULL, environment TEXT NOT NULL, observed_at TEXT, last_success_at TEXT,
 stale_after TEXT, status TEXT NOT NULL, error_code TEXT, cursor TEXT,
 PRIMARY KEY(source,environment)
);
