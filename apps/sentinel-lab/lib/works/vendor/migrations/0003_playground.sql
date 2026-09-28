CREATE TABLE IF NOT EXISTS eidos_pg_projects (
 id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES eidos_email_members(id), name TEXT NOT NULL,
 head TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS eidos_pg_owner ON eidos_pg_projects(owner_id, updated_at);
CREATE TABLE IF NOT EXISTS eidos_pg_revisions (
 id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES eidos_pg_projects(id), parent TEXT,
 document TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS eidos_pg_revision_project ON eidos_pg_revisions(project_id, created_at);
CREATE TABLE IF NOT EXISTS eidos_pg_assets (
 owner_id TEXT NOT NULL REFERENCES eidos_email_members(id), hash TEXT NOT NULL, data TEXT NOT NULL,
 PRIMARY KEY(owner_id,hash)
);
CREATE TABLE IF NOT EXISTS eidos_pg_orders (
 id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES eidos_email_members(id), request_id TEXT NOT NULL,
 project_id TEXT NOT NULL, revision_id TEXT NOT NULL, name TEXT NOT NULL, archive TEXT NOT NULL,
 amount INTEGER NOT NULL, mode TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
 session_id TEXT, checkout_url TEXT, payment_intent TEXT, created_at TEXT NOT NULL, paid_at TEXT,
 UNIQUE(owner_id,request_id)
);
CREATE TABLE IF NOT EXISTS eidos_pg_events (id TEXT PRIMARY KEY, received_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS eidos_pg_revoked (payment_intent TEXT PRIMARY KEY);
