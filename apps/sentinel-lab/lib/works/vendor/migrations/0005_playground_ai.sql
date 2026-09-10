CREATE TABLE IF NOT EXISTS eidos_pg_ai_control (id INTEGER PRIMARY KEY CHECK(id=1), enabled INTEGER NOT NULL DEFAULT 0 CHECK(enabled IN (0,1)));
INSERT OR IGNORE INTO eidos_pg_ai_control(id,enabled) VALUES(1,0);
CREATE TABLE IF NOT EXISTS eidos_pg_ai_requests (
 id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, project_id TEXT NOT NULL, request_hash TEXT NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('reserved','completed','rejected','unknown')), reserved_micro INTEGER NOT NULL CHECK(reserved_micro>=0), actual_micro INTEGER,
 model TEXT NOT NULL, pricing TEXT NOT NULL, kind TEXT NOT NULL, settings TEXT NOT NULL, provider_id TEXT,
 usage TEXT, result TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS eidos_pg_ai_owner ON eidos_pg_ai_requests(owner_id,state);
