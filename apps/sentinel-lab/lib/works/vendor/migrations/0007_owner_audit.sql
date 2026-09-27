-- Owner activity only: no headers, tokens, request bodies, prompts or IP addresses.
CREATE TABLE IF NOT EXISTS eidos_owner_audit(id TEXT PRIMARY KEY,actor TEXT NOT NULL,method TEXT NOT NULL,path TEXT NOT NULL,event TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS eidos_owner_audit_recent ON eidos_owner_audit(created_at DESC,id DESC);
