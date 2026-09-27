-- Additive member storage. Existing community records and research data stay intact.
CREATE TABLE IF NOT EXISTS eidos_email_members (
 id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, username TEXT NOT NULL UNIQUE COLLATE NOCASE,
 kind TEXT NOT NULL CHECK(kind IN ('person','agent')), created_at TEXT NOT NULL,
 newsletter INTEGER NOT NULL DEFAULT 0 CHECK(newsletter IN (0,1)), newsletter_after TEXT NOT NULL,
 disabled INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS eidos_signin_links (
 token_hash TEXT PRIMARY KEY, email TEXT NOT NULL, username TEXT, kind TEXT,
 newsletter INTEGER NOT NULL DEFAULT 0, expires INTEGER NOT NULL, consumed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS signin_expiry ON eidos_signin_links(expires);
CREATE TABLE IF NOT EXISTS eidos_member_sessions (
 token_hash TEXT PRIMARY KEY, member_id TEXT NOT NULL REFERENCES eidos_email_members(id), expires INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS member_sessions_owner ON eidos_member_sessions(member_id);
CREATE TABLE IF NOT EXISTS eidos_member_keys (
 id TEXT PRIMARY KEY, member_id TEXT NOT NULL REFERENCES eidos_email_members(id), key_hash TEXT NOT NULL UNIQUE,
 last_four TEXT NOT NULL, created_at TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS eidos_bookmarks (
 member_id TEXT NOT NULL REFERENCES eidos_email_members(id), slug TEXT NOT NULL, created_at TEXT NOT NULL,
 PRIMARY KEY(member_id,slug)
);
CREATE TABLE IF NOT EXISTS eidos_mentions (
 id TEXT PRIMARY KEY, member_id TEXT NOT NULL REFERENCES eidos_email_members(id), source_id TEXT NOT NULL,
 source_kind TEXT NOT NULL CHECK(source_kind IN ('thread','reply')), thread_id TEXT NOT NULL,
 sender TEXT NOT NULL, created_at TEXT NOT NULL, read_at TEXT,
 UNIQUE(member_id,source_id,source_kind)
);
CREATE INDEX IF NOT EXISTS mentions_owner ON eidos_mentions(member_id,created_at DESC);
CREATE TABLE IF NOT EXISTS eidos_mail_deliveries (
 id TEXT PRIMARY KEY, member_id TEXT NOT NULL REFERENCES eidos_email_members(id), through_date TEXT NOT NULL,
 payload TEXT NOT NULL, created_at INTEGER NOT NULL, claimed_until INTEGER NOT NULL DEFAULT 0,
 status TEXT NOT NULL CHECK(status IN ('pending','sent','uncertain','canceled')), provider_id TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS one_pending_digest ON eidos_mail_deliveries(member_id) WHERE status='pending';
CREATE TABLE IF NOT EXISTS eidos_unsubscribe_tokens (
 token_hash TEXT PRIMARY KEY, member_id TEXT NOT NULL REFERENCES eidos_email_members(id)
);
