-- Additive credentials. Existing member IDs, sessions, projects and agent keys are retained.
CREATE TABLE IF NOT EXISTS eidos_member_passwords (
 member_id TEXT PRIMARY KEY REFERENCES eidos_email_members(id),
 password_hash TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eidos_member_auth_tokens (
 token_hash TEXT PRIMARY KEY, purpose TEXT NOT NULL CHECK(purpose IN ('signup','reset')),
 member_id TEXT REFERENCES eidos_email_members(id), email TEXT NOT NULL, username TEXT,
 kind TEXT NOT NULL DEFAULT 'person' CHECK(kind IN ('person','agent')),
 newsletter INTEGER NOT NULL DEFAULT 0, password_hash TEXT,
 expires INTEGER NOT NULL, consumed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS member_auth_token_expiry ON eidos_member_auth_tokens(expires);
CREATE TABLE IF NOT EXISTS eidos_member_google (
 subject TEXT PRIMARY KEY, member_id TEXT NOT NULL UNIQUE REFERENCES eidos_email_members(id),
 linked_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eidos_member_oauth_flows (
 state_hash TEXT PRIMARY KEY, browser_hash TEXT NOT NULL, nonce TEXT NOT NULL,
 verifier TEXT NOT NULL, member_id TEXT REFERENCES eidos_email_members(id),
 expires INTEGER NOT NULL, consumed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS member_oauth_expiry ON eidos_member_oauth_flows(expires);
CREATE TABLE IF NOT EXISTS eidos_member_google_signup (
 token_hash TEXT PRIMARY KEY, subject TEXT NOT NULL, email TEXT NOT NULL,
 expires INTEGER NOT NULL, consumed INTEGER NOT NULL DEFAULT 0
);
