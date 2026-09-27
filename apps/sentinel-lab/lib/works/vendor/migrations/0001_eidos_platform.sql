-- Additive schema: does not touch Snapshot storage or the Sentinel research application.
CREATE TABLE IF NOT EXISTS eidos_quotas(bucket TEXT NOT NULL, period INTEGER NOT NULL, used INTEGER NOT NULL CHECK(used>=0), expires INTEGER NOT NULL, PRIMARY KEY(bucket,period));
CREATE INDEX IF NOT EXISTS quotas_expiry ON eidos_quotas(expires);
CREATE TABLE IF NOT EXISTS eidos_answer_cache(id TEXT PRIMARY KEY, prompt_hash TEXT NOT NULL, answer TEXT, expires INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS answer_cache_expiry ON eidos_answer_cache(expires);
CREATE TABLE IF NOT EXISTS eidos_threads(id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT NOT NULL, category TEXT NOT NULL CHECK(category IN ('build','design','agents')), author TEXT NOT NULL, author_type TEXT NOT NULL CHECK(author_type IN ('guest','agent','studio')), owner_id TEXT, status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','published','rejected')), allow_assistant INTEGER NOT NULL DEFAULT 0, request_assistant INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, published_at TEXT);
CREATE INDEX IF NOT EXISTS threads_public ON eidos_threads(status,category,created_at DESC);
CREATE TABLE IF NOT EXISTS eidos_replies(id TEXT PRIMARY KEY, thread_id TEXT NOT NULL REFERENCES eidos_threads(id) ON DELETE CASCADE, body TEXT NOT NULL, author TEXT NOT NULL, author_type TEXT NOT NULL CHECK(author_type IN ('guest','agent','studio','eidos')), owner_id TEXT, status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','published','rejected')), created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS replies_public ON eidos_replies(thread_id,status,created_at);
CREATE UNIQUE INDEX IF NOT EXISTS one_eidos_reply ON eidos_replies(thread_id) WHERE author_type='eidos';
CREATE TABLE IF NOT EXISTS eidos_agents(id TEXT PRIMARY KEY, name TEXT NOT NULL, key_hash TEXT NOT NULL UNIQUE, profile_url TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS eidos_orders(id TEXT PRIMARY KEY, receipt_hash TEXT NOT NULL UNIQUE, session_id TEXT UNIQUE, status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','paid','refunded')), created_at TEXT NOT NULL, paid_at TEXT, payment_intent TEXT);
CREATE TABLE IF NOT EXISTS eidos_stripe_events(id TEXT PRIMARY KEY, received_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS eidos_revoked_payments(payment_intent TEXT PRIMARY KEY, received_at TEXT NOT NULL);
