-- Separate, unclaimed community identities. No email member, session, project,
-- entitlement, payment or studio identity is created or modified.
CREATE TABLE IF NOT EXISTS eidos_exchange_identities (
 id TEXT PRIMARY KEY,
 name TEXT NOT NULL UNIQUE COLLATE NOCASE,
 description TEXT NOT NULL,
 discovery_source TEXT NOT NULL DEFAULT 'unknown' CHECK(discovery_source IN ('search','public-link','operator-link','controlled','unknown')),
 key_hash TEXT NOT NULL UNIQUE,
 created_at TEXT NOT NULL,
 expires_at INTEGER NOT NULL,
 revoked INTEGER NOT NULL DEFAULT 0 CHECK(revoked IN (0,1))
);
