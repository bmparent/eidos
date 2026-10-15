-- USD microdollars: 1 USD = 1000000 units. Integer CHECKs fail closed.
-- Additive accounting inside the existing Works commerce database, never a second wallet service.
CREATE TABLE IF NOT EXISTS eidos_pg_ai_policies (
 mode TEXT PRIMARY KEY CHECK(mode IN ('test','live')), enabled INTEGER NOT NULL DEFAULT 0 CHECK(enabled IN (0,1)),
 provider_budget INTEGER NOT NULL CHECK(typeof(provider_budget)='integer' AND provider_budget>=0 AND provider_budget<=1000000000000),
 rate_json TEXT NOT NULL, paused_reason TEXT, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eidos_pg_ai_accounts (
 owner_id TEXT NOT NULL REFERENCES eidos_email_members(id), mode TEXT NOT NULL CHECK(mode IN ('test','live')),
 requested_limit INTEGER NOT NULL CHECK(typeof(requested_limit)='integer' AND requested_limit>=0 AND requested_limit<=1000000000000),
 frozen INTEGER NOT NULL DEFAULT 0 CHECK(frozen IN (0,1)), updated_at TEXT NOT NULL,
 PRIMARY KEY(owner_id,mode)
);
CREATE TABLE IF NOT EXISTS eidos_pg_ai_grants (
 id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, mode TEXT NOT NULL CHECK(mode IN ('test','live')),
 order_id TEXT NOT NULL REFERENCES eidos_orders(id), grant_version TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('included','purchased')),
 amount INTEGER NOT NULL CHECK(typeof(amount)='integer' AND amount>0),
 revoked_amount INTEGER NOT NULL DEFAULT 0 CHECK(typeof(revoked_amount)='integer' AND revoked_amount>=0 AND revoked_amount<=amount),
 reversed INTEGER NOT NULL DEFAULT 0 CHECK(reversed IN (0,1)), payment_intent TEXT NOT NULL, created_at TEXT NOT NULL,
 UNIQUE(order_id,kind), FOREIGN KEY(owner_id,mode) REFERENCES eidos_pg_ai_accounts(owner_id,mode)
);
CREATE TABLE IF NOT EXISTS eidos_pg_ai_paid_receipts (
 order_id TEXT PRIMARY KEY REFERENCES eidos_orders(id), mode TEXT NOT NULL CHECK(mode IN ('test','live')),
 merchant_id TEXT NOT NULL, event_id TEXT NOT NULL REFERENCES eidos_stripe_events(id), currency TEXT NOT NULL CHECK(currency='usd'),
 amount_cents INTEGER NOT NULL CHECK(typeof(amount_cents)='integer' AND amount_cents>0),
 payment_intent TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eidos_pg_ai_checkout_merchants (
 order_id TEXT PRIMARY KEY REFERENCES eidos_orders(id), mode TEXT NOT NULL CHECK(mode IN ('test','live')),
 merchant_id TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eidos_pg_ai_actions (
 id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, mode TEXT NOT NULL CHECK(mode IN ('test','live')),
 idempotency_key TEXT NOT NULL, fingerprint TEXT NOT NULL, document_id TEXT NOT NULL, revision TEXT NOT NULL,
 scope TEXT NOT NULL, payload_json TEXT NOT NULL, period TEXT NOT NULL, resets_at TEXT NOT NULL,
 rate_json TEXT NOT NULL, estimate INTEGER NOT NULL CHECK(typeof(estimate)='integer' AND estimate>=0),
 maximum INTEGER NOT NULL CHECK(typeof(maximum)='integer' AND maximum>0), provider_maximum INTEGER NOT NULL CHECK(typeof(provider_maximum)='integer' AND provider_maximum>0),
 state TEXT NOT NULL DEFAULT 'quoted' CHECK(state IN ('quoted','reserved','submitted','settled','technical_waiver','reconciliation_pending','cancelled')),
 customer_status TEXT NOT NULL DEFAULT 'none' CHECK(customer_status IN ('none','held','settled','waived')),
 provider_status TEXT NOT NULL DEFAULT 'none' CHECK(provider_status IN ('none','held','settled','pessimistic')),
 held_included INTEGER NOT NULL DEFAULT 0 CHECK(typeof(held_included)='integer' AND held_included>=0),
 held_purchased INTEGER NOT NULL DEFAULT 0 CHECK(typeof(held_purchased)='integer' AND held_purchased>=0),
 charged_included INTEGER NOT NULL DEFAULT 0 CHECK(typeof(charged_included)='integer' AND charged_included>=0),
 charged_purchased INTEGER NOT NULL DEFAULT 0 CHECK(typeof(charged_purchased)='integer' AND charged_purchased>=0),
 provider_cost INTEGER NOT NULL DEFAULT 0 CHECK(typeof(provider_cost)='integer' AND provider_cost>=0),
 dispatch_token TEXT, provider_id TEXT, provider_request_id TEXT, usage_json TEXT, result_json TEXT, failure_reason TEXT,
 created_at TEXT NOT NULL, expires_at TEXT NOT NULL, submitted_at TEXT, customer_deadline TEXT, settled_at TEXT,
 UNIQUE(owner_id,mode,idempotency_key), UNIQUE(mode,provider_id), FOREIGN KEY(owner_id,mode) REFERENCES eidos_pg_ai_accounts(owner_id,mode),
 CHECK(charged_included+charged_purchased<=maximum), CHECK(charged_included<=held_included AND charged_purchased<=held_purchased)
);
CREATE INDEX IF NOT EXISTS pg_ai_actions_owner_period ON eidos_pg_ai_actions(owner_id,mode,period,customer_status);
CREATE INDEX IF NOT EXISTS pg_ai_actions_provider ON eidos_pg_ai_actions(mode,provider_status);
-- Frozen complete Responses payload, counted with the provider's exact structural tokenizer.
-- Separate additive table preserves historical action rows and old fixture contracts.
CREATE TABLE IF NOT EXISTS eidos_pg_ai_preflights (
 action_id TEXT PRIMARY KEY REFERENCES eidos_pg_ai_actions(id), request_hash TEXT NOT NULL,
 request_json TEXT NOT NULL, input_tokens INTEGER NOT NULL CHECK(typeof(input_tokens)='integer' AND input_tokens>=0),
 count_request_id TEXT NOT NULL, created_at TEXT NOT NULL
);
-- Every zero-fee counting attempt is claimed before outbound I/O and retained.
-- Quotas are shared by the same persistent ledger across workers and models.
CREATE TABLE IF NOT EXISTS eidos_pg_ai_count_claims (
 action_id TEXT PRIMARY KEY REFERENCES eidos_pg_ai_actions(id), owner_id TEXT NOT NULL,
 mode TEXT NOT NULL CHECK(mode IN ('test','live')), request_hash TEXT NOT NULL, request_json TEXT NOT NULL,
 client_request_id TEXT NOT NULL UNIQUE, provider_request_id TEXT, input_tokens INTEGER,
 state TEXT NOT NULL CHECK(state IN ('submitted','counted','failed')), failure_reason TEXT,
 created_at TEXT NOT NULL, completed_at TEXT,
 CHECK(input_tokens IS NULL OR (typeof(input_tokens)='integer' AND input_tokens>=0 AND input_tokens<=10000000))
);
CREATE INDEX IF NOT EXISTS pg_ai_count_owner_created ON eidos_pg_ai_count_claims(owner_id,mode,created_at);
CREATE TABLE IF NOT EXISTS eidos_pg_ai_journal (
 id TEXT PRIMARY KEY, owner_id TEXT, mode TEXT NOT NULL CHECK(mode IN ('test','live')), action_id TEXT,
 event TEXT NOT NULL, detail_json TEXT NOT NULL, created_at TEXT NOT NULL
);
-- Append-only diagnostic expenses. A pricing bound remains pessimistic until a
-- separate settled receipt exists. Original claim/result journal rows survive.
CREATE TABLE IF NOT EXISTS eidos_pg_ai_diagnostic_costs (
 claim_id TEXT NOT NULL REFERENCES eidos_pg_ai_journal(id), mode TEXT NOT NULL CHECK(mode='test'),
 status TEXT NOT NULL CHECK(status IN ('bounded','settled')),
 provider_cost INTEGER NOT NULL CHECK(typeof(provider_cost)='integer' AND provider_cost>=0 AND provider_cost<=1000000000000),
 evidence_json TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(claim_id,status)
);
CREATE VIEW IF NOT EXISTS eidos_pg_ai_balances AS
 SELECT a.owner_id,a.mode,a.requested_limit,a.frozen,
 COALESCE((SELECT SUM(g.amount-g.revoked_amount) FROM eidos_pg_ai_grants g WHERE g.owner_id=a.owner_id AND g.mode=a.mode AND g.kind='included'),0)-COALESCE((SELECT SUM(CASE WHEN x.customer_status='held' THEN x.held_included WHEN x.customer_status='settled' THEN x.charged_included ELSE 0 END) FROM eidos_pg_ai_actions x WHERE x.owner_id=a.owner_id AND x.mode=a.mode),0) AS available_included,
 COALESCE((SELECT SUM(g.amount-g.revoked_amount) FROM eidos_pg_ai_grants g WHERE g.owner_id=a.owner_id AND g.mode=a.mode AND g.kind='purchased'),0)-COALESCE((SELECT SUM(CASE WHEN x.customer_status='held' THEN x.held_purchased WHEN x.customer_status='settled' THEN x.charged_purchased ELSE 0 END) FROM eidos_pg_ai_actions x WHERE x.owner_id=a.owner_id AND x.mode=a.mode),0) AS available_purchased
 FROM eidos_pg_ai_accounts a;
