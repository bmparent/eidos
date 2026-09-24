-- Isolated Eidos Works database. Additive; no platform/customer tables are changed.
CREATE TABLE IF NOT EXISTS growth_events (
  id TEXT PRIMARY KEY, day TEXT NOT NULL, session_hash TEXT NOT NULL,
  event TEXT NOT NULL CHECK(event IN ('page_view','session_start','landing_page','friction_cta_click','friction_form_start','friction_submit','contact_submit','select_project','lab_open')),
  traffic TEXT NOT NULL CHECK(traffic IN ('public','production_qa','preview_qa','automation','bot')),
  path TEXT NOT NULL, landing TEXT NOT NULL, source TEXT NOT NULL, medium TEXT NOT NULL,
  campaign TEXT NOT NULL, content TEXT NOT NULL, referral TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS growth_day ON growth_events(day,traffic);
CREATE INDEX IF NOT EXISTS growth_session ON growth_events(session_hash,day);
CREATE TABLE IF NOT EXISTS growth_quotas (bucket TEXT NOT NULL, period INTEGER NOT NULL, used INTEGER NOT NULL, expires INTEGER NOT NULL, PRIMARY KEY(bucket,period));
