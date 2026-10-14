CREATE TABLE IF NOT EXISTS observatory_target_states (
  target_id TEXT PRIMARY KEY NOT NULL,
  revision INTEGER NOT NULL DEFAULT 0,
  profile TEXT NOT NULL,
  last_event_id TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  state_json TEXT NOT NULL CHECK (json_valid(state_json))
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS observatory_target_records (
  id TEXT PRIMARY KEY NOT NULL,
  target_id TEXT NOT NULL,
  observed_at TEXT NOT NULL,
  event_json TEXT NOT NULL CHECK (json_valid(event_json))
);
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS observatory_target_records_target_time
  ON observatory_target_records (target_id, observed_at DESC, id DESC);
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS observatory_target_records_time
  ON observatory_target_records (observed_at DESC, id DESC);
