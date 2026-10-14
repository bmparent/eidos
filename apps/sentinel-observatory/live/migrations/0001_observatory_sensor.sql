CREATE TABLE IF NOT EXISTS observatory_sensor_events (
  id TEXT PRIMARY KEY NOT NULL,
  observed_at TEXT NOT NULL,
  source_id TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('request', 'collector', 'policy_event')),
  event_json TEXT NOT NULL CHECK (json_valid(event_json))
);
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS observatory_sensor_events_observed_at
  ON observatory_sensor_events (observed_at DESC, id DESC);
