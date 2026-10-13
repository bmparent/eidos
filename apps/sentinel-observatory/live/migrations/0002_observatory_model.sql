CREATE TABLE IF NOT EXISTS observatory_model_jobs (
  id TEXT PRIMARY KEY NOT NULL,
  evidence_id TEXT NOT NULL,
  model TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  day TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('reserved', 'completed', 'rejected', 'failed')),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  reserved_micro_usd INTEGER NOT NULL CHECK (reserved_micro_usd >= 0),
  input_tokens INTEGER,
  output_tokens INTEGER,
  prompt_sha256 TEXT NOT NULL,
  result_json TEXT CHECK (result_json IS NULL OR json_valid(result_json)),
  error_code TEXT,
  UNIQUE (evidence_id, model, prompt_version)
);
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS observatory_model_jobs_daily
  ON observatory_model_jobs (day, created_at);
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS observatory_sensor_events_linked_evidence
  ON observatory_sensor_events (json_extract(event_json, '$.linked_evidence_id'), observed_at DESC);
