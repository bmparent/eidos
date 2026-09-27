import { createHash, randomUUID } from "node:crypto";
import { createClient, type Client } from "@libsql/client";

const NOW = "CAST(unixepoch('subsec') * 1000 AS INTEGER)";
// Longer than the dispatch route's 300-second maximum lifetime. An allocator
// must atomically leave RESERVED before calling the provider.
export const RESERVATION_MS = 360_000;
export const ALLOCATION_GRACE_MS = 360_000;
export type Admission = { jobId: string; lockDigest: string; phase: string; expiresAt: number; createdAt: number; owner: string; sourceCommit: string };

export function admissionConfigured(env: Record<string, string | undefined> = process.env) {
  try {
    const url = new URL(env.EIDOS_DATABASE_URL || "");
    return ["libsql:", "https:"].includes(url.protocol) && !url.username && !url.password && Boolean(env.EIDOS_DATABASE_AUTH_TOKEN);
  } catch { return false; }
}

export function validateRetryKey(value: string | null) {
  if (!value || !/^[a-zA-Z0-9_-]{16,128}$/.test(value)) throw new Error("IDEMPOTENCY_KEY_REQUIRED");
  return value;
}

function record(row: Record<string, unknown>): Admission {
  return { jobId: String(row.job_id), lockDigest: String(row.lock_digest), phase: String(row.phase), expiresAt: Number(row.expires_at), createdAt: Number(row.created_at), owner: String(row.owner), sourceCommit: String(row.source_commit) };
}

async function writeWithRetry<T>(operation: () => Promise<T>): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try { return await operation(); }
    catch (error) {
      // Only explicit lock contention is safe to retry here. Network errors
      // may have committed and must retain their existing retry identity.
      if (attempt >= 5 || !/SQLITE_BUSY|SQLITE_LOCKED/.test(String(error))) throw error;
      await new Promise(resolve => setTimeout(resolve, 20 * 2 ** attempt));
    }
  }
}

export class AdmissionStore {
  constructor(readonly client: Client, readonly scope: string) {}

  async initialize() {
    // A separate namespace: no Eidos Works tables or records are modified.
    await writeWithRetry(() => this.client.execute(`CREATE TABLE IF NOT EXISTS sentinel_lab_admissions (
      scope TEXT NOT NULL, retry_hash TEXT NOT NULL, job_id TEXT NOT NULL,
      lock_digest TEXT NOT NULL, source_commit TEXT NOT NULL, owner TEXT NOT NULL,
      phase TEXT NOT NULL, expires_at INTEGER NOT NULL, created_at INTEGER NOT NULL,
      PRIMARY KEY(scope, retry_hash), UNIQUE(scope, job_id))`));
  }

  async reserve(key: string, digest: string, commit: string, maximum: number): Promise<{ admission: Admission; acquired: boolean }> {
    validateRetryKey(key);
    await this.initialize();
    const retryHash = createHash("sha256").update(key).digest("hex");
    const jobId = `rd-${digest.slice(0, 12)}-${createHash("sha256").update(this.scope + ":" + retryHash).digest("hex").slice(0, 8)}`;
    const owner = randomUUID();
    // All statements execute on the primary in ONE write transaction. The
    // count and insertion cannot interleave with another function instance.
    const results = await writeWithRetry(() => this.client.batch([
      { sql: `UPDATE sentinel_lab_admissions SET phase='ABANDONED' WHERE scope=? AND phase='RESERVED' AND expires_at <= ${NOW}`, args: [this.scope] },
      { sql: `INSERT INTO sentinel_lab_admissions
        (scope,retry_hash,job_id,lock_digest,source_commit,owner,phase,expires_at,created_at)
        SELECT ?,?,?,?,?,?,'RESERVED',${NOW}+?,${NOW}
        WHERE (SELECT COUNT(*) FROM sentinel_lab_admissions WHERE scope=? AND phase IN ('RESERVED','ALLOCATING','RUNNING')) < ?
        ON CONFLICT(scope,retry_hash) DO NOTHING`, args: [this.scope, retryHash, jobId, digest, commit, owner, RESERVATION_MS, this.scope, maximum] },
      { sql: "SELECT * FROM sentinel_lab_admissions WHERE scope=? AND retry_hash=?", args: [this.scope, retryHash] },
    ], "write"));
    const row = results[2].rows[0];
    if (!row) throw new Error("RUNNER_CAPACITY_OCCUPIED");
    const admission = record(row);
    if (admission.lockDigest !== digest) throw new Error("IDEMPOTENCY_LOCK_CONFLICT");
    return { admission, acquired: admission.owner === owner };
  }

  async beginAllocation(admission: Admission, timeout: number) {
    const result = await this.client.execute({
      sql: `UPDATE sentinel_lab_admissions SET phase='ALLOCATING', expires_at=${NOW}+?
        WHERE scope=? AND job_id=? AND owner=? AND phase='RESERVED' AND expires_at>${NOW}`,
      args: [timeout + ALLOCATION_GRACE_MS, this.scope, admission.jobId, admission.owner],
    });
    if (result.rowsAffected !== 1) throw new Error("ADMISSION_RESERVATION_EXPIRED");
  }

  async running(jobId: string) {
    await this.client.execute({ sql: "UPDATE sentinel_lab_admissions SET phase='RUNNING' WHERE scope=? AND job_id=? AND phase='ALLOCATING'", args: [this.scope, jobId] });
  }

  async release(jobId: string) {
    await this.client.execute({ sql: `UPDATE sentinel_lab_admissions SET phase='TERMINAL', expires_at=MIN(expires_at,${NOW}) WHERE scope=? AND job_id=?`, args: [this.scope, jobId] });
  }

  async active() {
    await this.initialize();
    const result = await this.client.execute({ sql: "SELECT * FROM sentinel_lab_admissions WHERE scope=? AND phase IN ('ALLOCATING','RUNNING')", args: [this.scope] });
    return result.rows.map(record);
  }

  async expired(jobId: string) {
    const result = await this.client.execute({ sql: `SELECT job_id FROM sentinel_lab_admissions WHERE scope=? AND job_id=? AND expires_at<=${NOW}`, args: [this.scope, jobId] });
    return result.rows.length === 1;
  }

  async lookup(jobId: string) {
    await this.initialize();
    const result = await this.client.execute({ sql: "SELECT * FROM sentinel_lab_admissions WHERE scope=? AND job_id=?", args: [this.scope, jobId] });
    return result.rows[0] ? record(result.rows[0]) : null;
  }
}

let store: AdmissionStore | undefined;
export function sharedAdmission() {
  if (!admissionConfigured()) throw new Error("ADMISSION_DATABASE_NOT_CONFIGURED");
  return store ??= new AdmissionStore(createClient({ url: process.env.EIDOS_DATABASE_URL!, authToken: process.env.EIDOS_DATABASE_AUTH_TOKEN }),
    `${process.env.VERCEL_PROJECT_ID || "eidos-sentinel-lab"}:sandbox`);
}
