import { createClient, type Client, type InStatement } from "@libsql/client";
import { createHash, randomUUID } from "node:crypto";

export type Document = Record<string, any>;
export class LabError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}
export const hash = (value: string | Buffer) => createHash("sha256").update(value).digest("hex");
export const canonical = (value: unknown): string => {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") return "{" + Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => `${JSON.stringify(k)}:${canonical(v)}`).join(",") + "}";
  return JSON.stringify(value) ?? "null";
};
export const LIMITS = { uploadBytes: 2_000_000, rows: 5000, columns: 64, features: 16, entities: 8,
  passages: 256, datasetsPerOwner: 30, jobsPerOwner: 120, retentionDays: 30, concurrentJobs: 1,
  telemetryBatch: 100, telemetryBuffer: 1000, telemetryDaily: 10000, jobSeconds: 900, llmCalls: 0 };

export class GuidedStore {
  constructor(readonly client: Client, readonly scope: string) {}
  private ready?: Promise<void>;
  initialize() {
    return this.ready ??= this.client.batch([
      `CREATE TABLE IF NOT EXISTS sentinel_guided_objects (scope TEXT NOT NULL, owner TEXT NOT NULL, kind TEXT NOT NULL,
        id TEXT NOT NULL, body TEXT NOT NULL, created INTEGER NOT NULL, updated INTEGER NOT NULL, expires INTEGER NOT NULL,
        PRIMARY KEY(scope,kind,id))`,
      `CREATE INDEX IF NOT EXISTS sentinel_guided_owner ON sentinel_guided_objects(scope,owner,kind,updated)`,
      `CREATE TABLE IF NOT EXISTS sentinel_guided_jobs (scope TEXT NOT NULL, owner TEXT NOT NULL, id TEXT NOT NULL,
        retry_hash TEXT NOT NULL, request_hash TEXT NOT NULL, request TEXT NOT NULL, status TEXT NOT NULL, provider TEXT,
        result TEXT, error TEXT, lease TEXT, created INTEGER NOT NULL, updated INTEGER NOT NULL, expires INTEGER NOT NULL,
        PRIMARY KEY(scope,id), UNIQUE(scope,owner,retry_hash))`,
      `CREATE TABLE IF NOT EXISTS sentinel_guided_keys (scope TEXT NOT NULL, digest TEXT NOT NULL, owner TEXT NOT NULL,
        source TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0, expires INTEGER NOT NULL, PRIMARY KEY(scope,digest))`,
      `CREATE TABLE IF NOT EXISTS sentinel_guided_events (scope TEXT NOT NULL, owner TEXT NOT NULL, source TEXT NOT NULL,
        offset INTEGER NOT NULL, dedup TEXT NOT NULL, event_time TEXT NOT NULL, arrival INTEGER NOT NULL, body TEXT NOT NULL,
        PRIMARY KEY(scope,source,dedup), UNIQUE(scope,source,offset))`,
    ], "write").then(() => undefined).catch(error => { this.ready = undefined; throw error; });
  }
  async get(owner: string, kind: string, id: string): Promise<Document> {
    await this.initialize();
    const result = await this.client.execute({ sql: "SELECT body,expires FROM sentinel_guided_objects WHERE scope=? AND owner=? AND kind=? AND id=?",
      args: [this.scope, owner, kind, id] });
    if (!result.rows[0]) throw new LabError(404, "This item was not found in your workspace.");
    if (Number(result.rows[0].expires) < Date.now()) throw new LabError(410, "This item has expired. Its retained archive has not been deleted.");
    return JSON.parse(String(result.rows[0].body));
  }
  async list(owner: string, kind: string) {
    await this.initialize();
    const rows = await this.client.execute({ sql: "SELECT id,body,created,expires FROM sentinel_guided_objects WHERE scope=? AND owner=? AND kind=? ORDER BY updated DESC LIMIT 120", args: [this.scope, owner, kind] });
    return rows.rows.map(row => ({ ...JSON.parse(String(row.body)), expired: Number(row.expires) < Date.now() }));
  }
  async put(owner: string, kind: string, id: string, body: Document) {
    await this.initialize();
    const now = Date.now();
    const encoded = JSON.stringify(body);
    if (Buffer.byteLength(encoded) > 16_000_000) throw new LabError(413, "Result exceeds the durable storage limit.");
    await this.client.execute({ sql: `INSERT INTO sentinel_guided_objects VALUES (?,?,?,?,?,?,?,?)
      ON CONFLICT(scope,kind,id) DO UPDATE SET body=excluded.body,updated=excluded.updated WHERE sentinel_guided_objects.owner=excluded.owner`,
      args: [this.scope, owner, kind, id, encoded, now, now, now + LIMITS.retentionDays * 86400000] });
  }
  async job(owner: string, id: string): Promise<Document> {
    await this.initialize();
    const result = await this.client.execute({ sql: "SELECT * FROM sentinel_guided_jobs WHERE scope=? AND owner=? AND id=?", args: [this.scope, owner, id] });
    const row = result.rows[0];
    if (!row) throw new LabError(404, "This job was not found in your workspace.");
    return { ...row, request: JSON.parse(String(row.request)), provider: row.provider ? JSON.parse(String(row.provider)) : null,
      result: row.result ? JSON.parse(String(row.result)) : null };
  }
  async jobs(owner: string) {
    await this.initialize();
    const rows = await this.client.execute({ sql: "SELECT id,status,created,updated,error FROM sentinel_guided_jobs WHERE scope=? AND owner=? ORDER BY created DESC LIMIT 120", args: [this.scope, owner] });
    return rows.rows;
  }
  async enqueue(owner: string, retry: string, request: Document) {
    await this.initialize();
    if (!/^[a-zA-Z0-9_-]{16,128}$/.test(retry)) throw new LabError(400, "A stable Idempotency-Key is required.");
    const requestHash = hash(canonical(request));
    const id = "gj-" + hash(`${this.scope}:${owner}:${retry}`).slice(0, 32);
    const now = Date.now();
    const rows = await this.client.batch([
      { sql: `INSERT INTO sentinel_guided_jobs (scope,owner,id,retry_hash,request_hash,request,status,created,updated,expires)
        SELECT ?,?,?,?,?,?,'queued',?,?,? WHERE (SELECT COUNT(*) FROM sentinel_guided_jobs WHERE scope=? AND owner=?) < ?
        ON CONFLICT(scope,owner,retry_hash) DO NOTHING`, args: [this.scope, owner, id, hash(retry), requestHash, JSON.stringify(request), now, now, now + 30 * 86400000, this.scope, owner, LIMITS.jobsPerOwner] },
      { sql: "SELECT id,request_hash FROM sentinel_guided_jobs WHERE scope=? AND owner=? AND retry_hash=?", args: [this.scope, owner, hash(retry)] },
    ], "write");
    if (!rows[1].rows[0]) throw new LabError(429, "This workspace reached its 120-job pilot envelope. Export results and contact the operator.");
    if (String(rows[1].rows[0].request_hash) !== requestHash) throw new LabError(409, "This retry key belongs to different settings.");
    return this.job(owner, String(rows[1].rows[0].id));
  }
  async claim(owner: string, id: string) {
    const lease = randomUUID();
    const rows = await this.client.execute({ sql: "UPDATE sentinel_guided_jobs SET status='preparing',lease=?,updated=? WHERE scope=? AND owner=? AND id=? AND status='queued'", args: [lease, Date.now(), this.scope, owner, id] });
    return rows.rowsAffected === 1 ? lease : null;
  }
  async updateJob(owner: string, id: string, status: string, values: { provider?: Document; result?: Document; error?: string } = {}, lease?: string) {
    const setters = ["status=?", "updated=?"];
    const args: any[] = [status, Date.now()];
    for (const key of ["provider", "result", "error"] as const) if (key in values) {
      setters.push(`${key}=?`); args.push(key === "error" ? values[key] : JSON.stringify(values[key]));
    }
    args.push(this.scope, owner, id);
    if (lease) args.push(lease);
    await this.client.execute({ sql: `UPDATE sentinel_guided_jobs SET ${setters.join(",")} WHERE scope=? AND owner=? AND id=?
      AND status NOT IN ('cancelled','expired','completed')${lease ? " AND lease=?" : ""}`, args });
  }
}

let singleton: GuidedStore | undefined;
export function guidedStore() {
  if (singleton) return singleton;
  const local = process.env.EIDOS_GUIDED_LOCAL === "1" && !process.env.VERCEL;
  const url = local ? process.env.EIDOS_GUIDED_DATABASE_URL : process.env.EIDOS_DATABASE_URL;
  if (!url || (!local && (!/^(libsql|https):/.test(url) || !process.env.EIDOS_DATABASE_AUTH_TOKEN)))
    throw new LabError(503, "Durable Lab storage is not configured for this environment.");
  return singleton = new GuidedStore(createClient({ url, authToken: local ? undefined : process.env.EIDOS_DATABASE_AUTH_TOKEN }),
    process.env.EIDOS_GUIDED_SCOPE || `${process.env.VERCEL_ENV || "local"}:guided-v1`);
}
