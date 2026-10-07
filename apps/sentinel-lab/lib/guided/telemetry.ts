import { randomBytes, randomUUID } from "node:crypto";
import { guidedStore, hash, canonical, LabError, LIMITS, type Document, type GuidedStore } from "./store";

export async function createMonitor(owner: string, body: Document, store = guidedStore()) {
  if (typeof body.name !== "string" || !body.name.trim() || body.name.length > 80 ||
    typeof body.target !== "string" || !/^[\w.-]{1,80}$/.test(body.target) || typeof body.unit !== "string" || !body.unit || body.unit.length > 40)
    throw new LabError(400, "Name the source, measurement and unit.");
  if ((await store.list(owner, "monitor")).length >= 4) throw new LabError(429, "Maximum four monitors in this pilot workspace.");
  const id = "source-" + randomUUID();
  const monitor = { id, name: body.name.trim(), target: body.target, unit: body.unit, staleSeconds: 300,
    createdAt: new Date().toISOString(), processedOffset: 0, checkpoint: null, resetId: "initial",
    autoProcess: body.autoProcess === true, synthetic: body.synthetic === true, lastArrival: null,
    policy: "At-least-once ingress; exact deduplication; arrival-order offsets; late event time retained without model update." };
  await store.put(owner, "monitor", id, monitor);
  return { monitor, ...await issueKey(owner, id, store) };
}

export async function issueKey(owner: string, source: string, store = guidedStore()) {
  await store.get(owner, "monitor", source);
  const key = "eidos_ingest_" + randomBytes(32).toString("base64url");
  const digest = hash(key);
  await store.client.execute({ sql: "INSERT INTO sentinel_guided_keys (scope,digest,owner,source,expires) VALUES (?,?,?,?,?)",
    args: [store.scope, digest, owner, source, Date.now() + 30 * 86400000] });
  return { key, keyId: digest.slice(0, 16), expiresAt: new Date(Date.now() + 30 * 86400000).toISOString(),
    note: "Shown once. Store in the collector secret environment. Revoke from this source's settings." };
}

export async function revokeKeys(owner: string, source: string, store = guidedStore()) {
  await store.get(owner, "monitor", source);
  await store.client.execute({ sql: "UPDATE sentinel_guided_keys SET revoked=1 WHERE scope=? AND owner=? AND source=?", args: [store.scope, owner, source] });
}

function otlpValue(value: any): unknown {
  if (!value || typeof value !== "object") return undefined;
  return value.doubleValue ?? value.intValue ?? value.stringValue ?? value.boolValue;
}
export function otlpEvents(payload: Document, target: string) {
  const events: Document[] = [];
  for (const resource of payload.resourceMetrics || []) for (const scope of resource.scopeMetrics || []) for (const metric of scope.metrics || []) {
    if (metric.name !== target) continue;
    for (const point of metric.gauge?.dataPoints || metric.sum?.dataPoints || []) {
      const time = Number(BigInt(point.timeUnixNano) / 1000000n);
      const attrs = Object.fromEntries((point.attributes || []).map((a: any) => [a.key, otlpValue(a.value)]));
      events.push({ eventTime: new Date(time).toISOString(), value: Number(point.asDouble ?? point.asInt),
        attributes: attrs, sourceIdentity: Object.fromEntries((resource.resource?.attributes || []).map((a: any) => [a.key, otlpValue(a.value)])),
        id: hash(canonical({ resource: resource.resource, name: metric.name, point })) });
    }
  }
  return events;
}

export async function ingest(owner: string, source: string, payload: Document, store = guidedStore()) {
  const monitor = await store.get(owner, "monitor", source);
  const events = payload.resourceMetrics ? otlpEvents(payload, monitor.target) : payload.events;
  if (!Array.isArray(events) || !events.length || events.length > LIMITS.telemetryBatch)
    throw new LabError(400, "Send 1–100 events or OTLP JSON numeric gauge/sum points matching the configured measurement.");
  const arrival = Date.now();
  const dayStart = Math.floor(arrival / 86400000) * 86400000;
  for (const event of events) {
    if (!event || typeof event.id !== "string" || !/^[\w:.-]{1,128}$/.test(event.id) || typeof event.eventTime !== "string" ||
      !/^\d{4}-\d\d-\d\dT.*(?:Z|[+-]\d\d:\d\d)$/.test(event.eventTime) || !Number.isFinite(Date.parse(event.eventTime)) ||
      Date.parse(event.eventTime) > arrival + 300000 || typeof event.value !== "number" || !Number.isFinite(event.value) ||
      Buffer.byteLength(JSON.stringify(event)) > 4096)
      throw new LabError(400, "Each event needs a stable ID, finite numeric value and offset-bearing ISO event time; maximum 4 KB per event.");
  }
  // One transaction assigns offsets without gaps. ON CONFLICT does not reserve
  // an offset, so reconnect/replay cannot increment accepted counts twice.
  const tx = await store.client.transaction("write");
  let accepted = 0, duplicates = 0;
  try {
    const fresh = await tx.execute({ sql: "SELECT body FROM sentinel_guided_objects WHERE scope=? AND owner=? AND kind='monitor' AND id=?", args: [store.scope, owner, source] });
    const current = JSON.parse(String(fresh.rows[0].body));
    const count = await tx.execute({ sql: "SELECT COUNT(*) AS count,COALESCE(MAX(offset),0) AS last FROM sentinel_guided_events WHERE scope=? AND source=?", args: [store.scope, source] });
    let offset = Number(count.rows[0].last);
    const daily = await tx.execute({ sql: "SELECT COUNT(*) AS count FROM sentinel_guided_events WHERE scope=? AND source=? AND arrival>=?", args: [store.scope, source, dayStart] });
    let today = Number(daily.rows[0].count);
    for (const event of events) {
      const duplicate = await tx.execute({ sql: "SELECT body FROM sentinel_guided_events WHERE scope=? AND source=? AND dedup=?", args: [store.scope, source, event.id] });
      if (duplicate.rows.length) {
        const previous = JSON.parse(String(duplicate.rows[0].body));
        if (canonical(previous.original) !== canonical(event)) throw new LabError(409, "A deduplication ID was reused with different event content.");
        duplicates++; continue;
      }
      if (offset - current.processedOffset >= LIMITS.telemetryBuffer || today >= LIMITS.telemetryDaily)
        throw new LabError(429, "Backpressure: buffer or daily event limit reached. Keep this batch in the collector queue and retry after processing.");
      const record = { ...event, eventTime: new Date(event.eventTime).toISOString(), original: event, sourceIdentity: source,
        offset: ++offset, arrivalTime: new Date(arrival).toISOString(), synthetic: current.synthetic };
      await tx.execute({ sql: "INSERT INTO sentinel_guided_events VALUES (?,?,?,?,?,?,?,?)", args: [store.scope, owner, source, offset, event.id, record.eventTime, arrival, JSON.stringify(record)] });
      accepted++; today++;
    }
    current.lastArrival = new Date(arrival).toISOString();
    current.lastOffset = offset;
    await tx.execute({ sql: "UPDATE sentinel_guided_objects SET body=?,updated=? WHERE scope=? AND owner=? AND kind='monitor' AND id=?", args: [JSON.stringify(current), arrival, store.scope, owner, source] });
    await tx.commit();
    return { accepted, duplicates, lastOffset: offset, buffered: offset - current.processedOffset, autoProcess: current.autoProcess,
      policy: monitor.policy, synthetic: monitor.synthetic };
  } catch (error) { await tx.rollback(); throw error; } finally { tx.close(); }
}

export async function monitorStatus(owner: string, id: string, store = guidedStore()) {
  const monitor = await store.get(owner, "monitor", id);
  const events = await store.client.execute({ sql: "SELECT offset,body FROM sentinel_guided_events WHERE scope=? AND owner=? AND source=? ORDER BY offset DESC LIMIT 100", args: [store.scope, owner, id] });
  return { ...monitor, checkpoint: monitor.checkpoint ? { ...monitor.checkpoint, model: undefined } : null,
    events: events.rows.map(row => JSON.parse(String(row.body))),
    health: !monitor.lastArrival ? "awaiting first event" : Date.now() - Date.parse(monitor.lastArrival) > monitor.staleSeconds * 1000 ? "stale" : "receiving",
    buffered: (monitor.lastOffset || 0) - monitor.processedOffset, warmupRemaining: Math.max(0, 24 - (monitor.checkpoint?.accepted || 0)) };
}

export async function processMonitor(owner: string, id: string, store = guidedStore()) {
  const monitor = await store.get(owner, "monitor", id);
  const events = await store.client.execute({ sql: "SELECT body FROM sentinel_guided_events WHERE scope=? AND owner=? AND source=? AND offset>? ORDER BY offset LIMIT 1000", args: [store.scope, owner, id, monitor.processedOffset] });
  if (!events.rows.length) throw new LabError(409, "No unprocessed events. The saved checkpoint is current.");
  const records = events.rows.map(row => JSON.parse(String(row.body)));
  const retry = `monitor_${id}_${monitor.resetId}_${monitor.processedOffset}`;
  const existing = await store.client.execute({ sql: "SELECT id FROM sentinel_guided_jobs WHERE scope=? AND owner=? AND retry_hash=?", args: [store.scope, owner, hash(retry)] });
  if (existing.rows[0]) {
    const { startJob, pollJob } = await import("./jobs");
    const previousJob = await pollJob(owner, String(existing.rows[0].id), store);
    if (previousJob.status === "completed") {
      const current = await store.get(owner, "monitor", id);
      if (current.processedOffset > monitor.processedOffset && current.processedOffset < (current.lastOffset || 0))
        return processMonitor(owner, id, store);
    }
    return previousJob.status === "queued" ? startJob(owner, previousJob.id, store) : previousJob;
  }
  // Bound automatic compute to four admitted batches/day/source. The collector
  // retains buffered data when this budget is exhausted; no paid cron is added.
  const jobs = await store.client.execute({ sql: "SELECT COUNT(*) AS count FROM sentinel_guided_jobs WHERE scope=? AND owner=? AND created>=? AND json_extract(request,'$.monitorId')=?", args: [store.scope, owner, Date.now() - 86400000, id] });
  if (Number(jobs.rows[0].count) >= 4) throw new LabError(429, "This monitor reached its four-batch daily compute budget. Ingestion remains bounded and buffered.");
  const job = await store.enqueue(owner, retry, { operation: "telemetry", monitorId: id, events: records, checkpoint: monitor.checkpoint,
    resetId: monitor.resetId, target: monitor.target, unit: monitor.unit, staleSeconds: monitor.staleSeconds });
  const { startJob } = await import("./jobs");
  return startJob(owner, job.id, store);
}

export async function commitTelemetry(owner: string, job: Document, result: Document, store: GuidedStore) {
  const tx = await store.client.transaction("write");
  try {
    const row = await tx.execute({ sql: "SELECT body FROM sentinel_guided_objects WHERE scope=? AND owner=? AND kind='monitor' AND id=?", args: [store.scope, owner, job.request.monitorId] });
    const monitor = JSON.parse(String(row.rows[0].body));
    const previous = job.request.checkpoint?.processedOffset || 0;
    if (monitor.processedOffset === result.processedOffset) { await tx.commit(); return; }
    if (monitor.processedOffset !== previous || monitor.resetId !== job.request.resetId) throw new LabError(409, "Checkpoint changed while this batch ran. Its result cannot overwrite newer state.");
    monitor.checkpoint = result.checkpoint; monitor.processedOffset = result.processedOffset;
    monitor.lastOutcome = { ...result, checkpoint: undefined };
    await tx.execute({ sql: "UPDATE sentinel_guided_objects SET body=?,updated=? WHERE scope=? AND owner=? AND kind='monitor' AND id=?", args: [JSON.stringify(monitor), Date.now(), store.scope, owner, monitor.id] });
    await tx.commit();
  } catch (error) { await tx.rollback(); throw error; } finally { tx.close(); }
}
