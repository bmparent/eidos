import test from "node:test";
import assert from "node:assert/strict";
import { createClient } from "@libsql/client";
import { GuidedStore, canonical, hash } from "../lib/guided/store";
import { principal, requireSameOrigin } from "../lib/guided/auth";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { summarizePatterns } from "../lib/guided/patterns";
import { publicAddress, validateDestination } from "../lib/guided/fetch-source";
import { createMonitor, ingest, commitTelemetry, revokeKeys, otlpEvents } from "../lib/guided/telemetry";

const makeStore = () => new GuidedStore(createClient({ url: "file::memory:" }), "test-guided");

test("pattern descriptions exclude missing pairs and preserve original record references", () => {
  const result = summarizePatterns({ records: [{ x: 1, y: 2 }, { x: 2, y: 4 }, { x: 3, y: 6 }, { x: null, y: 100 }], recordIds: ['r1','r2','r3','r4'] }, { features: ['x','y'] });
  assert.equal(result.pairs[0].completePairs, 3); assert.equal(result.pairs[0].correlation, 1);
  assert.deepEqual(result.pairs[0].sourceRecords, ['r1','r2','r3']);
});

test("existing Eidos member credentials remain owner-bound and immediately revocable", async () => {
  const store = makeStore(); await store.initialize();
  await store.client.batch([
    "CREATE TABLE eidos_email_members(id TEXT PRIMARY KEY,disabled INTEGER)",
    "CREATE TABLE eidos_member_keys(member_id TEXT,key_hash TEXT,revoked INTEGER)",
    "CREATE TABLE eidos_member_sessions(member_id TEXT,token_hash TEXT,expires INTEGER)",
    "INSERT INTO eidos_email_members VALUES ('member-a',0)",
  ]);
  const token = 'ew_agent_' + 'b'.repeat(64);
  await store.client.execute({ sql: "INSERT INTO eidos_member_keys VALUES (?,?,0)", args: ['member-a', hash(token)] });
  const request = new Request('https://lab.example/api', { headers: { Authorization: `Bearer ${token}` } });
  assert.equal((await principal(request, store)).id, 'member:member-a');
  await store.client.execute('UPDATE eidos_member_keys SET revoked=1');
  await assert.rejects(principal(request, store), /Sign in/); store.client.close();
});

test("independent database clients share one retry identity and dispatch lease", async () => {
  const directory = mkdtempSync(join(tmpdir(), "eidos-guided-test-"));
  const stores = Array.from({ length: 6 }, () => new GuidedStore(createClient({ url: `file:${join(directory, "test.db").replaceAll("\\", "/")}` }), "distributed-test"));
  try {
    for (const store of stores) await store.initialize();
    const jobs = await Promise.all(stores.map(store => store.enqueue("alice", "stable_distributed_intent", { operation: "parse", bytes: 17 })));
    assert.equal(new Set(jobs.map(job => job.id)).size, 1);
    const claims = await Promise.all(stores.map(store => store.claim("alice", jobs[0].id)));
    assert.equal(claims.filter(Boolean).length, 1);
    // Reopening a client reconstructs the existing job after a lost response.
    const reopened = new GuidedStore(createClient({ url: `file:${join(directory, "test.db").replaceAll("\\", "/")}` }), "distributed-test");
    assert.equal((await reopened.enqueue("alice", "stable_distributed_intent", { operation: "parse", bytes: 17 })).id, jobs[0].id);
    reopened.client.close();
  } finally { stores.forEach(store => store.client.close()); }
});

test("browser Host normalization permits same-origin writes and rejects cross-origin writes", () => {
  requireSameOrigin(new Request("http://localhost:3210/api", { headers: { Host: "127.0.0.1:3210", Origin: "http://127.0.0.1:3210" } }));
  assert.throws(() => requireSameOrigin(new Request("https://lab.example/api", { headers: { Host: "lab.example", Origin: "https://evil.example" } })), /Cross-origin/);
  assert.throws(() => requireSameOrigin(new Request("https://lab.example/api", { headers: { "Sec-Fetch-Site": "cross-site" } })), /Cross-site/);
});

test("owner isolation covers datasets, runs, embeddings, sources, feedback and monitors", async () => {
  const store = makeStore();
  for (const kind of ["dataset", "source", "run", "feedback", "monitor"]) {
    await store.put("alice", kind, `${kind}-a`, { id: `${kind}-a`, vectors: [[1, 2]], secret: "private-record" });
    await assert.rejects(store.get("bob", kind, `${kind}-a`), /not found/);
    assert.deepEqual(await store.list("bob", kind), []);
  }
  const job = await store.enqueue("alice", "abcdefghijklmnop", { operation: "parse", content: "private" });
  await assert.rejects(store.job("bob", job.id), /not found/);
  store.client.close();
});

test("retry keys preserve job identity, settings and cancellation fencing", async () => {
  const store = makeStore();
  const results = await Promise.all(Array.from({ length: 8 }, () => store.enqueue("alice", "abcdefghijklmnop", { operation: "analyze", a: 1 })));
  assert.equal(new Set(results.map(r => r.id)).size, 1);
  await assert.rejects(store.enqueue("alice", "abcdefghijklmnop", { operation: "analyze", a: 2 }), /different settings/);
  const claims = await Promise.all(Array.from({ length: 8 }, () => store.claim("alice", results[0].id)));
  assert.equal(claims.filter(Boolean).length, 1);
  await store.updateJob("alice", results[0].id, "cancelled");
  await store.updateJob("alice", results[0].id, "completed", { result: { fake: true } });
  assert.equal((await store.job("alice", results[0].id)).status, "cancelled");
  store.client.close();
});

test("canonical input identity ignores object key order but preserves values", () => {
  assert.equal(canonical({ a: 1, b: [2] }), canonical({ b: [2], a: 1 }));
  assert.deepEqual(JSON.parse(canonical({ a: 1, b: [2] })), { a: 1, b: [2] });
  assert.notEqual(hash(canonical({ a: 1 })), hash(canonical({ a: 2 })));
});

test("public fetch rejects private, metadata, special-use and redirect targets", async () => {
  for (const ip of ["127.0.0.1", "10.0.0.1", "169.254.169.254", "172.16.0.1", "192.168.1.1", "100.64.0.1", "::1", "::ffff:127.0.0.1", "fe80::1", "2001:db8::1"]) assert.equal(publicAddress(ip), false, ip);
  assert.equal(publicAddress("8.8.8.8"), true);
  const dualStack = (async () => [{ address: "2606:4700:4700::1111", family: 6 }, { address: "8.8.8.8", family: 4 }]) as any;
  assert.deepEqual((await validateDestination("https://example.com/data", dualStack)).address, { address: "8.8.8.8", family: 4 });
  await assert.rejects(validateDestination("https://example.com/data", (async () => [{ address: "::1", family: 6 }]) as any));
  await assert.rejects(validateDestination("https://example.com/data", (async () => [{ address: "8.8.8.8", family: 4 }, { address: "127.0.0.1", family: 4 }]) as any));
  const privateResolver = (async () => [{ address: "10.0.0.1", family: 4 }]) as any;
  for (const url of ["http://example.com/data", "https://user:pass@example.com", "https://example.com:444/a", "https://example.com/data?token=secret", "https://example.com"]) {
    await assert.rejects(validateDestination(url, privateResolver));
  }
});

test("scoped credential cannot read data, switch sources, or survive revocation", async () => {
  const store = makeStore();
  const { monitor, key } = await createMonitor("alice", { name: "latency", target: "latency", unit: "ms" }, store);
  const req = new Request("https://lab.example/api", { headers: { Authorization: `Bearer ${key}` } });
  const user = await principal(req, store, true);
  assert.equal(user.id, "alice"); assert.equal(user.source, monitor.id);
  await assert.rejects(principal(req, store, false), /ingest only/);
  await revokeKeys("alice", monitor.id, store);
  await assert.rejects(principal(req, store, true), /expired or revoked/);
  store.client.close();
});

test("telemetry deduplication, conflict rollback, late data and checkpoint compare-and-set", async () => {
  const store = makeStore();
  const { monitor } = await createMonitor("alice", { name: "latency", target: "latency", unit: "ms", synthetic: true }, store);
  const events = [0, 1, 2].map(i => ({ id: `event-${i}`, value: 40 + i, eventTime: `2026-08-01T00:0${i}:00Z` }));
  const first = await ingest("alice", monitor.id, { events }, store);
  const replay = await ingest("alice", monitor.id, { events }, store);
  assert.equal(first.accepted, 3); assert.equal(first.lastOffset, 3); assert.equal(replay.duplicates, 3); assert.equal(replay.lastOffset, 3);
  await assert.rejects(ingest("alice", monitor.id, { events: [{ ...events[0], value: 99 }] }, store), /different event content/);
  const job = { request: { monitorId: monitor.id, checkpoint: null, resetId: "initial" } };
  const result = { processedOffset: 3, checkpoint: { processedOffset: 3 }, outcomes: [] };
  await commitTelemetry("alice", job, result, store);
  await commitTelemetry("alice", job, result, store);
  assert.equal((await store.get("alice", "monitor", monitor.id)).processedOffset, 3);
  await assert.rejects(commitTelemetry("alice", job, { ...result, processedOffset: 4 }, store), /changed while/);
  store.client.close();
});

test("OTLP JSON preserves event time and target metric selection", () => {
  const events = otlpEvents({ resourceMetrics: [{ resource: { attributes: [{ key: "service.name", value: { stringValue: "api" } }] }, scopeMetrics: [{ metrics: [
    { name: "latency", gauge: { dataPoints: [{ timeUnixNano: "1785542400000000000", asDouble: 42 }] } },
    { name: "unrelated", gauge: { dataPoints: [{ timeUnixNano: "1785542400000000000", asDouble: 100 }] } },
  ] }] }] }, "latency");
  assert.equal(events.length, 1); assert.equal(events[0].value, 42); assert.equal(events[0].sourceIdentity["service.name"], "api");
  assert.equal(events[0].id.length, 64);
});
