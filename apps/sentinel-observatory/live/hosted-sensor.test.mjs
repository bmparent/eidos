import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {DatabaseSync} from 'node:sqlite';
import {capability, snapshot, recordObservation} from './hosted-sensor.mjs';
import {parseSecurityLog} from './security-events.mjs';

// SQLite executes the real migration and retention queries, while this wrapper
// provides D1's prepare/bind/all/batch API. Fixtures are exclusively test evidence.
class D1SQLite {
  constructor({migrate = true} = {}) {
    this.sqlite = new DatabaseSync(':memory:');
    if (migrate) this.sqlite.exec(readFileSync(new URL('./migrations/0001_observatory_sensor.sql', import.meta.url), 'utf8'));
  }
  prepare(sql) {
    const statement = this.sqlite.prepare(sql);
    return {bind: (...args) => ({run: () => statement.run(...args), all: async () => ({results: statement.all(...args), success: true})})};
  }
  async batch(statements) {
    this.sqlite.exec('BEGIN');
    try {const rows = statements.map(statement => ({...statement.run(), success: true})); this.sqlite.exec('COMMIT'); return rows;}
    catch (error) {this.sqlite.exec('ROLLBACK'); throw error;}
  }
  close() {this.sqlite.close();}
}
const incoming = path => new Request(`https://observatory.example${path ?? '/api/observe'}`, {method: 'POST', headers: {Authorization: 'Bearer test-secret', Cookie: 'secret=omit', 'User-Agent': 'private-person'}});
const reply = (body, status = 200) => new Response(JSON.stringify(body), {status, headers: {'Content-Type': 'application/json'}});
const fixtureReceipt = (extra = {}) => ({event: {id: 'evt_' + 'a'.repeat(32), url: 'https://1.1.1.1/private-test-path', completed_at: new Date().toISOString(),
  latency_ms: 123, status: 301, headers: {authorization: 'secret response'}, coverage: {head_attempted: true, transport: {kind: 'authenticated_pinned_ip_relay'},
    network: {scope: 'observatory_outbound_head_connection', destination_ip: '1.1.1.1', destination_port: 443, protocol: 'TCP', application_protocol: 'HTTPS',
      peer_ip: '1.1.1.1', peer_port: 443, peer_matches_destination: true, socket_connected: true, tls_verified: true, tls_protocol: 'TLSv1.3', bytes_written: 80, bytes_read: 280}}, ...extra}});

test('native source is disconnected unless the exact D1 storage binding exists', async () => {
  assert.equal(capability({}).connected, false);
  assert.equal(capability({}).auto_start, false);
  const missing = await snapshot({});
  assert.equal(missing.connected, false);
  assert.equal(missing.text, '');
});

test('actual incoming and outbound response metadata persists without secrets or target paths', async () => {
  const db = new D1SQLite(), env = {OBSERVATORY_DB: db};
  try {
    const response = reply(fixtureReceipt());
    const saved = await recordObservation(env, incoming(), response, {request_duration_ms: 160});
    assert.equal(saved.recorded, 2);
    assert.ok((await response.json()).event.id.startsWith('evt_')); // Original response remains usable.
    const current = await snapshot(env);
    assert.equal(current.connected, true);
    assert.equal(current.auto_start, true);
    assert.equal(current.mode, 'observatory_edge');
    assert.equal(current.counts.records, 2);
    assert.doesNotMatch(current.text, /test-secret|private-person|secret response|private-test-path|Authorization|Cookie|User-Agent/);
    const rows = parseSecurityLog(current.text, {sourceId: current.source_id, sourceName: current.source_name, importedAt: current.observed_at, mode: current.mode});
    assert.equal(rows.events.length, 2);
    assert.equal(rows.summary.requests, 1);
    assert.equal(rows.summary.collector_events, 1);
    const collector = rows.events.find(row => row.kind === 'collector');
    assert.equal(collector.fields.dest_endpoint, '1.1.1.1');
    assert.equal(collector.fields.bytes_written, 80);
    assert.equal(collector.fields.bytes_read, 280);
    assert.equal(collector.fields.tls_verified, true);
    assert.equal(collector.actual_security_event, false);
    assert.equal(rows.coverage.state, 'observatory_edge');
    assert.ok(rows.coverage.missing.some(value => /LAN/.test(value)));
  } finally {db.close();}
});

test('explicit real policy enforcement is separate from the triggering request and tagged functional checks', async () => {
  const db = new D1SQLite(), env = {OBSERVATORY_DB: db};
  try {
    await recordObservation(env, incoming(), reply({error: {code: 'non_public_target', message: 'private target rejected'}}, 400), {test_context: 'functional_check'});
    const current = await snapshot(env);
    const rows = parseSecurityLog(current.text, {sourceId: current.source_id, importedAt: current.observed_at});
    assert.equal(rows.events.length, 2);
    const policy = rows.events.find(row => row.kind === 'policy_event');
    assert.equal(policy.actual_security_event, true);
    assert.equal(policy.attack_confirmed, false);
    assert.equal(policy.classification, 'observed_policy_enforcement');
    assert.equal(policy.fields.rule_action, 'deny');
    assert.match(policy.summary, /functional check/);
    assert.equal(rows.events.filter(row => row.actual_security_event).length, 1);
  } finally {db.close();}
});

test('TLS failures and HTTP errors stay operational, with unknown socket counters as null', async () => {
  const db = new D1SQLite(), env = {OBSERVATORY_DB: db};
  try {
    const failed = fixtureReceipt({status: null, error: {code: 'tls_error'}, coverage: {head_attempted: true, transport: {kind: 'authenticated_pinned_ip_relay'},
      network: {scope: 'observatory_outbound_head_connection', destination_ip: '1.1.1.1', destination_port: 443, protocol: 'TCP',
        application_protocol: 'HTTPS', peer_ip: '10.0.0.1', tls_verified: false, socket_connected: false, bytes_written: null, bytes_read: null}}});
    await recordObservation(env, incoming(), reply(failed));
    const current = await snapshot(env), rows = parseSecurityLog(current.text, {sourceId: current.source_id, importedAt: current.observed_at});
    const collector = rows.events.find(row => row.kind === 'collector');
    assert.equal(collector.fields.outcome, 'tls_validation_failed');
    assert.equal(collector.fields.tls_verified, false);
    assert.equal(collector.fields.network.peer_ip, null);
    assert.equal(collector.fields.bytes_written, null);
    assert.equal(collector.fields.bytes_read, null);
    assert.equal(collector.actual_security_event, false);
    await recordObservation(env, incoming(), reply({error: {code: 'invalid_json'}}, 400));
    const after = await snapshot(env);
    assert.equal(parseSecurityLog(after.text, {sourceId: after.source_id, importedAt: after.observed_at}).summary.policy_events, 0);
  } finally {db.close();}
});

test('feed polling, capabilities and static assets never manufacture observations', async () => {
  const db = new D1SQLite(), env = {OBSERVATORY_DB: db};
  try {
    for (const path of ['/api/security-events', '/api/capabilities', '/app.js', '/']) {
      assert.equal((await recordObservation(env, incoming(path), reply({}))).status, 'ignored');
    }
    assert.equal((await snapshot(env)).counts.records, 0);
  } finally {db.close();}
});

test('real SQLite retention keeps latest 1000 records and deletes records older than seven days', async () => {
  const db = new D1SQLite(), env = {OBSERVATORY_DB: db};
  try {
    const insert = db.sqlite.prepare('INSERT INTO observatory_sensor_events VALUES (?, ?, ?, ?, ?)');
    const old = new Date(Date.now() - 8 * 86400000).toISOString();
    insert.run('old-fixture', old, 'observatory-edge', 'request', '{}');
    for (let i = 0; i < 1005; i++) insert.run(`fixture-${i}`, new Date(Date.now() - (1005 - i) * 1000).toISOString(), 'observatory-edge', 'request', '{}');
    await recordObservation(env, incoming(), reply({}));
    assert.equal(db.sqlite.prepare('SELECT count(*) AS n FROM observatory_sensor_events').get().n, 1000);
    assert.equal(db.sqlite.prepare("SELECT count(*) AS n FROM observatory_sensor_events WHERE id='old-fixture'").get().n, 0);
    assert.equal((await snapshot(env)).counts.records, 1000);
  } finally {db.close();}
});

test('missing migrations fail visibly rather than returning a false zero-event verdict', async () => {
  const db = new D1SQLite({migrate: false}), env = {OBSERVATORY_DB: db};
  try {
    assert.equal((await recordObservation(env, incoming(), reply({}))).status, 'unavailable');
    const current = await snapshot(env);
    assert.equal(current.connected, false);
    assert.equal(current.coverage.state, 'unavailable');
    assert.ok(current.coverage.missing.some(value => /no security verdict/.test(value)));
  } finally {db.close();}
});

test('a sensor read purges expired rows without creating new observations', async () => {
  const db = new D1SQLite(), env = {OBSERVATORY_DB: db};
  try {
    db.sqlite.prepare('INSERT INTO observatory_sensor_events VALUES (?, ?, ?, ?, ?)')
      .run('expired-fixture', new Date(Date.now() - 8 * 86400000).toISOString(), 'observatory-edge', 'request', '{}');
    assert.equal((await snapshot(env)).counts.records, 0);
    assert.equal(db.sqlite.prepare('SELECT count(*) AS n FROM observatory_sensor_events').get().n, 0);
  } finally {db.close();}
});

test('an oversized response cannot be copied into sensor storage', async () => {
  const db = new D1SQLite(), env = {OBSERVATORY_DB: db};
  try {
    const saved = await recordObservation(env, incoming(), reply({padding: 'do-not-store'.repeat(7000)}));
    assert.equal(saved.recorded, 1);
    const current = await snapshot(env);
    assert.doesNotMatch(current.text, /do-not-store/);
    assert.equal(current.counts.records, 1);
  } finally {db.close();}
});

test('an incomplete response stream cannot stall sensor ingestion beyond its evidence deadline', async () => {
  const db = new D1SQLite(), env = {OBSERVATORY_DB: db};
  const stream = new ReadableStream({start(controller) {controller.enqueue(new TextEncoder().encode('{"event":'));}});
  const response = new Response(stream), started = performance.now();
  try {
    const result = await recordObservation(env, incoming(), response);
    assert.equal(result.status, 'unavailable');
    assert.ok(performance.now() - started < 3500);
    assert.equal((await snapshot(env)).counts.records, 0);
  } finally {await response.body.cancel(); db.close();}
});
