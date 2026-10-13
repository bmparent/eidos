import test from 'node:test';
import assert from 'node:assert/strict';
import {parseSecurityLog, normalizeSecurityRecord, analyzeSecurityEvents, SECURITY_LIMITS} from './security-events.mjs';

const importedAt = '2026-10-13T20:00:00Z';
const options = {sourceId: 'test-fixture-only', importedAt};
const stamp = index => new Date(Date.parse('2026-10-13T19:00:00Z') + index * 1000).toISOString();
const flow = (index, bytes = 100, extra = {}) => ({timestamp: stamp(index), event_type: 'flow', flow_id: index,
  src_ip: '10.0.0.2', dest_ip: '203.0.113.2', src_port: 5000, dest_port: 443, proto: 'TCP', flow: {bytes_toserver: bytes, bytes_toclient: 0, pkts_toserver: 2, age: 1}, ...extra});
const parse = (rows, opts = {}) => parseSecurityLog(rows.map(x => JSON.stringify(x)).join('\n'), {...options, ...opts});

test('sensor rule evidence has original timestamps and bounded safe metadata', () => {
  const alert = {timestamp: '2026-10-13T19:00:00.123456+0000', event_type: 'alert', flow_id: '18446744073709551610',
    src_ip: '10.0.0.4', dest_ip: '203.0.113.8', alert: {signature_id: 2100001, signature: 'TEST RULE', severity: 1, action: 'allowed'},
    payload: 'secret', http: {url: '/?token=secret', authorization: 'secret'}, tls: {certificate: 'secret'}};
  const result = parse([alert]);
  assert.equal(result.events[0].actual_security_event, true);
  assert.equal(result.events[0].attack_confirmed, false);
  assert.equal(result.events[0].observed_at, '2026-10-13T19:00:00.123Z');
  assert.equal(result.events[0].source_timestamp, '2026-10-13T19:00:00.123456+0000');
  assert.equal(result.events[0].fields.flow_id, '18446744073709551610');
  assert.equal(result.events[0].fields.src_ip, '10.0.0.4');
  assert.equal(result.events[0].evidence.id, result.events[0].id);
  assert.doesNotMatch(JSON.stringify(result.events), /secret|authorization|certificate|payload/);
});

test('JSON numeric 64-bit flow identifier remains lossless', () => {
  const row = '{"timestamp":"2026-10-13T19:00:00Z","event_type":"alert","flow_id":18446744073709551610,"alert":{"signature_id":1,"signature":"TEST"}}';
  assert.equal(parseSecurityLog(row, options).events[0].source_event_id, '18446744073709551610');
});

test('distinct microsecond sensor rows do not collide after browser timestamp normalization', () => {
  const alert = {timestamp: '2026-10-13T19:00:00.123001+0000', event_type: 'alert', flow_id: 1, alert: {signature_id: 1, signature: 'TEST'}};
  const rows = parse([alert, {...alert, timestamp: '2026-10-13T19:00:00.123002+0000'}]);
  assert.equal(rows.events.length, 2);
  assert.notEqual(rows.events[0].id, rows.events[1].id);
});

test('Zeek connection, notice, DNS and TLS preserve source time and exclude notice messages', () => {
  const shared = {ts: Date.parse(stamp(0)) / 1000, uid: 'TEST-UID', 'id.orig_h': '10.0.0.3', 'id.resp_h': '203.0.113.9', 'id.resp_p': 443, proto: 'tcp'};
  const result = parse([{...shared, orig_bytes: 10, resp_bytes: 30, duration: 1.1, conn_state: 'SF'},
    {...shared, note: 'Scan::Port_Scan', msg: 'private user secret', actions: ['Notice::ACTION_LOG']},
    {...shared, query: 'example.com', rcode_name: 'NXDOMAIN'}, {...shared, version: 'TLSv13'}]);
  assert.deepEqual(result.events.map(x => x.kind), ['connection', 'notice', 'dns', 'tls']);
  assert.equal(result.events[0].fields.bytes, 40);
  assert.equal(result.events[1].severity, 'unknown');
  assert.equal(result.events[1].actual_security_event, true);
  assert.doesNotMatch(JSON.stringify(result.events), /private user secret/);
});

test('duplicates and partial/malformed source records are excluded with visible gaps', () => {
  const row = JSON.stringify(flow(0));
  const result = parseSecurityLog(`${row}\n${row}\n{broken}\n{"event_type":"alert","timestamp":"2026-10-13T19:00:00Z"}`, options);
  assert.equal(result.events.length, 1);
  assert.equal(result.duplicate_count, 1);
  assert.equal(result.rejected_count, 2);
  assert.ok(result.coverage.missing.some(x => /excluded/.test(x)));
  assert.ok(result.coverage.missing.some(x => /not proof/.test(x)));
});

test('invalid timezone, future source times and unrelated HTTP logs are rejected', () => {
  assert.equal(parse([flow(0, 100, {timestamp: '2026-10-13T19:00:00'})]).rejected_count, 1);
  assert.equal(parse([flow(0, 100, {timestamp: '2026-10-14T19:00:00Z'})]).rejected_count, 1);
  assert.equal(parse([{status: 500, latency_ms: 9000, timestamp: stamp(0)}]).events.length, 0);
  assert.throws(() => normalizeSecurityRecord({ts: Date.parse(stamp(0)) / 1000, uid: 'HTTP', uri: '/login'}, options), /Unsupported Zeek record/);
});

test('bounded imports stop beyond row caps and reject oversized input', () => {
  const result = parse([flow(0), flow(1), flow(2)], {maxRecords: 2});
  assert.equal(result.events.length, 2);
  assert.equal(result.rejected_count, 1);
  assert.throws(() => parseSecurityLog('x'.repeat(SECURITY_LIMITS.maxBytes + 1), options), /exceeds 2 MB/);
});

test('network baseline has 12 warmup records and any deviation remains experimental', () => {
  const events = parse(Array.from({length: 16}, (_, i) => flow(i, i < 12 ? 100 : 90000000))).events;
  const result = analyzeSecurityEvents(events);
  assert.equal(result.observations.length, 16);
  assert.equal(result.observations.filter(x => x.state === 'calibrating').length, 12);
  assert.ok(result.anomalies.length > 0);
  assert.ok(result.observations.every(x => !x.actual_security_event && x.confidence === 'experimental'));
  assert.ok(result.observations.every(x => x.evidence_ids.length === 1 && events.some(e => e.id === x.evidence_ids[0])));
});

test('destination fanout is per origin and explicitly heuristic', () => {
  const oneOrigin = parse(Array.from({length: 21}, (_, i) => flow(i, 100, {dest_ip: `203.0.113.${i + 1}`, dest_port: i + 1}))).events;
  const result = analyzeSecurityEvents(oneOrigin);
  const fanout = result.anomalies.find(x => x.metric === 'logged_destination_fanout');
  assert.equal(fanout.value, 21);
  assert.equal(fanout.confidence, 'heuristic');
  assert.equal(fanout.actual_security_event, false);
  const manyOrigins = oneOrigin.map((row, i) => ({...row, fields: {...row.fields, src_ip: `10.0.0.${i + 1}`}}));
  assert.equal(analyzeSecurityEvents(manyOrigins).observations.filter(x => x.metric === 'logged_destination_fanout').length, 0);
});

test('alert-only exports do not invent network features, empty windows or forecasts', () => {
  const batch = parse([{timestamp: stamp(0), event_type: 'alert', alert: {signature_id: 1, signature: 'TEST'}}]);
  const analyzed = analyzeSecurityEvents(batch.events);
  assert.deepEqual(analyzed.observations, []);
  assert.equal(analyzed.coverage.state, 'insufficient_data');
  assert.equal(analyzed.summary.actual_security_events, 1);
});
