/** Actual Observatory application-edge observations; no packet sniffing or WAF.
 * Persist only response-side allowlisted metadata. The protected Sites dispatch
 * owns access control. No request body, auth header, cookie or incoming IP is read.
 */
import {normalizeTarget} from './collector.mjs';

const SENSOR_MAX_RECORDS = 1000;
const SENSOR_RETENTION_MS = 7 * 24 * 60 * 60 * 1000;
const SENSOR_MAX_RESPONSE_BYTES = 65536;
const sensorRules = Object.freeze({same_origin_required: 'deny', non_public_target: 'deny', recursive_target: 'deny', credentials_not_allowed: 'deny',
  unsupported_protocol: 'deny', unsupported_port: 'deny', payload_too_large: 'deny', sample_rate_limited: 'rate_limit'});
const sensorMissing = Object.freeze(['LAN and internal-network traffic', 'Packet capture', 'Other sites and remote target visitor traffic',
  'Provider firewall or IDS events', 'Authentication denials before protected Sites dispatch']);
const sensorPrivacy = 'Shows a seven-day window of this Observatory API request outcomes and requested public HEAD metadata, capped at 1000 records. Older rows are purged on the next sensor read or write; no unattended cleanup is claimed. No client IP, cookies, auth headers, user agent, query strings, request bodies, target paths or payloads are collected.';
const sensorNumber = (value, max = 86400000) => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= max ? value : null;
const sensorStatus = value => Number.isInteger(value) && value >= 100 && value <= 599 ? value : null;
const sensorTime = value => typeof value === 'string' && /(?:Z|[+-]\d\d:\d\d)$/.test(value) && Number.isFinite(Date.parse(value)) ? new Date(value).toISOString() : null;
const sensorObject = value => value !== null && typeof value === 'object' && !Array.isArray(value) ? value : {};
const sensorDatabase = env => env?.OBSERVATORY_DB && typeof env.OBSERVATORY_DB.prepare === 'function' && typeof env.OBSERVATORY_DB.batch === 'function' ? env.OBSERVATORY_DB : null;
const sensorId = () => `edge_${crypto.randomUUID().replaceAll('-', '')}`;
const sensorInteger = value => Number.isSafeInteger(value) && value >= 0 ? value : null;
async function sensorBounded(work, timeoutMs = 2500) {
  let timer;
  try {return await Promise.race([work, new Promise((_, reject) => {timer = setTimeout(() => reject(new Error('Sensor evidence deadline')), timeoutMs);})]);}
  finally {clearTimeout(timer);}
}
function sensorHost(value) {
  try {return normalizeTarget(value).hostname;} catch {return null;}
}
function sensorPublicIp(value) {
  if (typeof value !== 'string' || !/^[0-9a-f:.]+$/i.test(value)) return null;
  try {const result = normalizeTarget(value); return result.kind === 'hostname' ? null : result.hostname;} catch {return null;}
}

export function capability(env = {}) {
  return {connected: Boolean(sensorDatabase(env)), mode: 'observatory_edge', auto_start: Boolean(sensorDatabase(env)),
    source_id: 'observatory-edge', source_name: 'Observatory application-edge sensor', format: 'auto', interval_ms: 5000,
    max_records: SENSOR_MAX_RECORDS, retention_days: 7, starts_capture: false, privacy: sensorPrivacy,
    reason: sensorDatabase(env) ? 'Attached to actual Observatory API traffic and requested outbound HEAD checks' : 'Observatory sensor storage is not attached',
    coverage: {scope: 'This protected Observatory service only', missing: [...sensorMissing]}};
}

async function sensorReadResponse(response, deadline = performance.now() + 2500) {
  const clone = response.clone();
  const declared = clone.headers.get('content-length');
  if (declared !== null && (!/^\d+$/.test(declared) || Number(declared) > SENSOR_MAX_RESPONSE_BYTES)) return {};
  if (!clone.body) return {};
  const reader = clone.body.getReader(), chunks = [];
  let size = 0;
  try {
    while (true) {
      const budget = deadline - performance.now();
      if (budget <= 0) throw new Error('Sensor response deadline');
      const {done, value} = await sensorBounded(reader.read(), budget);
      if (done) break;
      size += value.byteLength;
      if (size > SENSOR_MAX_RESPONSE_BYTES) {void reader.cancel().catch(() => {}); return {};}
      chunks.push(value);
    }
    const all = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {all.set(chunk, offset); offset += chunk.length;}
    return sensorObject(JSON.parse(new TextDecoder().decode(all)));
  } catch {void reader.cancel().catch(() => {}); return {};}
  finally {reader.releaseLock();}
}

function sensorOutcome(error, status) {
  const code = typeof error?.code === 'string' ? error.code : '';
  if (code === 'tls_error') return 'tls_validation_failed';
  if (code === 'timeout') return 'timeout';
  if (code.startsWith('dns_')) return 'dns_failure';
  if (['connection_error', 'transport_error', 'collector_transport_error'].includes(code)) return 'connection_failed';
  if (code) return 'upstream_error';
  return status !== null ? 'ok' : 'upstream_error';
}

function sensorNetwork(raw) {
  const value = sensorObject(raw);
  if (value.scope !== 'observatory_outbound_head_connection') return null;
  const destination = sensorPublicIp(value.destination_ip);
  if (!destination) return null;
  return {scope: 'observatory_outbound_head_connection', destination_ip: destination,
    destination_port: [80, 443].includes(value.destination_port) ? value.destination_port : null,
    protocol: 'TCP', application_protocol: ['HTTP', 'HTTPS'].includes(value.application_protocol) ? value.application_protocol : null,
    peer_ip: sensorPublicIp(value.peer_ip), peer_port: [80, 443].includes(value.peer_port) ? value.peer_port : null,
    peer_matches_destination: typeof value.peer_matches_destination === 'boolean' ? value.peer_matches_destination : null,
    socket_connected: typeof value.socket_connected === 'boolean' ? value.socket_connected : null,
    tls_verified: typeof value.tls_verified === 'boolean' ? value.tls_verified : null,
    tls_protocol: ['TLSv1.2', 'TLSv1.3', 'other'].includes(value.tls_protocol) ? value.tls_protocol : null,
    bytes_written: sensorInteger(value.bytes_written), bytes_read: sensorInteger(value.bytes_read)};
}

export async function recordObservation(env, request, response, ctx = {}) {
  const db = sensorDatabase(env);
  if (!db) return {recorded: 0, status: 'disconnected'};
  // Watching/polling this feed and fetching static assets cannot generate events.
  let path;
  try {path = new URL(request.url).pathname;} catch {return {recorded: 0, status: 'ignored'};}
  if (path !== '/api/observe') return {recorded: 0, status: 'ignored'};
  try {
    const deadline = performance.now() + 2500;
    const now = new Date().toISOString(), payload = await sensorReadResponse(response, deadline);
    const testContext = ctx?.test_context === 'functional_check' ? 'functional_check' : undefined;
    const code = typeof payload.error?.code === 'string' && Object.hasOwn(sensorRules, payload.error.code) ? payload.error.code : null;
    const incoming = {sensor: 'observatory_edge', timestamp: now, event_id: sensorId(), event_type: 'request', route: 'observe',
      method: ['GET', 'POST', 'HEAD'].includes(request.method) ? request.method : 'other', status: sensorStatus(response.status),
      duration_ms: sensorNumber(ctx?.request_duration_ms), timestamp_scope: 'response_completed_at',
      src_endpoint: 'Your Observatory session', dest_endpoint: 'Sentinel edge', outcome: code ? 'blocked' : response.status < 400 ? 'ok' : 'invalid_request',
      test_context: testContext};
    const rows = [incoming];
    if (code) rows.push({...incoming, event_id: sensorId(), event_type: 'policy_event', rule_id: code, rule_action: sensorRules[code]});
    const event = sensorObject(payload.event);
    if (typeof event.id === 'string' && /^evt_[a-f0-9]{32}$/.test(event.id)) {
      const destination = sensorHost(event.url ?? payload.target?.url);
      if (destination) {
        const network = sensorNetwork(event.coverage?.network);
        const transport = event.coverage?.transport?.kind;
        rows.push({sensor: 'observatory_edge', timestamp: sensorTime(event.completed_at) ?? now, event_id: sensorId(), event_type: 'collector', route: 'observe',
          method: 'HEAD', status: sensorStatus(event.status), duration_ms: sensorNumber(event.latency_ms),
          src_endpoint: 'Observatory observer', dest_endpoint: destination, outcome: sensorOutcome(event.error, sensorStatus(event.status)),
          head_attempted: event.coverage?.head_attempted === true, linked_evidence_id: event.id,
          transport: ['authenticated_pinned_ip_relay', 'named_service_adapter'].includes(transport) ? transport : 'cloudflare_public_head',
          network, test_context: testContext});
      }
    }
    const statements = rows.map(row => db.prepare('INSERT OR IGNORE INTO observatory_sensor_events (id, observed_at, source_id, kind, event_json) VALUES (?, ?, ?, ?, ?)')
      .bind(row.event_id, row.timestamp, 'observatory-edge', row.event_type, JSON.stringify(row)));
    const cutoff = new Date(Date.parse(now) - SENSOR_RETENTION_MS).toISOString();
    statements.push(db.prepare('DELETE FROM observatory_sensor_events WHERE observed_at < ? OR id IN (SELECT id FROM observatory_sensor_events ORDER BY observed_at DESC, id DESC LIMIT -1 OFFSET ?)')
      .bind(cutoff, SENSOR_MAX_RECORDS));
    const budget = deadline - performance.now();
    if (budget <= 0) return {recorded: 0, status: 'unavailable'};
    const saved = await sensorBounded(db.batch(statements), budget);
    if (Array.isArray(saved) && saved.some(row => row?.success === false)) return {recorded: 0, status: 'unavailable'};
    return {recorded: rows.length, status: 'stored'};
  } catch {return {recorded: 0, status: 'unavailable'};}
}

export async function snapshot(env = {}) {
  const caps = capability(env), observedAt = new Date().toISOString(), db = sensorDatabase(env);
  if (!db) return {...caps, observed_at: observedAt, text: '', counts: {records: 0}, coverage: {state: 'disconnected', missing: [...sensorMissing, 'No sensor storage binding']}};
  try {
    const deadline = performance.now() + 2500;
    const cutoff = new Date(Date.parse(observedAt) - SENSOR_RETENTION_MS).toISOString();
    // Reads also purge expired rows; polling itself does not insert observations.
    await sensorBounded(db.prepare('DELETE FROM observatory_sensor_events WHERE observed_at < ?').bind(cutoff).run(), Math.max(1, deadline - performance.now()));
    const result = await sensorBounded(db.prepare('SELECT event_json FROM observatory_sensor_events WHERE observed_at >= ? ORDER BY observed_at DESC, id DESC LIMIT ?')
      .bind(cutoff, SENSOR_MAX_RECORDS).all(), Math.max(1, deadline - performance.now()));
    if (result.success === false) throw new Error('Sensor query unavailable');
    const records = Array.isArray(result.results) ? result.results.map(row => row.event_json).filter(value => typeof value === 'string' && value.length <= 8192).reverse() : [];
    return {...caps, observed_at: observedAt, text: records.join('\n'), counts: {records: records.length},
      coverage: {state: 'observatory_edge', scope: 'Only /api/observe requests reaching this service and requested outbound HEAD checks',
        missing: [...sensorMissing], retention_days: 7, retention_cleanup: 'Older records purged on sensor read/write; no unattended cleanup',
        max_records: SENSOR_MAX_RECORDS, newest_event_at: records.length ? sensorTime(JSON.parse(records.at(-1)).timestamp) : null}};
  } catch {return {...caps, connected: false, auto_start: false, observed_at: observedAt, text: '', counts: {records: 0},
    reason: 'Observatory sensor storage is unavailable', coverage: {state: 'unavailable', missing: [...sensorMissing, 'Current sensor evidence unavailable; no security verdict']}};}
}
