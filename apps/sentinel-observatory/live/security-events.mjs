/** Bounded, dependency-free adapters for sensor evidence. No packet capture.
 * Sensor alerts are source-reported rule matches, not verified attacks. Sentinel's
 * numerical port is unchanged and its network-feature interpretation is experimental.
 */
import {SentinelPort, ENGINE_PROVENANCE} from './brain.mjs';

export const SECURITY_LIMITS = Object.freeze({maxBytes: 2 * 1024 * 1024, maxRecords: 2000, maxLineBytes: 65536});
export const SECURITY_PRIVACY = 'Only allowlisted sensor metadata is retained in browser memory. Addresses and DNS names may identify your network. Payloads, HTTP headers, URLs, certificates and Zeek notice messages are omitted. Nothing is sent to a model automatically.';
const obj = x => x !== null && typeof x === 'object' && !Array.isArray(x) ? x : {};
const clean = (x, max = 240) => typeof x === 'string' ? x.replace(/[\u0000-\u001f\u007f]/g, '').slice(0, max) : null;
const number = (x, max = 1e15) => typeof x === 'number' && Number.isFinite(x) && x >= 0 && x <= max ? x : null;
const port = x => Number.isInteger(x) && x >= 0 && x <= 65535 ? x : null;
const identifier = x => typeof x === 'string' ? clean(x, 160) : Number.isSafeInteger(x) && x >= 0 ? String(x) : null;
const compact = x => Object.fromEntries(Object.entries(x).filter(([, v]) => v !== null && v !== undefined));
const time = value => {
  if (typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 8640000000000) return new Date(value * 1000).toISOString();
  if (typeof value !== 'string') return null;
  const normalized = value.replace(/([+-]\d{2})(\d{2})$/, '$1:$2');
  if (!/(Z|[+-]\d\d:\d\d)$/i.test(normalized) || !Number.isFinite(Date.parse(normalized))) return null;
  return new Date(normalized).toISOString();
};
function hash(text) {
  let a = 0x811c9dc5, b = 0x9e3779b9;
  for (let i = 0; i < text.length; i++) { a = Math.imul(a ^ text.charCodeAt(i), 16777619); b = Math.imul(b ^ text.charCodeAt(i), 2246822519); }
  return `${(a >>> 0).toString(16).padStart(8, '0')}${(b >>> 0).toString(16).padStart(8, '0')}`;
}
const count = (items, key) => items.reduce((out, e) => {out[e[key]] = (out[e[key]] || 0) + 1; return out;}, {});
const sumKnown = values => values.some(x => x !== null) ? values.reduce((s, x) => s + (x ?? 0), 0) : null;

/** Returns allowlisted source metadata; raw packet content is deliberately omitted. */
export function normalizeSecurityRecord(raw, {sourceId = 'local-sensor', format = 'auto', sourceName = 'Sensor log', importedAt = new Date().toISOString()} = {}) {
  const record = obj(raw);
  const source = format === 'suricata' || format === 'suricata_eve' || (format === 'auto' && typeof record.event_type === 'string') ? 'suricata'
    : format === 'zeek' || format === 'zeek_json' || (format === 'auto' && record.ts !== undefined && (record.uid !== undefined || record.note !== undefined)) ? 'zeek' : null;
  if (!source) throw new Error('Unsupported sensor schema; use Suricata EVE or Zeek JSON');
  const observedAt = time(source === 'suricata' ? record.timestamp : record.ts);
  const sourceTimestamp = source === 'suricata' ? clean(record.timestamp, 80) : number(record.ts, 8640000000000);
  if (!observedAt) throw new Error('Missing source timestamp or timezone; record excluded');
  if (Date.parse(observedAt) > Date.parse(importedAt) + 300000) throw new Error('Source timestamp is over five minutes in the future; check sensor clock');
  const sid = clean(String(sourceId), 160);
  if (!sid) throw new Error('Source identifier required');
  let kind, fields, summary, severity = 'informational', sourceEventId;
  if (source === 'suricata') {
    kind = clean(record.event_type, 40);
    if (!['alert', 'flow', 'dns', 'tls'].includes(kind)) throw new Error('Unsupported Suricata record kind; accepted: alert, flow, DNS, TLS');
    const flow = obj(record.flow), alert = obj(record.alert), dns = obj(record.dns), tls = obj(record.tls);
    if (kind === 'alert' && !identifier(alert.signature_id) && !clean(alert.signature)) throw new Error('Alert record lacks a rule signature');
    sourceEventId = identifier(record.flow_id);
    fields = compact({src_ip: clean(record.src_ip, 80), dest_ip: clean(record.dest_ip, 80), src_port: port(record.src_port), dest_port: port(record.dest_port),
      protocol: clean(record.proto, 24), app_protocol: clean(record.app_proto, 40), flow_id: sourceEventId, transaction_id: identifier(record.tx_id), packet_index: identifier(record.pcap_cnt),
      ...(kind === 'flow' ? {bytes: sumKnown([number(flow.bytes_toserver), number(flow.bytes_toclient)]), packets: sumKnown([number(flow.pkts_toserver), number(flow.pkts_toclient)]),
        bytes_to_server: number(flow.bytes_toserver), bytes_to_client: number(flow.bytes_toclient), duration_seconds: number(flow.age), conn_state: clean(flow.state, 32)} : {}),
      ...(kind === 'alert' ? {signature_id: identifier(alert.signature_id), signature: clean(alert.signature), category: clean(alert.category), action: clean(alert.action, 40), source_severity: number(alert.severity, 255)} : {}),
      ...(kind === 'dns' ? {dns_name: clean(dns.rrname ?? obj(dns.queries?.[0]).rrname, 253), dns_rcode: clean(dns.rcode, 40), dns_type: clean(dns.rrtype ?? obj(dns.queries?.[0]).rrtype, 40)} : {}),
      ...(kind === 'tls' ? {tls_sni: clean(tls.sni, 253), tls_version: clean(tls.version, 40)} : {}),
    });
    severity = kind === 'alert' ? ({1: 'high', 2: 'medium', 3: 'low'}[alert.severity] ?? 'unknown') : severity;
    summary = kind === 'alert' ? `Suricata rule match: ${fields.signature ?? `signature ${fields.signature_id}`}`
      : kind === 'flow' ? 'Suricata logged a network flow'
      : kind === 'dns' ? `Suricata DNS record${fields.dns_name ? `: ${fields.dns_name}` : ''}` : `Suricata TLS record${fields.tls_sni ? `: ${fields.tls_sni}` : ''}`;
  } else {
    const connection = obj(record.id);
    const field = (flat, nested) => record[flat] ?? connection[nested];
    kind = typeof record.note === 'string' ? 'notice' : record.query !== undefined ? 'dns' : record.version !== undefined ? 'tls'
      : ['orig_bytes', 'resp_bytes', 'conn_state', 'duration', 'orig_pkts', 'resp_pkts'].some(key => record[key] !== undefined) ? 'connection' : null;
    if (!kind) throw new Error('Unsupported Zeek record kind; accepted: connection, notice, DNS, TLS');
    if (kind === 'connection' && !identifier(record.uid)) throw new Error('Zeek connection record lacks uid');
    sourceEventId = identifier(record.uid);
    fields = compact({src_ip: clean(field('id.orig_h', 'orig_h') ?? record.src, 80), dest_ip: clean(field('id.resp_h', 'resp_h') ?? record.dst, 80),
      src_port: port(field('id.orig_p', 'orig_p')), dest_port: port(field('id.resp_p', 'resp_p')), protocol: clean(record.proto, 24), flow_id: sourceEventId,
      ...(kind === 'connection' ? {bytes: sumKnown([number(record.orig_bytes), number(record.resp_bytes)]), packets: sumKnown([number(record.orig_pkts), number(record.resp_pkts)]),
        bytes_to_server: number(record.orig_bytes), bytes_to_client: number(record.resp_bytes), duration_seconds: number(record.duration), conn_state: clean(record.conn_state, 32), app_protocol: clean(record.service, 40)} : {}),
      ...(kind === 'notice' ? {notice_type: clean(record.note), actions: Array.isArray(record.actions) ? record.actions.slice(0, 8).map(v => clean(v, 80)).filter(Boolean) : undefined} : {}),
      ...(kind === 'dns' ? {dns_name: clean(record.query, 253), dns_rcode: clean(record.rcode_name, 40), dns_type: clean(record.qtype_name, 40)} : {}),
      ...(kind === 'tls' ? {tls_sni: clean(record.server_name, 253), tls_version: clean(record.version, 40)} : {}),
    });
    severity = kind === 'notice' ? 'unknown' : severity;
    summary = kind === 'notice' ? `Zeek notice: ${fields.notice_type}` : kind === 'connection' ? 'Zeek logged a network connection'
      : kind === 'dns' ? `Zeek DNS record${fields.dns_name ? `: ${fields.dns_name}` : ''}` : `Zeek TLS record${fields.tls_sni ? `: ${fields.tls_sni}` : ''}`;
  }
  const id = `sensor_${hash(JSON.stringify([sid, source, sourceTimestamp, kind, fields]))}`;
  const actual = kind === 'alert' || kind === 'notice';
  return {id, source_id: sid, source, source_name: clean(sourceName, 160), source_event_id: sourceEventId, source_timestamp: sourceTimestamp, observed_at: observedAt,
    received_at: time(importedAt), kind, actual_security_event: actual, severity, summary,
    classification: actual ? 'sensor_reported_event' : 'network_telemetry', attack_confirmed: false, source_authenticity: 'not independently verified', fields,
    evidence: {id, source, source_id: sid, source_event_id: sourceEventId, timestamp: observedAt, source_timestamp: sourceTimestamp, fields},
    limitation: actual ? 'A real record reported by the supplied sensor export; rule matches and notices may include false positives or policy events. Confirm with source context.' : 'Logged network metadata; does not prove suspicious activity or comprehensive network visibility'};
}

export function summarizeSecurityImport(batch) {
  const events = Array.isArray(batch?.events) ? batch.events : [];
  return {records: events.length, actual_security_events: events.filter(x => x.actual_security_event).length,
    flows: events.filter(x => x.kind === 'flow' || x.kind === 'connection').length, dns: events.filter(x => x.kind === 'dns').length,
    tls: events.filter(x => x.kind === 'tls').length, rejected: batch?.rejected_count ?? 0, duplicates: batch?.duplicate_count ?? 0,
    by_source: count(events, 'source'), by_kind: count(events, 'kind')};
}

export function parseSecurityLog(text, options = {}) {
  if (typeof text !== 'string') throw new Error('Expected JSON or newline-delimited JSON text');
  if (new TextEncoder().encode(text).length > SECURITY_LIMITS.maxBytes) throw new Error('Sensor import exceeds 2 MB; use a bounded export or the local live sensor');
  const importedAt = time(options.importedAt ?? new Date().toISOString());
  if (!importedAt) throw new Error('Import time must contain a timezone');
  const limit = Math.max(1, Math.min(SECURITY_LIMITS.maxRecords, Number.isInteger(options.maxRecords) ? options.maxRecords : SECURITY_LIMITS.maxRecords));
  const events = [], rejected = [], seen = new Set();
  let rejectedCount = 0, duplicates = 0, recordCount = 0;
  const reject = (line, reason) => {rejectedCount++; if (rejected.length < 30) rejected.push({line, reason});};
  // Keep 64-bit Suricata identifiers lossless rather than rounding them via JSON.parse.
  const parse = line => JSON.parse(line.replace(/("flow_id"\s*:\s*)(\d{16,})(?=\s*[,}])/g, '$1"$2"'));
  const accept = (raw, line) => {
    recordCount++;
    if (recordCount > limit) {reject(line, 'Record limit reached; export a narrower observation window'); return;}
    try {
      const event = normalizeSecurityRecord(raw, {...options, importedAt});
      if (seen.has(event.id)) {duplicates++; return;}
      seen.add(event.id); events.push(event);
    } catch (error) {reject(line, String(error.message).slice(0, 200));}
  };
  const trimmed = text.trim();
  if (trimmed.startsWith('[')) {
    let raw;
    try {raw = parse(trimmed);} catch {throw new Error('Malformed JSON array; use valid JSON or NDJSON');}
    if (!Array.isArray(raw)) throw new Error('Expected a JSON array');
    raw.forEach((record, i) => accept(record, i + 1));
  } else {
    text.split(/\r?\n/).forEach((line, index) => {
      if (!line.trim()) return;
      if (new TextEncoder().encode(line).length > SECURITY_LIMITS.maxLineBytes) {recordCount++; reject(index + 1, 'Oversized sensor row excluded'); return;}
      try {accept(parse(line), index + 1);} catch {recordCount++; reject(index + 1, 'Malformed JSON row excluded');}
    });
  }
  events.sort((a, b) => Date.parse(a.observed_at) - Date.parse(b.observed_at));
  const sources = [...new Set(events.map(x => x.source))], kinds = [...new Set(events.map(x => x.kind))];
  const coverage = {state: options.mode === 'live_log_tail' ? 'live_log_tail' : 'historical_import', sources, kinds,
    first_observed_at: events[0]?.observed_at ?? null, last_observed_at: events.at(-1)?.observed_at ?? null,
    imported_at: importedAt, network_visibility: 'Only records exported by the connected sensor; no packet capture by Observatory',
    missing: ['Traffic outside sensor placement', 'Encrypted payload contents', 'Source authenticity verification',
      ...(!kinds.some(k => k === 'alert' || k === 'notice') ? ['No alert or notice rows in this window; this is not proof of no security events'] : []),
      ...(!kinds.some(k => k === 'flow' || k === 'connection') ? ['No flow or connection rows; byte and activity comparisons unavailable'] : []),
      ...(rejectedCount ? ['Some rows were excluded; coverage is incomplete'] : [])]};
  const batch = {events, rejected, rejected_count: rejectedCount, duplicate_count: duplicates, input_records: recordCount, coverage, privacy: SECURITY_PRIVACY};
  return {...batch, summary: summarizeSecurityImport(batch)};
}

/** Historical record-level features. No artificial missing-time zeroes or forecasts. */
export function analyzeSecurityEvents(events, {binSeconds = 60} = {}) {
  if (!Array.isArray(events) || events.length > SECURITY_LIMITS.maxRecords) throw new Error('Analyze at most 2000 normalized sensor records');
  if (!Number.isInteger(binSeconds) || binSeconds < 1 || binSeconds > 3600) throw new Error('binSeconds must be 1–3600');
  const groups = new Map(), observations = [], seen = new Set();
  for (const event of events) {
    if (!event || typeof event.id !== 'string' || seen.has(event.id) || !time(event.observed_at)) continue;
    seen.add(event.id);
    if (!['flow', 'connection'].includes(event.kind)) continue;
    const source = event.source_id;
    if (!groups.has(source)) groups.set(source, []);
    groups.get(source).push(event);
  }
  for (const [sourceId, rows] of groups) {
    rows.sort((a, b) => Date.parse(a.observed_at) - Date.parse(b.observed_at));
    const detector = new SentinelPort();
    let samples = 0;
    for (const row of rows) {
      if (number(row.fields?.bytes) === null) continue;
      const feature = Math.log1p(row.fields.bytes), scored = detector.step(feature);
      samples++;
      const unusual = !scored.warmup && scored.status !== 'GREEN';
      observations.push({id: `network_${row.id}`, source_id: sourceId, metric: 'log1p_logged_connection_bytes', observed_at: row.observed_at,
        value: row.fields.bytes, feature, event_score: scored.warmup ? null : scored.event_score, samples,
        state: scored.warmup ? 'calibrating' : unusual ? 'unusual' : 'baseline', warmup_remaining: Math.max(0, 12 - samples),
        actual_security_event: false, confidence: 'experimental', evidence_ids: [row.id],
        summary: unusual ? 'Logged connection volume differs from the provisional Sentinel baseline' : scored.warmup ? 'Learning a provisional connection-volume baseline' : 'Connection volume fits the provisional baseline',
        engine: ENGINE_PROVENANCE.engine, engine_scope: 'One numeric feature: log1p of bytes reported per logged connection; not threat classification',
        limitation: 'No labeled network validation. Connection records may have different durations. This flag needs review; it does not establish an attack or predict future activity'});
    }
    // Explicitly observed buckets only: sparse logging cannot be interpreted as zero traffic.
    const bins = new Map();
    for (const row of rows) {
      if (!row.fields.src_ip) continue;
      const start = Math.floor(Date.parse(row.observed_at) / (binSeconds * 1000)) * binSeconds * 1000;
      const key = `${start}|${row.fields.src_ip}`;
      if (!bins.has(key)) bins.set(key, {start, origin: row.fields.src_ip, records: []});
      bins.get(key).records.push(row);
    }
    for (const {start, origin, records} of bins.values()) {
      const endpoints = new Set(records.map(x => x.fields.dest_ip).filter(Boolean));
      const ports = new Set(records.map(x => x.fields.dest_port).filter(x => x !== undefined));
      if (endpoints.size >= 20 || ports.size >= 20) observations.push({id: `fanout_${hash(`${sourceId}:${start}:${origin}`)}`, source_id: sourceId,
        metric: 'logged_destination_fanout', observed_at: new Date(start).toISOString(), value: endpoints.size, ports: ports.size,
        event_score: null, state: 'unusual', actual_security_event: false, confidence: 'heuristic',
        evidence_ids: records.slice(0, 50).map(x => x.id), summary: `One origin has high logged destination diversity: ${endpoints.size} hosts / ${ports.size} ports in an observed ${binSeconds}-second bucket`,
        limitation: 'A transparent review rule (20 hosts or ports), not a scan verdict. Normal software can contact many destinations. Source timestamps may refer to connection start or logging time'});
    }
  }
  observations.sort((a, b) => Date.parse(a.observed_at) - Date.parse(b.observed_at));
  return {observations, anomalies: observations.filter(x => x.state === 'unusual'),
    summary: {records: seen.size, actual_security_events: events.filter(x => x.actual_security_event).length,
      unusual_observations: observations.filter(x => x.state === 'unusual').length, calibrated_sources: [...groups].filter(([, rows]) => rows.filter(x => number(x.fields?.bytes) !== null).length > 12).length},
    coverage: {state: groups.size ? 'experimental_network_features' : 'insufficient_data', missing: ['No attack ground truth', 'No throughput estimation', 'No zero-fill for unobserved time buckets', 'No future prediction']}};
}
