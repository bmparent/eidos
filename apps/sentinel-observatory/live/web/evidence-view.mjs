/** Presentation-only evidence coordination. No detector, sensor or model changes. */
const timestamp = value => Date.parse(value);
export const VIEW_LIMIT = 2000;

export function retainUniqueEvents(events, limit = VIEW_LIMIT) {
  const seen = new Set();
  return events.filter(event => {
    if (!event || typeof event.id !== 'string' || !Number.isFinite(timestamp(event.observed_at)) || seen.has(event.id)) return false;
    seen.add(event.id); return true;
  }).sort((a, b) => timestamp(b.observed_at) - timestamp(a.observed_at)).slice(0, limit);
}

export function evidenceEndpoints(event) {
  if (event?.kind === 'collector' && event.fields?.head_attempted === false) return {source: null, target: null};
  return {source: event?.fields?.src_ip || event?.fields?.src_endpoint || null,
    target: event?.fields?.dest_ip || event?.fields?.dest_endpoint || null};
}

export function selectEvidenceWindow(events, {start = null, end = null, host = null, kind = 'all'} = {}) {
  return events.filter(event => {
    const t = timestamp(event.observed_at), endpoints = evidenceEndpoints(event);
    return (start === null || t >= start) && (end === null || t <= end)
      && (!host || endpoints.source === host || endpoints.target === host)
      && (kind === 'all' || kind === 'alerts' && ['alert', 'notice'].includes(event.kind)
        || kind === 'connections' && ['flow', 'connection'].includes(event.kind) || kind === event.kind);
  });
}

/** Only occupied bins are emitted: absent records never establish zero traffic. */
export function evidenceSeries(events, maxBins = 40) {
  if (!events.length) return [];
  const times = events.map(event => timestamp(event.observed_at)).filter(Number.isFinite);
  const first = Math.min(...times), last = Math.max(...times);
  const width = Math.max(1000, Math.ceil((last - first + 1) / Math.max(1, maxBins)));
  const bins = new Map();
  for (const event of events) {
    const t = timestamp(event.observed_at); if (!Number.isFinite(t)) continue;
    const index = Math.floor((t - first) / width);
    const bin = bins.get(index) || {index, start: first + index * width, count: 0, securityEvents: 0, eventIds: []};
    bin.count++; bin.securityEvents += event.actual_security_event === true ? 1 : 0; bin.eventIds.push(event.id); bins.set(index, bin);
  }
  return [...bins.values()].sort((a, b) => a.index - b.index);
}

export function sourceEvidenceStatus(source, {active = false, hidden = false, issue = null, unauthorized = false, lastReceivedAt = null, now = Date.now()} = {}) {
  if (source?.mode === 'fixture') return {state: 'fixture', label: 'LABELED FIXTURE · SYNTHETIC', explanation: 'Deterministic test records; no sensor observation or security result.'};
  if (source?.mode !== 'live') return {state: 'historical', label: 'HISTORICAL IMPORT', explanation: 'A retained capture; current source coverage is unknown.'};
  if (unauthorized) return {state: 'unauthorized', label: 'UNAUTHORIZED', explanation: 'Access to the source was denied. Retained evidence is preserved; current traffic is unknown.'};
  if (issue) return {state: active ? 'reconnecting' : 'unavailable', label: active ? 'RECONNECTING' : 'UNAVAILABLE', explanation: 'The source cannot currently be read. Retained evidence is preserved; current traffic is unknown.'};
  if (!active || hidden) return {state: 'paused', label: hidden ? 'PAUSED · TAB HIDDEN' : 'PAUSED', explanation: 'No fresh source requests are being made; this does not establish zero traffic.'};
  if (!lastReceivedAt) return {state: 'connecting', label: 'CONNECTING', explanation: 'Waiting for a source receipt; current traffic is unknown.'};
  if (now - timestamp(lastReceivedAt) > 20000) return {state: 'stale', label: 'STALE FEED', explanation: 'No recent feed receipt; current traffic is unknown.'};
  if (source?.coverage?.truncated || source?.rejectedCount) return {state: 'partial', label: 'PARTIAL COVERAGE', explanation: 'The feed is reachable, but bounded or rejected records limit coverage.'};
  const latest = Math.max(...(source?.events || []).map(event => timestamp(event.observed_at)));
  return {state: 'receiving', label: 'RECEIVING', explanation: !Number.isFinite(latest) ? 'No retained events in the supplied source window; this is not an all-clear result.'
    : now - latest > 60000 ? 'The feed is reachable; its newest event is older than one minute.' : 'Recent source events received; coverage remains limited to this source.'};
}
