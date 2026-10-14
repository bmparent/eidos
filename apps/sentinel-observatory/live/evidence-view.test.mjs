import test from 'node:test';
import assert from 'node:assert/strict';
import {retainUniqueEvents, selectEvidenceWindow, evidenceSeries, sourceEvidenceStatus, evidenceEndpoints} from './web/evidence-view.mjs';
const start = Date.parse('2026-10-13T12:00:00Z');
const event = (id, seconds, extra = {}) => ({id, observed_at: new Date(start + seconds * 1000).toISOString(), kind: 'flow', fields: {src_ip: '192.0.2.1', dest_ip: '192.0.2.2'}, ...extra});

test('repeated records in one batch and across reconnects do not inflate evidence', () => {
  const rows = retainUniqueEvents([event('a', 0), event('a', 0), event('b', 1), {id: 'bad', observed_at: 'invalid'}]);
  assert.deepEqual(rows.map(row => row.id), ['b', 'a']);
  assert.equal(retainUniqueEvents([...rows, ...rows]).length, 2);
});
test('bounded retention keeps the newest records without changing stable identity', () => {
  assert.deepEqual(retainUniqueEvents([event('a', 0), event('b', 1), event('c', 2)], 2).map(row => row.id), ['c', 'b']);
});
test('one immutable time window, host and kind selection drive all derived views', () => {
  const rows = [event('a', 0), event('b', 60), event('c', 120, {kind: 'alert'})];
  const window = {start: start + 60000, end: start + 120000, host: '192.0.2.2', kind: 'alerts'};
  assert.deepEqual(selectEvidenceWindow(rows, window).map(row => row.id), ['c']);
  assert.deepEqual(window, {start: start + 60000, end: start + 120000, host: '192.0.2.2', kind: 'alerts'});
  assert.equal(selectEvidenceWindow(rows, {...window, host: 'missing'}).length, 0);
});
test('time series does not fabricate empty bins or bandwidth', () => {
  const bins = evidenceSeries([event('a', 0), event('b', 120, {actual_security_event: true})]);
  assert.equal(bins.length, 2); assert.equal(bins[1].securityEvents, 1);
  assert.deepEqual(bins.flatMap(bin => bin.eventIds), ['a', 'b']);
  assert.equal(bins.some(bin => bin.count === 0), false);
});
test('preflight never creates an observed network relationship', () => {
  assert.deepEqual(evidenceEndpoints(event('a', 0, {kind: 'collector', fields: {head_attempted: false, src_endpoint: 'a', dest_endpoint: 'b'}})), {source: null, target: null});
});
test('missing source receipts, disconnection and auth denial never imply zero traffic', () => {
  const source = {mode: 'live', events: []};
  for (const settings of [{}, {active: true}, {active: true, issue: 'HTTP 503'}, {active: true, unauthorized: true}, {active: true, lastReceivedAt: new Date(start).toISOString(), now: start + 21000}]) {
    const status = sourceEvidenceStatus(source, settings);
    assert.match(status.explanation, /unknown|zero traffic/);
  }
});
test('fresh, partial, historical and explicitly synthetic sources remain distinct', () => {
  const live = {mode: 'live', events: [event('a', 0)]};
  const settings = {active: true, now: start + 1000, lastReceivedAt: new Date(start).toISOString()};
  assert.equal(sourceEvidenceStatus(live, settings).state, 'receiving');
  assert.equal(sourceEvidenceStatus({...live, rejectedCount: 1}, settings).state, 'partial');
  assert.equal(sourceEvidenceStatus({mode: 'import'}).state, 'historical');
  assert.equal(sourceEvidenceStatus({mode: 'fixture'}).state, 'fixture');
});
