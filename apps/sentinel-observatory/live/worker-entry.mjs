import {handleObservation, ipCollectorConfiguration} from './collector.mjs';
import {capability, snapshot, recordObservation} from './hosted-sensor.mjs';
import {hostedInterpretationCapability, handleInterpretation} from './sol-interpreter.mjs';

// build_worker.py embeds the exact inspected web sources here. No runtime CDN.
const ASSETS = /* OBSERVATORY_ASSETS */ Object.freeze({});
const VERSION = 'sentinel-live-v5-sol';
const SECURITY = {
  'cache-control': 'no-store',
  'x-content-type-options': 'nosniff',
  'referrer-policy': 'no-referrer',
  'permissions-policy': 'camera=(), microphone=(), geolocation=()',
  'content-security-policy': "frame-ancestors 'self' https://chatgpt.com",
};

function response(body, type, status = 200, extra = {}) {
  return new Response(body, {status, headers: {...SECURITY, 'content-type': type, ...extra}});
}

function secureResponse(result) {
  const headers = new Headers(result.headers);
  for (const [key, value] of Object.entries(SECURITY)) headers.set(key, value);
  return new Response(result.body, {status: result.status, headers});
}

export default {
  async fetch(request, env, ctx) {
    const path = new URL(request.url).pathname;
    // Sites dispatch enforces this application's unchanged owner-only access
    // policy, including service access. Never forward identity or auth headers.
    if (path === '/api/observe') {
      const started = performance.now();
      const result = await handleObservation(request, env, ctx);
      // A functional-check label requires the configured server credential;
      // arbitrary browser labels/headers never establish verification context.
      const secret = env?.OBSERVATORY_COLLECTOR_SECRET;
      const testContext = typeof secret === 'string' && secret.length >= 32 && request.headers.get('X-Sentinel-Functional-Check') === secret ? 'functional_check' : undefined;
      const stored = await recordObservation(env, request, result, {request_duration_ms: performance.now() - started, test_context: testContext});
      const headers = new Headers(result.headers);
      headers.set('X-Sentinel-Sensor-Write', stored.status);
      return secureResponse(new Response(result.body, {status: result.status, headers}));
    }
    if (path === '/api/interpret') {
      return secureResponse(await handleInterpretation(request, env, ctx));
    }
    if (!['GET', 'HEAD'].includes(request.method)) {
      return response(JSON.stringify({error: 'method_not_allowed'}), 'application/json', 405, {allow: 'GET, HEAD'});
    }
    if (path === '/api/capabilities') {
      return response(JSON.stringify({
        version: VERSION,
        mode: 'hosted',
        direct_ip: Boolean(ipCollectorConfiguration(env)),
        ip_transport: ipCollectorConfiguration(env) ? 'Authenticated Node collector; validated literal public address pinned, TLS verified' : 'No direct IP collector configured',
        interval_ms: 15000,
        max_targets: 4,
        model: {...hostedInterpretationCapability(env), eido_status: 'evaluation_candidate'},
        hosted_interpretation: hostedInterpretationCapability(env),
        security_sensor: capability(env),
        recorded_url: '/recorded',
        transport: 'Cloudflare Worker public HTTP HEAD',
        sampling: 'While connected and visible in your browser; no unattended background monitor',
        history: 'Browser-local retained records; export JSON for a portable copy',
        missing_coverage: ['Visitor traffic', 'Packet capture', 'Internal services', 'Origin processing time', 'TLS certificate inspection'],
      }), 'application/json; charset=utf-8');
    }
    if (path === '/api/security-events') {
      const sensor = await snapshot(env);
      return response(JSON.stringify(sensor), 'application/json; charset=utf-8', sensor.connected ? 200 : 503);
    }
    const routes = {'/': ['html', 'text/html; charset=utf-8'], '/app.js': ['app', 'text/javascript; charset=utf-8'], '/styles.css': ['css', 'text/css; charset=utf-8'], '/brain.mjs': ['brain', 'text/javascript; charset=utf-8'], '/browser-interpreter.mjs': ['interpreter', 'text/javascript; charset=utf-8'], '/browser-model-worker.mjs': ['model_worker', 'text/javascript; charset=utf-8'], '/security-events.mjs': ['security', 'text/javascript; charset=utf-8'], '/network-panel.mjs': ['network_panel', 'text/javascript; charset=utf-8'], '/evidence-view.mjs': ['evidence_view', 'text/javascript; charset=utf-8'], '/evidence-fixture': ['evidence_fixture', 'text/html; charset=utf-8'], '/recorded': ['recorded', 'text/html; charset=utf-8']};
    const route = routes[path];
    if (!route || !ASSETS[route[0]]) return response('Not found', 'text/plain; charset=utf-8', 404);
    return response(request.method === 'HEAD' ? null : ASSETS[route[0]], route[1]);
  },
};
