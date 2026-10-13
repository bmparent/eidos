import {handleObservation, ipCollectorConfiguration} from './collector.mjs';

// build_worker.py embeds the exact inspected web sources here. No runtime CDN.
const ASSETS = /* OBSERVATORY_ASSETS */ Object.freeze({});
const VERSION = 'sentinel-live-v3';
const SECURITY = {
  'cache-control': 'no-store',
  'x-content-type-options': 'nosniff',
  'referrer-policy': 'no-referrer',
  'permissions-policy': 'camera=(), microphone=(), geolocation=()',
};

function response(body, type, status = 200, extra = {}) {
  return new Response(body, {status, headers: {...SECURITY, 'content-type': type, ...extra}});
}

export default {
  async fetch(request, env, ctx) {
    const path = new URL(request.url).pathname;
    // Sites dispatch enforces this application's unchanged owner-only access
    // policy, including service access. Never forward identity or auth headers.
    if (path === '/api/observe') return handleObservation(request, env, ctx);
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
        model: {connected: false, mode: 'browser_local_qwen_opt_in', local_adapter_available: true, browser_adapter_available: true, model_id: 'Qwen3-0.6B', hardware: 'WebGPU', eido_status: 'evaluation_candidate'},
        security_sensor: {connected: false, mode: 'browser_log_import', formats: ['Suricata EVE JSON', 'Zeek JSON'], reason: 'No packet or security sensor is attached to this hosted service. Import a real log or connect the local sensor companion.'},
        recorded_url: '/recorded',
        transport: 'Cloudflare Worker public HTTP HEAD',
        sampling: 'While connected and visible in your browser; no unattended background monitor',
        history: 'Browser-local retained records; export JSON for a portable copy',
        missing_coverage: ['Visitor traffic', 'Packet capture', 'Internal services', 'Origin processing time', 'TLS certificate inspection'],
      }), 'application/json; charset=utf-8');
    }
    if (path === '/api/security-events') {
      return response(JSON.stringify({connected: false, observed_at: new Date().toISOString(), source_id: null, source_name: null, format: 'auto', text: '', coverage: {state: 'disconnected', missing: ['Network traffic sensor', 'IDS alerts', 'Firewall events']}}), 'application/json; charset=utf-8');
    }
    const routes = {'/': ['html', 'text/html; charset=utf-8'], '/app.js': ['app', 'text/javascript; charset=utf-8'], '/styles.css': ['css', 'text/css; charset=utf-8'], '/brain.mjs': ['brain', 'text/javascript; charset=utf-8'], '/browser-interpreter.mjs': ['interpreter', 'text/javascript; charset=utf-8'], '/browser-model-worker.mjs': ['model_worker', 'text/javascript; charset=utf-8'], '/security-events.mjs': ['security', 'text/javascript; charset=utf-8'], '/network-panel.mjs': ['network_panel', 'text/javascript; charset=utf-8'], '/recorded': ['recorded', 'text/html; charset=utf-8']};
    const route = routes[path];
    if (!route || !ASSETS[route[0]]) return response('Not found', 'text/plain; charset=utf-8', 404);
    return response(request.method === 'HEAD' ? null : ASSETS[route[0]], route[1]);
  },
};
