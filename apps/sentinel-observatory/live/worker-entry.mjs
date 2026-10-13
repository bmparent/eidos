import {handleObservation} from './collector.mjs';

// build_worker.py embeds the exact inspected web sources here. No runtime CDN.
const ASSETS = /* OBSERVATORY_ASSETS */ Object.freeze({});
const VERSION = 'sentinel-live-v2';
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
        direct_ip: false,
        interval_ms: 15000,
        max_targets: 4,
        model: {connected: false, mode: 'deterministic_evidence_rules', local_adapter_available: true},
        recorded_url: '/recorded',
        transport: 'Cloudflare Worker public HTTP HEAD',
        sampling: 'While connected and visible in your browser; no unattended background monitor',
        history: 'Browser-local retained records; export JSON for a portable copy',
        missing_coverage: ['Visitor traffic', 'Packet capture', 'Internal services', 'Origin processing time', 'TLS certificate inspection'],
      }), 'application/json; charset=utf-8');
    }
    const routes = {'/': ['html', 'text/html; charset=utf-8'], '/app.js': ['app', 'text/javascript; charset=utf-8'], '/styles.css': ['css', 'text/css; charset=utf-8'], '/brain.mjs': ['brain', 'text/javascript; charset=utf-8'], '/recorded': ['recorded', 'text/html; charset=utf-8']};
    const route = routes[path];
    if (!route || !ASSETS[route[0]]) return response('Not found', 'text/plain; charset=utf-8', 404);
    return response(request.method === 'HEAD' ? null : ASSETS[route[0]], route[1]);
  },
};
