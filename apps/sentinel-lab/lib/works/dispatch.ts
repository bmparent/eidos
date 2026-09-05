import { timingSafeEqual } from 'node:crypto';
import { platformDatabase } from './database';
import { json, readText, type Context, type PlatformEnv } from './vendor/functions/_shared/platform/core';
import * as assistant from './vendor/functions/api/assistant';
import * as config from './vendor/functions/api/public-config';
import * as threads from './vendor/functions/api/community/threads';
import * as replies from './vendor/functions/api/community/replies';
import * as agents from './vendor/functions/api/community/agents';
import * as moderate from './vendor/functions/api/community/moderate';
import * as maintenance from './vendor/functions/api/community/maintenance';
import * as checkout from './vendor/functions/api/shop/checkout';
import * as status from './vendor/functions/api/shop/status';
import * as download from './vendor/functions/api/shop/download';
import * as webhook from './vendor/functions/api/shop/webhook';
import * as feed from './vendor/functions/community/feed';
import * as sitemap from './vendor/functions/community/sitemap.xml';
import * as thread from './vendor/functions/community/thread/[id]';

type Handler = (context: Context) => Promise<Response>;
type Module = { onRequestGet?: Handler; onRequestPost?: Handler };
const routes: Record<string, Module> = {
  '/api/assistant': assistant, '/api/public-config': config,
  '/api/community/threads': threads, '/api/community/replies': replies,
  '/api/community/agents': agents, '/api/community/moderate': moderate,
  '/api/community/maintenance': maintenance,
  '/api/shop/checkout': checkout, '/api/shop/status': status,
  '/api/shop/download': download, '/api/shop/webhook': webhook,
  '/community/feed': feed, '/community/sitemap.xml': sitemap,
};
const names = [
  'EIDOS_ADMIN_TOKEN', 'EIDOS_RATE_SECRET', 'OPENAI_API_KEY', 'EIDOS_ASSISTANT_MODEL',
  'EIDOS_AI_ENABLED', 'EIDOS_AI_DAILY_TOKENS', 'EIDOS_PROACTIVE_ENABLED', 'EIDOS_MAINTENANCE_TOKEN',
  'TURNSTILE_SITE_KEY', 'TURNSTILE_SECRET_KEY', 'GA_MEASUREMENT_ID', 'PUBLIC_SITE_URL',
  'STRIPE_SECRET_KEY', 'EIDOS_KIT_WEBHOOK_SECRET', 'EIDOS_SHOP_ENABLED',
] as const;
export function platformEnvironment(source: Record<string, string | undefined> = process.env): PlatformEnv {
  const selected = Object.fromEntries(names.map(name => [name, source[name]]));
  return { ...selected, EIDOS_RUNTIME: 'sentinel', EIDOS_DB: platformDatabase(source) };
}
function authenticated(request: Request, expected?: string) {
  const actual = request.headers.get('x-eidos-platform-token') || '';
  if (!expected || expected.length < 32 || Buffer.byteLength(actual) !== Buffer.byteLength(expected)) return false;
  return timingSafeEqual(Buffer.from(actual), Buffer.from(expected));
}

/** Accept only the authenticated Eidos Works relay. Never import the research executor here. */
export async function dispatchWorks(request: Request, source: Record<string, string | undefined> = process.env, suppliedEnv?: PlatformEnv) {
  try {
    const inputUrl = new URL(request.url);
    const path = inputUrl.pathname.replace(/^\/api\/works\/v1/, '').replace(/\/$/, '') || '/';
    if (path === '/health' && request.method === 'GET')
      return json({ service: 'eidos-works-platform', version: 1 });
    if (!authenticated(request, source.EIDOS_PLATFORM_TOKEN)) return json({ error: 'Studio relay authentication required.' }, 401);
    const expectedSite = new URL(source.PUBLIC_SITE_URL || 'https://eidos-works.com').origin;
    const actualSite = request.headers.get('x-eidos-site-origin');
    // Preview hosts require an explicit exact-origin allowlist. Never accept wildcard *.pages.dev.
    const allowed = [expectedSite, ...(source.EIDOS_PREVIEW_ORIGINS || '').split(',').map(s => s.trim()).filter(Boolean)];
    if (!actualSite || !allowed.includes(actualSite)) return json({ error: 'Unrecognized studio origin.' }, 403);
    const match = path.match(/^\/community\/thread\/([a-zA-Z0-9-]+)$/);
    const endpoint = match ? thread : routes[path];
    if (!endpoint) return json({ error: 'Unknown studio endpoint.' }, 404);
    const handler = request.method === 'GET' ? endpoint.onRequestGet :
      request.method === 'POST' ? (endpoint as Module).onRequestPost : undefined;
    if (!handler) return json({ error: 'Method not supported.' }, 405);
    const headers = new Headers();
    for (const name of ['origin', 'content-type', 'authorization', 'stripe-signature']) {
      const value = request.headers.get(name);
      if (value) headers.set(name, value);
    }
    headers.set('CF-Connecting-IP', request.headers.get('x-eidos-client-ip') || 'unknown');
    const payload = request.method === 'POST' ? await readText(request, path === '/api/shop/webhook' ? 64000 : 12000) : undefined;
    const forwarded = new Request(actualSite + path + inputUrl.search, {
      method: request.method, headers, body: payload,
    });
    const env = suppliedEnv || platformEnvironment(source);
    // Build/test callers may inject local fixtures; production environment never enables bypasses.
    return handler({ request: forwarded, env, params: match ? { id: match[1] } : undefined });
  } catch {
    return json({ error: 'The studio service is temporarily unavailable.' }, 503);
  }
}
