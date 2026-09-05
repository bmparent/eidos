export interface Statement {
  bind(...values: unknown[]): Statement;
  first<T = Record<string, unknown>>(): Promise<T | null>;
  all<T = Record<string, unknown>>(): Promise<{ results: T[] }>;
  run(): Promise<{ meta: { changes: number } }>;
}
export interface Database {
  prepare(sql: string): Statement;
  batch(statements: Statement[]): Promise<unknown[]>;
}
export interface PlatformEnv {
  EIDOS_RUNTIME?: 'sentinel';
  EIDOS_DB?: Database;
  EIDOS_ADMIN_TOKEN?: string;
  EIDOS_RATE_SECRET?: string;
  OPENAI_API_KEY?: string;
  EIDOS_ASSISTANT_MODEL?: string;
  EIDOS_AI_ENABLED?: string;
  EIDOS_AI_DAILY_TOKENS?: string;
  EIDOS_PROACTIVE_ENABLED?: string;
  EIDOS_MAINTENANCE_TOKEN?: string;
  TURNSTILE_SITE_KEY?: string;
  TURNSTILE_SECRET_KEY?: string;
  EIDOS_LOCAL_TEST?: string;
  GA_MEASUREMENT_ID?: string;
  PUBLIC_SITE_URL?: string;
  STRIPE_SECRET_KEY?: string;
  EIDOS_KIT_WEBHOOK_SECRET?: string;
  EIDOS_SHOP_ENABLED?: string;
}
export type Context = {
  request: Request;
  env: PlatformEnv;
  params?: Record<string, string>;
  waitUntil?: (promise: Promise<unknown>) => void;
};
export class HttpError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}
export const clean = (v: unknown, max = 1000) =>
  typeof v === 'string' ? v.trim().slice(0, max) : '';
export const record = (v: unknown): v is Record<string, unknown> =>
  Boolean(v && typeof v === 'object' && !Array.isArray(v));
export function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      'cache-control': 'no-store',
      'x-content-type-options': 'nosniff',
      'referrer-policy': 'no-referrer',
    },
  });
}
export function guarded(handler: (context: Context) => Promise<Response>) {
  return async (context: Context) => {
    try {
      return await handler(context);
    } catch (error) {
      return json(
        {
          error:
            error instanceof HttpError
              ? error.message
              : 'This service could not complete the request. Please try again shortly.',
        },
        error instanceof HttpError ? error.status : 503,
      );
    }
  };
}
export async function readText(request: Request, max = 12000) {
  if (Number(request.headers.get('content-length')) > max)
    throw new HttpError(413, 'The request is too large.');
  const reader = request.body?.getReader();
  if (!reader) return '';
  let length = 0;
  const chunks: Uint8Array[] = [];
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > max) {
        await reader.cancel();
        throw new HttpError(413, 'The request is too large.');
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return new TextDecoder().decode(bytes);
}
export async function body(request: Request, max = 12000) {
  if (
    !request.headers
      .get('content-type')
      ?.toLowerCase()
      .startsWith('application/json')
  )
    throw new HttpError(415, 'Send a JSON request.');
  const raw = await readText(request, max);
  try {
    const value: unknown = JSON.parse(raw);
    if (!record(value)) throw Error();
    return value;
  } catch {
    throw new HttpError(400, 'The request could not be read.');
  }
}
export function local(request: Request, env: PlatformEnv) {
  return (
    env.EIDOS_LOCAL_TEST === 'true' &&
    ['localhost', '127.0.0.1', '[::1]', 'terminal.local'].includes(
      new URL(request.url).hostname,
    )
  );
}
export function origin(request: Request) {
  if (request.headers.get('origin') !== new URL(request.url).origin)
    throw new HttpError(403, 'This request must come from the Eidos website.');
}
export function db(env: PlatformEnv) {
  if (!env.EIDOS_DB)
    throw new HttpError(
      503,
      'Community services are being prepared. Please contact the studio in the meantime.',
    );
  return env.EIDOS_DB;
}
export async function hash(text: string) {
  return [
    ...new Uint8Array(
      await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text)),
    ),
  ]
    .map((n) => n.toString(16).padStart(2, '0'))
    .join('');
}
export async function fingerprint(request: Request, env: PlatformEnv) {
  if ((!env.EIDOS_RATE_SECRET || env.EIDOS_RATE_SECRET.length < 32) && !local(request, env))
    throw new HttpError(
      503,
      'This service is being prepared. Please try again later.',
    );
  const key = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(env.EIDOS_RATE_SECRET || 'local-test-only'),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  return [
    ...new Uint8Array(
      await crypto.subtle.sign(
        'HMAC',
        key,
        new TextEncoder().encode(
          `${new Date().toISOString().slice(0, 10)}:${request.headers.get('CF-Connecting-IP') || 'unknown'}`,
        ),
      ),
    ),
  ]
    .map((n) => n.toString(16).padStart(2, '0'))
    .join('');
}
/** A single atomic SQLite statement prevents concurrent requests from overspending. Reservations never get refunded. */
export async function reserve(
  database: Database,
  bucket: string,
  units: number,
  limit: number,
  seconds = 86400,
) {
  if (units > limit || limit < 1) return false;
  const period = Math.floor(Date.now() / 1000 / seconds);
  const expires = (period + 2) * seconds;
  const result = await database
    .prepare(
      'INSERT INTO eidos_quotas (bucket,period,used,expires) VALUES (?,?,?,?) ON CONFLICT(bucket,period) DO UPDATE SET used=used+excluded.used WHERE used+excluded.used<=? RETURNING used',
    )
    .bind(bucket, period, units, expires, limit)
    .first();
  return Boolean(result);
}
export async function admin(request: Request, env: PlatformEnv) {
  if (!env.EIDOS_ADMIN_TOKEN || env.EIDOS_ADMIN_TOKEN.length < 32)
    throw new HttpError(503, 'Operator access is not configured.');
  const token = (request.headers.get('authorization') || '').replace(
    /^Bearer /,
    '',
  );
  if ((await hash(token)) !== (await hash(env.EIDOS_ADMIN_TOKEN)))
    throw new HttpError(401, 'Operator authentication is required.');
}
export async function challenge(
  request: Request,
  env: PlatformEnv,
  token: unknown,
  action: string,
) {
  if (local(request, env)) return;
  if (!env.TURNSTILE_SECRET_KEY)
    throw new HttpError(
      503,
      'Posting is being prepared. Please contact the studio for now.',
    );
  if (typeof token !== 'string' || !token || token.length > 2048)
    throw new HttpError(400, 'Please complete the verification and try again.');
  const response = await fetch(
    'https://challenges.cloudflare.com/turnstile/v0/siteverify',
    {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        secret: env.TURNSTILE_SECRET_KEY,
        response: token,
        remoteip: request.headers.get('CF-Connecting-IP') || undefined,
      }),
      signal: AbortSignal.timeout(8000),
    },
  );
  const result = (await response.json()) as {
    success?: boolean;
    hostname?: string;
    action?: string;
  };
  if (
    !response.ok ||
    !result.success ||
    result.hostname !== new URL(request.url).hostname ||
    result.action !== action
  )
    throw new HttpError(
      400,
      'Verification expired or failed. Please verify again.',
    );
}
export function siteOrigin(env: PlatformEnv) {
  try {
    const u = new URL(env.PUBLIC_SITE_URL || 'https://eidos-works.com');
    if (
      u.protocol === 'https:' ||
      ['localhost', '127.0.0.1'].includes(u.hostname)
    )
      return u.origin;
  } catch {
    /* use canonical */
  }
  return 'https://eidos-works.com';
}
export function escapeHtml(value: unknown) {
  return String(value ?? '').replace(
    /[&<>"']/g,
    (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[
        c
      ]!,
  );
}

export function communityAvailable(request:Request,env:PlatformEnv){
 const localMode=local(request,env);
 return Boolean(env.EIDOS_DB && env.EIDOS_ADMIN_TOKEN && env.EIDOS_ADMIN_TOKEN.length>=32 && ((env.EIDOS_RATE_SECRET && env.EIDOS_RATE_SECRET.length>=32)||localMode) && ((env.TURNSTILE_SITE_KEY&&env.TURNSTILE_SECRET_KEY)||localMode));
}
export function requireCommunity(request:Request,env:PlatformEnv){if(!communityAvailable(request,env))throw new HttpError(503,'Community posting is being prepared. Please contact the studio for now.');}
