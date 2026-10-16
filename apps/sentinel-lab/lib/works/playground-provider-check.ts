import { timingSafeEqual } from 'node:crypto';

export const diagnosticReply = (body: unknown, status = 200) => Response.json(body, { status, headers: { 'cache-control': 'no-store' } });
export function authorizePlaygroundDiagnostic(request: Request, env: Record<string, string | undefined>, mode: 'test-auth' | 'test-count-one' | 'test-generation-one') {
  if (env.VERCEL_ENV !== 'preview' || env.EIDOS_PLAYGROUND_PROVIDER_DIAGNOSTICS !== mode) return diagnosticReply({ error: 'Not found.' }, 404);
  const expected = env.EIDOS_PLATFORM_TOKEN || '', actual = request.headers.get('x-eidos-platform-token') || '';
  if (expected.length < 32 || Buffer.byteLength(actual) !== Buffer.byteLength(expected) || !timingSafeEqual(Buffer.from(actual), Buffer.from(expected)))
    return diagnosticReply({ error: 'Owner diagnostic authentication required.' }, 401);
  if (request.headers.get('x-eidos-site-origin') !== 'https://eidosworks-playground-20261014.pages.dev') return diagnosticReply({ error: 'Unrecognized origin.' }, 403);
  return null;
}

/** Read-only owner diagnostic. No inference, counting, database writes or key readback. */
export async function checkPlaygroundProvider(request: Request, env: Record<string, string | undefined> = process.env, transport: typeof fetch = fetch) {
  const reply = diagnosticReply, rejected = authorizePlaygroundDiagnostic(request, env, 'test-auth');
  if (rejected) return rejected;
  const bindings = { accountsEnabled: env.EIDOS_ACCOUNTS_ENABLED === 'true', mailKeyPresent: Boolean(env.RESEND_API_KEY),
    mailFromPresent: Boolean(env.EIDOS_MAIL_FROM), rateSecretReady: (env.EIDOS_RATE_SECRET?.length || 0) >= 32,
    turnstileSitePresent: Boolean(env.TURNSTILE_SITE_KEY), turnstileSecretPresent: Boolean(env.TURNSTILE_SECRET_KEY),
    playgroundKeyPresent: (env.EIDOS_PLAYGROUND_OPENAI_API_KEY?.length || 0) >= 20,
    validationDatabaseSelected: env.EIDOS_DATABASE_BINDING_PREFIX === 'EIDOS_TEMPLATES_TEST' };
  if (request.headers.get('x-eidos-bindings-only') === 'true') return reply({ bindings, inferenceRequests: 0, countRequests: 0,
    sourceCommit: env.EIDOS_SOURCE_COMMIT || null, bundleHash: env.EIDOS_SOURCE_BUNDLE_HASH || null });
  const key = env.EIDOS_PLAYGROUND_OPENAI_API_KEY;
  if (!key || key.length < 20) return reply({ authenticated: false, reason: 'dedicated_key_missing', inferenceRequests: 0 }, 503);
  try {
    const response = await transport('https://api.openai.com/v1/models', { redirect: 'error', cache: 'no-store', signal: AbortSignal.timeout(15000), headers: { authorization: 'Bearer ' + key } });
    const requestId = response.headers.get('x-request-id');
    const providerRequestId = requestId && /^[!-~]{1,200}$/.test(requestId) ? requestId : null;
    if (!response.ok) return reply({ authenticated: false, providerStatus: response.status, providerRequestId, inferenceRequests: 0 }, 502);
    const text = await response.text();
    if (text.length > 500000) throw Error('oversized_metadata');
    const data = JSON.parse(text);
    if (!Array.isArray(data.data)) throw Error('invalid_metadata');
    const ids = new Set(data.data.map((entry: { id?: unknown }) => entry?.id));
    const projectHeader=response.headers.get('openai-project'), organizationHeader=response.headers.get('openai-organization');
    return reply({ authenticated: true, providerStatus: response.status, providerRequestId, bindings,
      providerProject: /^proj_[A-Za-z0-9_-]{1,100}$/.test(projectHeader || '') ? projectHeader : null,
      providerOrganization: /^org-[A-Za-z0-9_-]{1,100}$/.test(organizationHeader || '') ? organizationHeader : null,
      models: { 'gpt-6-astra': ids.has('gpt-6-astra'), 'gpt-6.1-sol': ids.has('gpt-6.1-sol') },
      inferenceRequests: 0, countRequests: 0, generationAccess: 'not_tested',
      sourceCommit: env.EIDOS_SOURCE_COMMIT || null, bundleHash: env.EIDOS_SOURCE_BUNDLE_HASH || null });
  } catch {
    return reply({ authenticated: false, reason: 'provider_metadata_unavailable', inferenceRequests: 0 }, 502);
  }
}
