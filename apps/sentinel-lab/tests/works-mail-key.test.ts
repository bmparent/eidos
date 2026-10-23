import test from 'node:test';
import assert from 'node:assert/strict';
import { platformEnvironment } from '../lib/works/dispatch';
import { checkPlaygroundProvider } from '../lib/works/playground-provider-check';

const source = { VERCEL_ENV: 'preview', EIDOS_PLAYGROUND_DELIVERY_MODE: 'test',
  PUBLIC_SITE_URL: 'https://eidosworks-playground-test-20261015.pages.dev', RESEND_playground_preview: 're_preview_fixture_only' };

test('owner mail alias reaches the platform only on the isolated TEST Preview', () => {
  assert.equal(platformEnvironment(source).RESEND_API_KEY, source.RESEND_playground_preview);
  assert.equal(platformEnvironment({ ...source, RESEND_API_KEY: '' }).RESEND_API_KEY, source.RESEND_playground_preview);
  for (const change of [{ VERCEL_ENV: 'production' }, { VERCEL_ENV: undefined },
    { EIDOS_PLAYGROUND_DELIVERY_MODE: 'live' }, { EIDOS_PLAYGROUND_DELIVERY_MODE: undefined },
    { PUBLIC_SITE_URL: 'https://eidos-works.com' }, { PUBLIC_SITE_URL: 'https://another-preview.pages.dev' }])
    assert.equal(platformEnvironment({ ...source, ...change }).RESEND_API_KEY, undefined);
});

test('canonical Resend binding retains priority and alias is not copied into PlatformEnv', () => {
  const env = platformEnvironment({ ...source, RESEND_API_KEY: 're_canonical_fixture_only' });
  assert.equal(env.RESEND_API_KEY, 're_canonical_fixture_only');
  assert.equal('RESEND_playground_preview' in env, false);
});

test('private diagnostic reports effective mail presence without revealing the alias value', async () => {
  const token = 'owner-private-fixture-32-characters';
  const request = new Request('https://preview.test/api/playground-provider-check', { headers: {
    'x-eidos-platform-token': token, 'x-eidos-site-origin': 'https://eidosworks-playground-20261014.pages.dev',
    'x-eidos-bindings-only': 'true' } });
  let calls = 0;
  const forbidden: typeof fetch = async () => { calls++; throw Error('No network expected'); };
  const response = await checkPlaygroundProvider(request, { ...source, EIDOS_PLATFORM_TOKEN: token,
    EIDOS_PLAYGROUND_PROVIDER_DIAGNOSTICS: 'test-auth' }, forbidden);
  const body = await response.text();
  assert.equal(response.status, 200); assert.equal(JSON.parse(body).bindings.mailKeyPresent, true);
  assert.equal(calls, 0); assert.equal(body.includes(source.RESEND_playground_preview), false);
});
