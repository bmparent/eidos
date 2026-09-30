import test from 'node:test';
import assert from 'node:assert/strict';
import {
  capturePinnedPublicHtml,
  normalizeCaptureUrl,
  publicIpv4,
} from '../lib/works/snapshotCapture';
import { POST } from '../app/api/works/snapshot-capture/route';

test('private and special-use addresses cannot be capture targets', () => {
  for (const address of ['127.0.0.1', '10.4.5.6', '169.254.169.254',
    '172.16.0.2', '192.168.1.1', '100.64.2.3', '203.0.113.4', '::1']) {
    assert.equal(publicIpv4(address), false, address);
  }
  assert.equal(publicIpv4('8.8.8.8'), true);
  for (const url of ['http://localhost/', 'http://127.0.0.1/',
    'https://metadata.google.internal/', 'http://public.example:8080/']) {
    assert.throws(() => normalizeCaptureUrl(url), url);
  }
});

test('the validated address is the one used for the connection', async () => {
  const calls: Array<{ host: string; address: string }> = [];
  const result = await capturePinnedPublicHtml('https://example.com/start', {
    resolve: async (host) => {
      assert.equal(host, 'example.com');
      return ['8.8.8.8'];
    },
    request: async (url, address) => {
      calls.push({ host: url.hostname, address });
      return { status: 200, contentType: 'text/html; charset=utf-8', html: '<h1>Example</h1>' };
    },
  });
  assert.deepEqual(calls, [{ host: 'example.com', address: '8.8.8.8' }]);
  assert.equal(result.finalUrl, 'https://example.com/start');
});

test('a redirect is independently resolved and private DNS is denied', async () => {
  const calls: string[] = [];
  await assert.rejects(capturePinnedPublicHtml('https://example.com/', {
    resolve: async (host) => host === 'example.com' ? ['8.8.8.8'] : ['127.0.0.1'],
    request: async (url, address) => {
      calls.push(url.hostname + '/' + address);
      return { status: 302, location: 'http://rebind.example/path' };
    },
  }), /dns-not-public/);
  assert.deepEqual(calls, ['example.com/8.8.8.8']);
});

test('mixed public and private DNS answers fail closed', async () => {
  let connected = false;
  await assert.rejects(capturePinnedPublicHtml('https://example.com/', {
    resolve: async () => ['8.8.8.8', '10.0.0.2'],
    request: async () => {
      connected = true;
      return { status: 200, contentType: 'text/html', html: 'bad' };
    },
  }), /dns-not-public/);
  assert.equal(connected, false);
});

test('capture route rejects missing authorization before parsing a URL', async () => {
  const original = process.env.EIDOS_SNAPSHOT_CAPTURE_TOKEN;
  process.env.EIDOS_SNAPSHOT_CAPTURE_TOKEN = 'fixture-token-with-at-least-thirty-two-characters';
  try {
    const response = await POST(new Request('https://example.test/api/works/snapshot-capture', {
      method: 'POST',
      body: JSON.stringify({ websiteUrl: 'https://example.com/' }),
    }));
    assert.equal(response.status, 403);
  } finally {
    if (original === undefined) delete process.env.EIDOS_SNAPSHOT_CAPTURE_TOKEN;
    else process.env.EIDOS_SNAPSHOT_CAPTURE_TOKEN = original;
  }
});

test('authorized route rejects an oversized streamed body before URL handling', async () => {
  const original = process.env.EIDOS_SNAPSHOT_CAPTURE_TOKEN;
  const token = 'fixture-token-with-at-least-thirty-two-characters';
  process.env.EIDOS_SNAPSHOT_CAPTURE_TOKEN = token;
  try {
    const response = await POST(new Request('https://example.test/api/works/snapshot-capture', {
      method: 'POST',
      headers: { authorization: 'Bearer ' + token },
      body: new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('x'.repeat(5000)));
          controller.close();
        },
      }),
      duplex: 'half',
    } as RequestInit));
    assert.equal(response.status, 413);
  } finally {
    if (original === undefined) delete process.env.EIDOS_SNAPSHOT_CAPTURE_TOKEN;
    else process.env.EIDOS_SNAPSHOT_CAPTURE_TOKEN = original;
  }
});
