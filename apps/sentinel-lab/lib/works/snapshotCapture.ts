import { lookup } from 'node:dns/promises';
import * as http from 'node:http';
import * as https from 'node:https';
import { BlockList, isIP } from 'node:net';

const MAX_REDIRECTS = 3;
const MAX_HTML_BYTES = 90_000;
const REQUEST_TIMEOUT_MS = 8_000;
const blocked = new BlockList();
for (const [base, prefix] of [
  ['0.0.0.0', 8], ['10.0.0.0', 8], ['100.64.0.0', 10],
  ['127.0.0.0', 8], ['169.254.0.0', 16], ['172.16.0.0', 12],
  ['192.0.0.0', 24], ['192.0.2.0', 24], ['192.88.99.0', 24],
  ['192.168.0.0', 16], ['198.18.0.0', 15], ['198.51.100.0', 24],
  ['203.0.113.0', 24], ['224.0.0.0', 4], ['240.0.0.0', 4],
] as const) blocked.addSubnet(base, prefix, 'ipv4');

export function publicIpv4(address: string) {
  return isIP(address) === 4 && !blocked.check(address, 'ipv4');
}

export function normalizeCaptureUrl(raw: string) {
  if (raw.length > 2048) throw new Error('invalid-url');
  const url = new URL(raw);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) {
    throw new Error('invalid-url');
  }
  if (url.port && !((url.protocol === 'http:' && url.port === '80') ||
    (url.protocol === 'https:' && url.port === '443'))) throw new Error('invalid-port');
  const host = url.hostname.toLowerCase().replace(/\.$/, '');
  if (!host || isIP(host) || !host.includes('.') ||
    ['localhost', 'metadata.google.internal'].includes(host) ||
    ['.localhost', '.local', '.internal', '.home', '.lan', '.test', '.invalid', '.onion']
      .some((suffix) => host.endsWith(suffix))) throw new Error('invalid-host');
  url.hash = '';
  return url;
}

type PinnedResponse = {
  status: number;
  location?: string;
  contentType?: string;
  html?: string;
};

function requestPinned(url: URL, address: string): Promise<PinnedResponse> {
  return new Promise((resolve, reject) => {
    const transport = url.protocol === 'https:' ? https : http;
    const request = transport.request(url, {
      method: 'GET',
      agent: false,
      headers: {
        accept: 'text/html,application/xhtml+xml;q=0.9',
        'user-agent': 'EidosSnapshot/1.0 (+https://eidos-works.com/snapshot)',
      },
      lookup: (_hostname, _options, callback) => callback(null, address, 4),
      servername: url.protocol === 'https:' ? url.hostname : undefined,
      timeout: REQUEST_TIMEOUT_MS,
    }, (response) => {
      const status = response.statusCode ?? 0;
      const location = response.headers.location;
      const contentType = response.headers['content-type'] ?? '';
      if (status >= 300 && status < 400) {
        response.resume();
        resolve({ status, location });
        return;
      }
      if (status < 200 || status >= 300 ||
        !/^(text\/html|application\/xhtml\+xml)(?:\s*;|\s*$)/i.test(contentType)) {
        response.resume();
        reject(new Error('not-html'));
        return;
      }
      const declared = Number(response.headers['content-length'] ?? 0);
      if (Number.isFinite(declared) && declared > MAX_HTML_BYTES) {
        response.destroy();
        reject(new Error('content-too-large'));
        return;
      }
      const chunks: Buffer[] = [];
      let size = 0;
      response.on('data', (chunk: Buffer) => {
        size += chunk.byteLength;
        if (size > MAX_HTML_BYTES) {
          response.destroy(new Error('content-too-large'));
          return;
        }
        chunks.push(chunk);
      });
      response.on('error', reject);
      response.on('end', () => resolve({
        status,
        contentType,
        html: Buffer.concat(chunks).toString('utf8'),
      }));
    });
    request.on('timeout', () => request.destroy(new Error('capture-timeout')));
    request.on('error', reject);
    request.end();
  });
}

export async function capturePinnedPublicHtml(
  rawUrl: string,
  dependencies: {
    resolve?: (hostname: string) => Promise<string[]>;
    request?: (url: URL, pinnedAddress: string) => Promise<PinnedResponse>;
  } = {},
) {
  let url = normalizeCaptureUrl(rawUrl);
  const resolve = dependencies.resolve ?? (async (hostname: string) =>
    (await lookup(hostname, { all: true, family: 4 })).map((answer) => answer.address));
  const request = dependencies.request ?? requestPinned;
  for (let hop = 0; hop <= MAX_REDIRECTS; hop += 1) {
    const addresses = await resolve(url.hostname);
    if (!addresses.length || addresses.some((address) => !publicIpv4(address))) {
      throw new Error('dns-not-public');
    }
    const result = await request(url, addresses[0]);
    if (result.status >= 300 && result.status < 400) {
      if (!result.location || hop === MAX_REDIRECTS) throw new Error('redirect-failed');
      url = normalizeCaptureUrl(new URL(result.location, url).toString());
      continue;
    }
    if (result.status < 200 || result.status >= 300 ||
      typeof result.html !== 'string' || Buffer.byteLength(result.html) > MAX_HTML_BYTES ||
      !/^(text\/html|application\/xhtml\+xml)(?:\s*;|\s*$)/i.test(result.contentType ?? '')) {
      throw new Error('capture-invalid');
    }
    return { finalUrl: url.toString(), html: result.html };
  }
  throw new Error('redirect-failed');
}
