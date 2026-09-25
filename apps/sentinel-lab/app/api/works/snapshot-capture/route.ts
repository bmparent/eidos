import { timingSafeEqual } from 'node:crypto';
import { capturePinnedPublicHtml } from '@/lib/works/snapshotCapture';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 30;

function authorized(request: Request) {
  const expected = process.env.EIDOS_SNAPSHOT_CAPTURE_TOKEN;
  const provided = request.headers.get('authorization')?.replace(/^Bearer /, '');
  if (!expected || expected.length < 32 || !provided) return false;
  const left = Buffer.from(expected);
  const right = Buffer.from(provided);
  return left.length === right.length && timingSafeEqual(left, right);
}

async function readBoundedBody(request: Request) {
  if (!request.body) return '';
  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let bytes = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > 4096) throw new Error('body-too-large');
      chunks.push(value);
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
  const merged = new Uint8Array(bytes);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new TextDecoder().decode(merged);
}

export async function POST(request: Request) {
  if (!authorized(request)) return new Response(null, { status: 403 });
  if (Number(request.headers.get('content-length') ?? 0) > 4096) {
    return new Response(null, { status: 413 });
  }
  try {
    const raw = await readBoundedBody(request);
    const body = JSON.parse(raw) as { websiteUrl?: unknown };
    if (typeof body.websiteUrl !== 'string') return new Response(null, { status: 400 });
    const result = await capturePinnedPublicHtml(body.websiteUrl);
    return Response.json(result, { headers: { 'cache-control': 'no-store' } });
  } catch (error) {
    if (error instanceof Error && error.message === 'body-too-large') {
      return new Response(null, { status: 413 });
    }
    return Response.json({ error: 'capture-unavailable' }, {
      status: 422,
      headers: { 'cache-control': 'no-store' },
    });
  }
}
