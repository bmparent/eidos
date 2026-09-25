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

export async function POST(request: Request) {
  if (!authorized(request)) return new Response(null, { status: 403 });
  if (Number(request.headers.get('content-length') ?? 0) > 4096) {
    return new Response(null, { status: 413 });
  }
  try {
    const raw = await request.text();
    if (raw.length > 4096) return new Response(null, { status: 413 });
    const body = JSON.parse(raw) as { websiteUrl?: unknown };
    if (typeof body.websiteUrl !== 'string') return new Response(null, { status: 400 });
    const result = await capturePinnedPublicHtml(body.websiteUrl);
    return Response.json(result, { headers: { 'cache-control': 'no-store' } });
  } catch {
    return Response.json({ error: 'capture-unavailable' }, {
      status: 422,
      headers: { 'cache-control': 'no-store' },
    });
  }
}
