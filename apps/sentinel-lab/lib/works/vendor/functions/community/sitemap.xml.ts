import {
  db,
  escapeHtml,
  siteOrigin,
  type Context,
} from '../_shared/platform/core';
export async function onRequestGet({ env }: Context) {
  try {
    const threads = env.EIDOS_DB
      ? (
          await db(env)
            .prepare(
              "SELECT id,published_at FROM eidos_threads WHERE status='published' ORDER BY published_at DESC LIMIT 1000",
            )
            .all<{ id: string; published_at: string }>()
        ).results
      : [];
    return new Response(
      '<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' +
        threads
          .map(
            (t) =>
              `<url><loc>${escapeHtml(siteOrigin(env) + '/community/thread/' + t.id)}</loc><lastmod>${escapeHtml(t.published_at.slice(0, 10))}</lastmod></url>`,
          )
          .join('') +
        '</urlset>',
      {
        headers: {
          'content-type': 'application/xml',
          'cache-control': 'public, max-age=60',
        },
      },
    );
  } catch {
    return new Response('Temporarily unavailable', { status: 503 });
  }
}
