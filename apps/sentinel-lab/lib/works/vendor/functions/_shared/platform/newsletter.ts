import {
  db,
  escapeHtml as e,
  hash,
  HttpError,
  readText,
  record,
  siteOrigin,
  type PlatformEnv,
} from './core';
import { randomToken, type Member } from './memberAuth';
import { sendMemberMail, type Mail } from './memberMail';

type Publication = {
  slug: string;
  title: string;
  publishedAt: string;
  byline: string;
  excerpt: string;
  body: { heading: string; paragraphs: string[]; bullets?: string[] }[];
  sources: { title?: string; label?: string; url: string }[];
};
type Delivery = {
  id: string;
  member_id: string;
  through_date: string;
  payload: string;
  created_at: number;
};
function publication(value: unknown): Publication {
  if (
    !record(value) ||
    typeof value.slug !== 'string' ||
    !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(value.slug) ||
    typeof value.title !== 'string' ||
    typeof value.byline !== 'string' ||
    typeof value.excerpt !== 'string' ||
    typeof value.publishedAt !== 'string' ||
    !Number.isFinite(Date.parse(value.publishedAt)) ||
    !Array.isArray(value.body) ||
    !value.body.every(
      (s) =>
        record(s) &&
        typeof s.heading === 'string' &&
        Array.isArray(s.paragraphs) &&
        s.paragraphs.every((p) => typeof p === 'string') &&
        (s.bullets === undefined ||
          (Array.isArray(s.bullets) &&
            s.bullets.every((b) => typeof b === 'string'))),
    ) ||
    !Array.isArray(value.sources) ||
    !value.sources.every(
      (s) =>
        record(s) &&
        typeof s.url === 'string' &&
        (s.title === undefined || typeof s.title === 'string') &&
        (s.label === undefined || typeof s.label === 'string'),
    )
  )
    throw Error('Invalid publication feed');
  return {
    ...value,
    publishedAt: new Date(value.publishedAt).toISOString(),
  } as Publication;
}
export function publicationText(p: Publication, site: string) {
  const sections = p.body.map((s) =>
    [
      s.heading,
      ...s.paragraphs,
      ...(s.bullets || []).map((b) => '• ' + b),
    ].join('\n\n'),
  );
  return [
    p.title,
    `By ${p.byline} · ${p.publishedAt.slice(0, 10)}`,
    p.excerpt,
    ...sections,
    'Sources',
    ...p.sources.map((s) => `${s.title || s.label || 'Source'}: ${s.url}`),
    `${site}/insights/${p.slug}`,
  ].join('\n\n');
}
/** Operators may select the same static feed on their hosting origin when the
 * public site's bot protection challenges server requests. Customer links keep
 * using PUBLIC_SITE_URL. Never derive this URL from a request or follow redirects. */
function publicationFeedUrl(env: PlatformEnv) {
  if (!env.EIDOS_PUBLICATION_FEED_URL) return siteOrigin(env) + '/insights-feed.json';
  const url = new URL(env.EIDOS_PUBLICATION_FEED_URL);
  if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash || url.pathname !== '/insights-feed.json')
    throw Error('Invalid configured publication feed URL');
  return url.href;
}
/** The existing authenticated hourly maintenance job delivers yesterday's complete
 * articles, with a per-member daily identity, durable claims, and provider deduplication. */
export async function deliverNewsletters(env: PlatformEnv) {
  if (
    env.EIDOS_ACCOUNTS_ENABLED !== 'true' ||
    env.EIDOS_NEWSLETTER_ENABLED !== 'true' ||
    !env.RESEND_API_KEY ||
    !env.EIDOS_MAIL_FROM
  )
    return { sent: 0, enabled: false };
  const database = db(env),
    site = siteOrigin(env),
    now = Math.floor(Date.now() / 1000);
  const day = new Date().toISOString().slice(0, 10),
    cutoff = day + 'T00:00:00.000Z';
  const response = await fetch(publicationFeedUrl(env), {
    redirect: 'error',
    signal: AbortSignal.timeout(8000),
  });
  if (!response.ok) throw new HttpError(503, `Publication feed unavailable (HTTP ${response.status}).`);
  const feed = JSON.parse(await readText(response, 4_000_000)) as {
    items: Publication[];
  };
  if (!Array.isArray(feed.items) || feed.items.length > 1000)
    throw Error('Invalid publication feed');
  const items = feed.items
    .map(publication)
    .filter((p) => Date.parse(p.publishedAt) < Date.parse(cutoff))
    .sort((a, b) => Date.parse(a.publishedAt) - Date.parse(b.publishedAt));
  // If a provider outcome is uncertain beyond its 24-hour dedupe window, stop.
  // Never automatically replay an old email under a fresh idempotency key.
  await database
    .prepare(
      "UPDATE eidos_mail_deliveries SET status='uncertain' WHERE status='pending' AND created_at<?",
    )
    .bind(now - 23 * 3600)
    .run();
  const { results: members } = await database
    .prepare(
      `SELECT m.* FROM eidos_email_members m WHERE newsletter=1 AND disabled=0 AND newsletter_after<?
    AND NOT EXISTS(SELECT 1 FROM eidos_mail_deliveries d WHERE d.member_id=m.id AND (d.id='digest:'||m.id||':'||? OR d.status IN ('pending','uncertain')))
    ORDER BY newsletter_after LIMIT 10`,
    )
    .bind(items.at(-1)?.publishedAt || '', day)
    .all<Member>();
  for (const m of members) {
    const fresh = items.filter(
      (p) => Date.parse(p.publishedAt) > Date.parse(m.newsletter_after),
    );
    if (!fresh.length) continue;
    const token = randomToken(),
      unsubscribe = `${site}/account/unsubscribe#token=${token}`;
    const full = fresh
      .map((p) => publicationText(p, site))
      .join('\n\n────────────────────\n\n');
    const mail: Mail = {
      to: m.email,
      subject: `Eidos Works · ${fresh.length} new ${fresh.length === 1 ? 'paper' : 'papers'}`,
      text: `Your free Eidos Works reading\n\n${full}\n\nYou chose to receive new white papers and posts.\nUnsubscribe: ${unsubscribe}\nManage your account: ${site}/account`,
      html: `<!doctype html><html lang="en"><body style="margin:0;background:#edf3f5;color:#183342;font:16px/1.7 Arial,sans-serif"><main style="max-width:700px;margin:0 auto;padding:32px"><p style="letter-spacing:3px">EIDOS WORKS / INSIGHTS</p>${fresh.map((p) => `<article style="background:#fff;padding:26px;margin:20px 0;border-radius:12px"><h1 style="line-height:1.2">${e(p.title)}</h1><p>${e(p.byline)} · ${e(p.publishedAt.slice(0, 10))}</p><p>${e(p.excerpt)}</p>${p.body.map((s) => `<h2>${e(s.heading)}</h2>${s.paragraphs.map((t) => `<p>${e(t)}</p>`).join('')}${s.bullets?.length ? `<ul>${s.bullets.map((b) => `<li>${e(b)}</li>`).join('')}</ul>` : ''}`).join('')}<h2>Sources</h2><ul>${p.sources.map((s) => `<li>${e(s.title || s.label || 'Source')}: ${e(s.url)}</li>`).join('')}</ul><p><a href="${site}/insights/${p.slug}">Read on Eidos Works</a></p></article>`).join('')}<p>You chose to receive new white papers and posts for free.</p><p><a href="${unsubscribe}">Unsubscribe</a> · <a href="${site}/account">Your account</a></p></main></body></html>`,
    };
    const id = `digest:${m.id}:${day}`;
    await database.batch([
      database
        .prepare(
          'INSERT INTO eidos_unsubscribe_tokens(token_hash,member_id) VALUES(?,?)',
        )
        .bind(await hash(token), m.id),
      database
        .prepare(
          "INSERT OR IGNORE INTO eidos_mail_deliveries(id,member_id,through_date,payload,created_at,status) VALUES(?,?,?,?,?,'pending')",
        )
        .bind(id, m.id, fresh.at(-1)!.publishedAt, JSON.stringify(mail), now),
    ]);
  }
  const { results: pending } = await database
    .prepare(
      "SELECT d.* FROM eidos_mail_deliveries d JOIN eidos_email_members m ON m.id=d.member_id WHERE d.status='pending' AND d.claimed_until<? AND m.newsletter=1 AND m.disabled=0 ORDER BY d.created_at LIMIT 10",
    )
    .bind(now)
    .all<Delivery>();
  let sent = 0;
  for (const d of pending) {
    const claim = await database
      .prepare(
        "UPDATE eidos_mail_deliveries SET claimed_until=? WHERE id=? AND status='pending' AND claimed_until<? RETURNING id",
      )
      .bind(now + 120, d.id, now)
      .first();
    if (!claim) continue;
    const subscribed = await database
      .prepare(
        'SELECT id FROM eidos_email_members WHERE id=? AND newsletter=1 AND disabled=0',
      )
      .bind(d.member_id)
      .first();
    if (!subscribed) continue;
    try {
      const provider = await sendMemberMail(
        env,
        JSON.parse(d.payload) as Mail,
        d.id,
      );
      await database.batch([
        database
          .prepare(
            "UPDATE eidos_mail_deliveries SET status='sent',provider_id=?,payload='' WHERE id=?",
          )
          .bind(provider, d.id),
        database
          .prepare(
            'UPDATE eidos_email_members SET newsletter_after=MAX(newsletter_after,?) WHERE id=?',
          )
          .bind(d.through_date, d.member_id),
      ]);
      sent++;
    } catch {
      /* Retain the same payload/key for a later bounded attempt. */
    }
  }
  return { sent, enabled: true };
}
