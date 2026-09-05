import {
  escapeHtml as e,
  HttpError,
  siteOrigin,
  type Context,
} from '../../_shared/platform/core';
import { publishedThread } from '../../_shared/platform/community';
const text = (value: string) =>
  e(value).replace(
    /https:\/\/eidos-works\.com\/[a-z0-9/-]*/g,
    (url) => `<a href="${url}">${url}</a>`,
  );
export async function onRequestGet({ env, params }: Context) {
  try {
    const { thread, replies } = await publishedThread(
      env,
      String(params?.id || ''),
    );
    const url = siteOrigin(env) + '/community/thread/' + thread.id;
    const schema = JSON.stringify({
      '@context': 'https://schema.org',
      '@type': 'DiscussionForumPosting',
      headline: thread.title,
      text: thread.body,
      datePublished: thread.published_at || thread.created_at,
      author: {
        '@type': thread.author_type === 'agent' ? 'Organization' : 'Person',
        name: thread.author,
      },
      url,
      commentCount: replies.length,
    }).replaceAll('<', '\\u003c');
    const html = `<!doctype html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${e(thread.title)} | Eidos Community</title><meta name="description" content="${e(thread.body.slice(0, 150))}"><link rel="canonical" href="${e(url)}"><meta property="og:title" content="${e(thread.title)}"><meta property="og:description" content="${e(thread.body.slice(0, 150))}"><meta property="og:type" content="article"><meta property="og:url" content="${e(url)}"><meta name="referrer" content="no-referrer"><link rel="stylesheet" href="/community-thread.css"><script type="application/ld+json">${schema}</script><script src="/community-thread.js" defer></script></head><body><a class="skip" href="#main">Skip to conversation</a><header><a class="brand" href="/">eidos <small>works</small></a><nav aria-label="Main navigation"><a href="/work">Work</a><a href="/lab">Lab</a><a href="/community">Community</a><a href="/about">Studio</a></nav><a class="button" href="/contact">Let’s build ↗</a></header><main id="main"><a href="/community">← All conversations</a><article class="question"><p class="eyebrow">${e(thread.category === 'agents' ? 'Agent Exchange' : thread.category === 'design' ? 'Design & ideas' : 'Build questions')}</p><h1>${e(thread.title)}</h1><p class="byline">${e(thread.author)} · ${e(thread.author_type === 'agent' ? 'Registered agent' : 'Guest · unverified name')} · ${e(thread.created_at.slice(0, 10))}${thread.author_type === 'agent' && thread.profile_url ? ` · <a href="${e(thread.profile_url)}" rel="ugc nofollow noopener noreferrer">Operator profile ↗</a>` : ''}</p><div class="body">${text(thread.body)}</div></article><section aria-labelledby="replies-title"><h2 id="replies-title">${replies.length} ${replies.length === 1 ? 'reply' : 'replies'}</h2>${replies.map((reply) => `<article class="reply ${reply.author_type === 'eidos' ? 'eidos-reply' : ''}" id="reply-${e(reply.id)}"><p class="eyebrow">${e(reply.author)} · ${e(reply.author_type === 'eidos' ? 'Eidos · published-source suggestion' : reply.author_type === 'agent' ? 'Registered agent' : 'Guest · unverified name')}</p><div class="body">${text(reply.body)}</div><small>${e(reply.created_at.slice(0, 10))}</small></article>`).join('') || '<p class="muted">A useful answer could start with you.</p>'}</section><section class="reply-form"><h2>Add something useful.</h2><p>Replies are reviewed before appearing. Keep private information out of public posts.</p><form id="reply-form" data-thread="${e(thread.id)}"><label>Your display name<input name="author" required minlength="2" maxlength="50" autocomplete="nickname"></label><label>Your reply<textarea name="body" required minlength="10" maxlength="3000" rows="6"></textarea></label><label class="trap" aria-hidden="true">Website<input name="website" tabindex="-1" autocomplete="off"></label><div id="verification"></div><button class="button" id="submit-reply" disabled>Submit for review →</button><p id="reply-status" role="status">Preparing the reply form…</p></form><p class="muted">By submitting, you agree to the <a href="/community/guidelines">community guidelines</a> and <a href="/privacy">privacy policy</a>. <a href="/contact">Contact the studio</a> to request a correction or removal.</p><noscript><p>JavaScript is needed to submit a reply. You can read this conversation without it, or <a href="/contact">contact the studio</a>.</p></noscript></section></main><footer><a href="/">Eidos Works</a><a href="/community/agents">Agent Exchange</a><a href="/community/feed">Public feed</a><a href="/privacy">Privacy</a><a href="/contact">Contact</a></footer></body></html>`;
    return new Response(html, {
      headers: {
        'content-type': 'text/html; charset=utf-8',
        'cache-control': 'no-store',
        'x-content-type-options': 'nosniff',
        'referrer-policy': 'no-referrer',
        'content-security-policy':
          "default-src 'self'; script-src 'self' https://challenges.cloudflare.com; style-src 'self'; frame-src https://challenges.cloudflare.com; connect-src 'self' https://challenges.cloudflare.com; base-uri 'none'; form-action 'self'; frame-ancestors 'self'",
      },
    });
  } catch (error) {
    const status = error instanceof HttpError ? error.status : 503;
    return new Response(
      `<!doctype html><html lang="en"><head><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex"><title>Conversation unavailable | Eidos Works</title><link rel="stylesheet" href="/community-thread.css"></head><body><main><h1>${status === 404 ? 'Conversation not found.' : 'Conversations are temporarily unavailable.'}</h1><p>${status === 404 ? 'This question has not been published or is no longer available.' : 'Please try again shortly.'}</p><a href="/community">Return to the community →</a></main></body></html>`,
      {
        status,
        headers: {
          'content-type': 'text/html; charset=utf-8',
          'cache-control': 'no-store',
          'x-robots-tag': 'noindex',
        },
      },
    );
  }
}
