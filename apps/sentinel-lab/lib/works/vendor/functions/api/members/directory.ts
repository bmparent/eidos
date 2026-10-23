import { db, guarded, HttpError, json } from '../../_shared/platform/core';
import { accountsReady } from '../../_shared/platform/memberAuth';
export const onRequestGet = guarded(async ({ request, env }) => {
  const params = new URL(request.url).searchParams;
  // Public reading does not depend on email delivery or accepting new accounts.
  // Look up the exact identity; never attach activity by a display-name match.
  if (params.has('username')) {
    const username = (params.get('username') || '').toLowerCase();
    if (!/^[a-z][a-z0-9_]{2,23}$/.test(username))
      throw new HttpError(404, 'This public profile was not found.');
    const page = (name: string) => {
      const value = params.get(name) || '0';
      if (!/^(0|[1-9][0-9]{0,2})$/.test(value) || Number(value) > 500)
        throw new HttpError(400, 'Choose a valid activity page.');
      return Number(value);
    };
    const postsPage = page('postsPage'), repliesPage = page('repliesPage');
    const database = db(env);
    const member = await database.prepare(
      'SELECT id,username,kind,created_at FROM eidos_email_members WHERE username=? AND disabled=0',
    ).bind(username).first<{ id: string; username: string; kind: string; created_at: string }>();
    if (!member) throw new HttpError(404, 'This public profile was not found.');
    const [posts, replies] = await Promise.all([
      database.prepare(
        "SELECT t.id,t.title,substr(t.body,1,400) AS excerpt,t.category,t.created_at,(SELECT COUNT(*) FROM eidos_replies r WHERE r.thread_id=t.id AND r.status='published') AS reply_count FROM eidos_threads t WHERE t.owner_id=? AND t.status='published' ORDER BY t.created_at DESC,t.id DESC LIMIT 21 OFFSET ?",
      ).bind(member.id, postsPage * 20).all(),
      database.prepare(
        "SELECT r.id,r.thread_id,t.title AS thread_title,substr(r.body,1,400) AS excerpt,r.created_at FROM eidos_replies r JOIN eidos_threads t ON t.id=r.thread_id AND t.status='published' WHERE r.owner_id=? AND r.status='published' ORDER BY r.created_at DESC,r.id DESC LIMIT 21 OFFSET ?",
      ).bind(member.id, repliesPage * 20).all(),
    ]);
    return json({
      member: { username: member.username, kind: member.kind, created_at: member.created_at },
      posts: posts.results.slice(0, 20), replies: replies.results.slice(0, 20),
      postsPage, repliesPage,
      morePosts: posts.results.length > 20, moreReplies: replies.results.length > 20,
    });
  }
  if (!accountsReady(request, env)) return json({ members: [] });
  const q = (params.get('q') || '').toLowerCase();
  if (!/^[a-z][a-z0-9_]{1,23}$/.test(q))
    throw new HttpError(400, 'Enter at least two username characters.');
  const { results } = await db(env)
    .prepare(
      'SELECT username,kind,created_at FROM eidos_email_members WHERE disabled=0 AND substr(username,1,?)=? ORDER BY username LIMIT 8',
    )
    .bind(q.length, q)
    .all();
  return json({ members: results });
});
