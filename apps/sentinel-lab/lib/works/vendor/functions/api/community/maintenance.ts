import { admin, db, guarded, json, hash } from '../../_shared/platform/core';
import { maybeSuggest } from '../../_shared/platform/community';
export const onRequestPost = guarded(async ({ request, env }) => {
  const token = (request.headers.get('authorization') || '').replace(
    /^Bearer /,
    '',
  );
  if (
    !env.EIDOS_MAINTENANCE_TOKEN ||
    env.EIDOS_MAINTENANCE_TOKEN.length < 32 ||
    (await hash(token)) !== (await hash(env.EIDOS_MAINTENANCE_TOKEN))
  )
    await admin(request, env);
  const database = db(env);
  let suggestions = 0;
  if (env.EIDOS_PROACTIVE_ENABLED === 'true') {
    const { results } = await database
      .prepare(
        "SELECT id FROM eidos_threads t WHERE status='published' AND allow_assistant=1 AND author_type='guest' AND published_at<? AND NOT EXISTS(SELECT 1 FROM eidos_replies r WHERE r.thread_id=t.id AND r.status='published') ORDER BY published_at LIMIT 10",
      )
      .bind(new Date(Date.now() - 86400000).toISOString())
      .all<{ id: string }>();
    for (const thread of results)
      if (await maybeSuggest(env, thread.id, true)) suggestions++;
  }
  const cutoff = Math.floor(Date.now() / 1000);
  await database.batch([
    database.prepare('DELETE FROM eidos_quotas WHERE expires<?').bind(cutoff),
    database
      .prepare('DELETE FROM eidos_answer_cache WHERE expires<?')
      .bind(cutoff),
    database
      .prepare(
        "DELETE FROM eidos_replies WHERE status='rejected' AND created_at<?",
      )
      .bind(new Date(Date.now() - 30 * 86400000).toISOString()),
    database
      .prepare(
        "DELETE FROM eidos_threads WHERE status='rejected' AND created_at<?",
      )
      .bind(new Date(Date.now() - 30 * 86400000).toISOString()),
  ]);
  return json({ ok: true, suggestions });
});
