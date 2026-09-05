import {
  admin,
  body,
  clean,
  db,
  guarded,
  hash,
  HttpError,
  json,
  origin,
} from '../../_shared/platform/core';
import { maybeSuggest, type Reply } from '../../_shared/platform/community';
export const onRequestGet = guarded(async ({ request, env }) => {
  await admin(request, env);
  const database = db(env);
  const threads = await database
    .prepare(
      "SELECT * FROM eidos_threads WHERE status='pending' ORDER BY created_at LIMIT 100",
    )
    .all();
  const replies = await database
    .prepare(
      "SELECT * FROM eidos_replies WHERE status='pending' ORDER BY created_at LIMIT 100",
    )
    .all();
  const agents = await database
    .prepare(
      'SELECT id,name,profile_url,revoked,created_at FROM eidos_agents ORDER BY created_at DESC LIMIT 100',
    )
    .all();
  const quotas = await database
    .prepare(
      "SELECT bucket,used,period FROM eidos_quotas WHERE bucket IN ('ai-global','public-eidos-suggestions') AND period=?",
    )
    .bind(Math.floor(Date.now() / 86400000))
    .all();
  return json({
    threads: threads.results,
    replies: replies.results,
    agents: agents.results,
    quotas: quotas.results,
  });
});
export const onRequestPost = guarded(async ({ request, env }) => {
  origin(request);
  await admin(request, env);
  const input = await body(request);
  const database = db(env);
  const action = clean(input.action, 30),
    id = clean(input.id, 36);
  if (action === 'register-agent') {
    const name = clean(input.name, 50),
      profile = clean(input.profileUrl, 250);
    let url: URL;
    try {
      url = new URL(profile);
    } catch {
      throw new HttpError(400, 'Add the operator’s public HTTPS profile URL.');
    }
    if (
      name.length < 2 ||
      url.protocol !== 'https:' ||
      url.username ||
      url.password
    )
      throw new HttpError(400, 'Add a valid agent name and operator profile.');
    const key = 'eidos_' + crypto.randomUUID() + crypto.randomUUID();
    const agentId = crypto.randomUUID();
    await database
      .prepare(
        'INSERT INTO eidos_agents(id,name,key_hash,profile_url,created_at) VALUES(?,?,?,?,?)',
      )
      .bind(
        agentId,
        name,
        await hash(key),
        url.toString(),
        new Date().toISOString(),
      )
      .run();
    return json(
      {
        id: agentId,
        key,
        message: 'Copy this key now. It will not be shown again.',
      },
      201,
    );
  }
  if (action === 'revoke-agent') {
    await database
      .prepare('UPDATE eidos_agents SET revoked=1 WHERE id=?')
      .bind(id)
      .run();
    return json({ ok: true });
  }
  if (
    !['publish', 'reject', 'unpublish'].includes(action) ||
    !['thread', 'reply'].includes(String(input.kind))
  )
    throw new HttpError(400, 'Choose a valid moderation action.');
  const status = action === 'publish' ? 'published' : 'rejected';
  if (input.kind === 'thread') {
    const result = await database
      .prepare(
        "UPDATE eidos_threads SET status=?,published_at=CASE WHEN ?='published' THEN COALESCE(published_at,?) ELSE published_at END WHERE id=?",
      )
      .bind(status, status, new Date().toISOString(), id)
      .run();
    if (!result.meta.changes) throw new HttpError(404, 'Question not found.');
    if (status === 'published') await maybeSuggest(env, id);
  } else {
    const reply = await database
      .prepare('SELECT * FROM eidos_replies WHERE id=?')
      .bind(id)
      .first<Reply>();
    if (!reply) throw new HttpError(404, 'Reply not found.');
    await database
      .prepare('UPDATE eidos_replies SET status=? WHERE id=?')
      .bind(status, id)
      .run();
    if (
      status === 'published' &&
      reply.author_type === 'guest' &&
      /@eidos\b/i.test(reply.body)
    ) {
      await database
        .prepare('UPDATE eidos_threads SET request_assistant=1 WHERE id=?')
        .bind(reply.thread_id)
        .run();
      await maybeSuggest(env, reply.thread_id);
    }
  }
  return json({ ok: true });
});
