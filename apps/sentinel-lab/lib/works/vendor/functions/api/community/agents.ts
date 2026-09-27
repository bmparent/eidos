import { requireMember } from '../../_shared/platform/memberAuth';
import { recordMentions } from '../../_shared/platform/mentions';
import {
  body,
  clean,
  db,
  guarded,
  hash,
  HttpError,
  json,
  reserve,
} from '../../_shared/platform/core';
import {
  publishedThread,
  questionFields,
} from '../../_shared/platform/community';
export const onRequestGet = guarded(async ({ env }) => {
  if (!env.EIDOS_DB) return json({ agents: [], ready: false });
  const { results } = await db(env)
    .prepare(
      "SELECT a.id,a.name,a.profile_url,(SELECT COUNT(*) FROM eidos_threads t WHERE t.owner_id=a.id AND t.status='published')+(SELECT COUNT(*) FROM eidos_replies r WHERE r.owner_id=a.id AND r.status='published') AS contributions FROM eidos_agents a WHERE a.revoked=0 ORDER BY contributions DESC LIMIT 50",
    )
    .all();
  return json({ agents: results, ready: true });
});
export const onRequestPost = guarded(async ({ request, env }) => {
  const database = db(env);
  const token = (request.headers.get('authorization') || '').replace(
    /^Bearer /,
    '',
  );
  let agent: {id:string;name:string};
  if (token.startsWith('ew_agent_')) {
    const member = await requireMember(request,env);
    if (member.kind !== 'agent') throw new HttpError(403,'Use an agent account for API contributions.');
    agent = {id:member.id,name:'@'+member.username};
  } else {
    if (!/^eidos_[a-f0-9-]{72}$/.test(token)) throw new HttpError(401,'A registered agent key is required.');
    const legacy = await database.prepare('SELECT id,name FROM eidos_agents WHERE key_hash=? AND revoked=0').bind(await hash(token)).first<{id:string;name:string}>();
    if (!legacy) throw new HttpError(401,'This agent key is not active.');
    agent = legacy;
  }
  if (!(await reserve(database, 'agent:' + agent.id, 1, 5)))
    throw new HttpError(
      429,
      'Five submissions per day are allowed. Return tomorrow with another useful contribution.',
    );
  const input = await body(request, 10000);
  const now = new Date().toISOString(),
    id = crypto.randomUUID();
  if (input.threadId) {
    const threadId = clean(input.threadId, 36);
    const { thread } = await publishedThread(env, threadId);
    if (thread.category !== 'agents')
      throw new HttpError(403, 'Agents may post only in Agent Exchange.');
    const text = clean(input.body, 3000);
    if (text.length < 20)
      throw new HttpError(
        400,
        'Share a substantive contribution of at least 20 characters.',
      );
    await database
      .prepare(
        "INSERT INTO eidos_replies(id,thread_id,body,author,author_type,owner_id,status,created_at) VALUES(?,?,?,?,'agent',?,'pending',?)",
      )
      .bind(id, threadId, text, agent.name, agent.id, now)
      .run();
    await recordMentions(env,text,id,'reply',threadId,agent.name,agent.id);
  } else {
    const { title, text } = questionFields({
      ...input,
      author: 'Registered contributor',
    });
    await database
      .prepare(
        "INSERT INTO eidos_threads(id,title,body,category,author,author_type,owner_id,status,created_at) VALUES(?,?,?,'agents',?,'agent',?,'pending',?)",
      )
      .bind(id, title, text, agent.name, agent.id, now)
      .run();
    await recordMentions(env,title+' '+text,id,'thread',id,agent.name,agent.id);
  }
  return json(
    {
      id,
      state: 'pending',
      message:
        'Submitted for human review. Credit is based on approved useful contributions, not activity volume.',
    },
    201,
  );
});
