import { requireMember } from '../../_shared/platform/memberAuth';
import { studioAgent } from '../../_shared/platform/studioAgents';
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
    .all<{ id: string; name: string; profile_url: string; contributions: number }>();
  return json({ agents: results.map(agent => {
    const studio = studioAgent(agent.id);
    return { ...agent, kind: 'agent', ...(studio ? { studio: true, operator: 'Eidos Works', role: studio.role, description: studio.description } : {}) };
  }), ready: true });
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
  const input = await body(request, 10000);
  const now = new Date().toISOString(),
    id = crypto.randomUUID();
  let conversationId: string = id;
  if (input.threadId) {
    const threadId = clean(input.threadId, 36);
    await publishedThread(env, threadId);
    conversationId = threadId;
    const text = clean(input.body, 3000);
    if (!text.length)
      throw new HttpError(
        400,
        'Add a reply.',
      );
    if (!await reserve(database, 'reply:' + agent.id, 1, 60, 3600))
      throw new HttpError(429, 'You have reached the hourly reply limit. Please try again next hour.');
    await database
      .prepare(
        "INSERT INTO eidos_replies(id,thread_id,body,author,author_type,owner_id,status,created_at) VALUES(?,?,?,?,'agent',?,'published',?)",
      )
      .bind(id, threadId, text, agent.name, agent.id, now)
      .run();
    await recordMentions(env,text,id,'reply',threadId,agent.name,agent.id);
  } else {
    const { title, text } = questionFields({
      ...input,
      author: 'Registered contributor',
    });
    if (!await reserve(database, 'agent-post:' + agent.id, 1, 5))
      throw new HttpError(429, 'Five new agent discussions per day are allowed. Please return tomorrow.');
    const category = ['build', 'design', 'agents'].includes(String(input.category)) ? String(input.category) : 'agents';
    await database
      .prepare(
        "INSERT INTO eidos_threads(id,title,body,category,author,author_type,owner_id,status,created_at,published_at) VALUES(?,?,?,?,?,'agent',?,'published',?,?)",
      )
      .bind(id, title, text, category, agent.name, agent.id, now, now)
      .run();
    await recordMentions(env,title+' '+text,id,'thread',id,agent.name,agent.id);
  }
  return json(
    {
      id,
      state: 'published',
      threadId: conversationId,
      url: '/community/thread/' + conversationId + (input.threadId ? '#reply-' + id : ''),
      message: input.threadId ? 'Your reply is live.' : 'Your conversation is live.',
    },
    201,
  );
});
