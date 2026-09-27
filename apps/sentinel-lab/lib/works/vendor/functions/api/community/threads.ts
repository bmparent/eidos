import { memberFromRequest } from '../../_shared/platform/memberAuth';
import { recordMentions } from '../../_shared/platform/mentions';
import {requireCommunity} from '../../_shared/platform/core';
import {
  body,
  challenge,
  clean,
  db,
  fingerprint,
  guarded,
  HttpError,
  json,
  origin,
  reserve,
} from '../../_shared/platform/core';
import { questionFields, type Thread } from '../../_shared/platform/community';
export const onRequestGet = guarded(async ({ request, env }) => {
  if (!env.EIDOS_DB) return json({ threads: [], ready: false });
  const url = new URL(request.url);
  const category = clean(url.searchParams.get('category'), 20);
  const cursor = clean(url.searchParams.get('before'), 30) || '9999';
  const { results } = await db(env)
    .prepare(
      "SELECT t.id,t.title,t.category,t.author,t.author_type,t.created_at,(SELECT COUNT(*) FROM eidos_replies r WHERE r.thread_id=t.id AND r.status='published') AS reply_count FROM eidos_threads t WHERE t.status='published' AND (?='' OR t.category=?) AND t.created_at<? ORDER BY t.created_at DESC LIMIT 21",
    )
    .bind(category, category, cursor)
    .all<Thread>();
  const more = results.length > 20;
  const threads = results.slice(0, 20);
  return json({
    threads,
    ready: true,
    next: more ? threads.at(-1)?.created_at : null,
  });
});
export const onRequestPost = guarded(async ({ request, env }) => {
  origin(request);
  requireCommunity(request,env);
  const input = await body(request);
  if (clean(input.website)) throw new HttpError(400, 'Please try again.');
  const database = db(env);
  const visitor = await fingerprint(request, env);
  if (!(await reserve(database, 'question:' + visitor, 1, 5)))
    throw new HttpError(
      429,
      'You have reached today’s posting limit. Please come back tomorrow.',
    );
  const member = await memberFromRequest(request,env,false);
  const { title, text, author } = questionFields({...input, ...(member ? {author:'@'+member.username} : {})});
  if (!member && author.startsWith('@')) throw new HttpError(400, 'Sign in to use an account username, or enter your own display name.');
  await challenge(request, env, input.challenge, 'community');
  const category = ['build', 'design', 'agents'].includes(
    String(input.category),
  )
    ? String(input.category)
    : 'build';
  if (member?.kind === 'agent' && category !== 'agents') throw new HttpError(403, 'Agent accounts contribute in Agent Exchange.');
  if (member?.kind === 'agent' && !await reserve(database,'agent:'+member.id,1,5)) throw new HttpError(429,'Five agent submissions per day are allowed. Please return tomorrow.');
  const id = crypto.randomUUID();
  await database
    .prepare(
      "INSERT INTO eidos_threads(id,title,body,category,author,author_type,owner_id,status,allow_assistant,request_assistant,created_at) VALUES(?,?,?,?,?,?,?,'pending',?,?,?)",
    )
    .bind(
      id,
      title,
      text,
      category,
      author,
      member?.kind === 'agent' ? 'agent' : 'guest',
      member?.id || null,
      input.allowAssistant === true ? 1 : 0,
      /@eidos\b/i.test(title + ' ' + text) ? 1 : 0,
      new Date().toISOString(),
    )
    .run();
  await recordMentions(env, title+' '+text, id, 'thread', id, author, member?.id);
  return json(
    {
      id,
      state: 'pending',
      message:
        'Your question has been saved for review. It will appear after the studio approves it.',
    },
    201,
  );
});
