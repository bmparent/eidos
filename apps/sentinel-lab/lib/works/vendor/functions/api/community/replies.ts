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
import { maybeSuggest, publishedThread } from '../../_shared/platform/community';
export const onRequestPost = guarded(async ({ request, env }) => {
  origin(request);
  requireCommunity(request,env);
  const input = await body(request);
  if (clean(input.website)) throw new HttpError(400, 'Please try again.');
  const member = await memberFromRequest(request,env,false);
  const text = clean(input.body, 3000),
    author = member ? '@'+member.username : clean(input.author, 50),
    id = clean(input.threadId, 36);
  if (
    !text.length ||
    author.length < 2 ||
    /^(eidos|brent parent|eidos works|eidos assistant|studio|admin)$/i.test(
      author,
    )
  )
    throw new HttpError(
      400,
      'Add your name and a reply.',
    );
  const database = db(env);
  const visitor = member?.id || await fingerprint(request, env);
  if (!(await reserve(database, 'reply:' + visitor, 1, member ? 60 : 30, 3600)))
    throw new HttpError(429, 'You have reached the hourly reply limit. Please try again next hour.');
  const {thread} = await publishedThread(env, id);
  if (!member && author.startsWith('@')) throw new HttpError(400, 'Sign in to use an account username, or enter your own display name.');
  if (!member) await challenge(request, env, input.challenge, 'community');
  const replyId = crypto.randomUUID();
  const now = new Date().toISOString();
  await database
    .prepare(
      "INSERT INTO eidos_replies(id,thread_id,body,author,author_type,owner_id,status,created_at) VALUES(?,?,?,?,?,?,'published',?)",
    )
    .bind(replyId, id, text, author, member?.kind === 'agent' ? 'agent' : 'guest', member?.id || null, now)
    .run();
  await recordMentions(env,text,replyId,'reply',id,author,member?.id);
  if (member?.kind !== 'agent' && thread.author_type !== 'agent' && /@eidos\b/i.test(text)) {
    await database.prepare('UPDATE eidos_threads SET request_assistant=1 WHERE id=?').bind(id).run();
    await maybeSuggest(env, id).catch(() => false);
  }
  return json(
    {
      id: replyId,
      state: 'published',
      url: '/community/thread/' + id + '#reply-' + replyId,
      reply: { id: replyId, thread_id: id, body: text, author, author_type: member?.kind === 'agent' ? 'agent' : 'guest', status: 'published', created_at: now },
      message: 'Your reply is live.',
    },
    201,
  );
});
