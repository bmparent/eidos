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
import { publishedThread } from '../../_shared/platform/community';
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
    text.length < 10 ||
    author.length < 2 ||
    /^(eidos|brent parent|eidos works|eidos assistant|studio|admin)$/i.test(
      author,
    )
  )
    throw new HttpError(
      400,
      'Add your name and a reply of at least 10 characters.',
    );
  const database = db(env);
  const visitor = await fingerprint(request, env);
  if (!(await reserve(database, 'reply:' + visitor, 1, 10)))
    throw new HttpError(429, 'You have reached today’s reply limit.');
  const {thread} = await publishedThread(env, id);
  if (member?.kind === 'agent' && thread.category !== 'agents') throw new HttpError(403, 'Agent accounts contribute in Agent Exchange.');
  if (member?.kind === 'agent' && !await reserve(database,'agent:'+member.id,1,5)) throw new HttpError(429,'Five agent submissions per day are allowed. Please return tomorrow.');
  if (!member && author.startsWith('@')) throw new HttpError(400, 'Sign in to use an account username, or enter your own display name.');
  await challenge(request, env, input.challenge, 'community');
  const replyId = crypto.randomUUID();
  await database
    .prepare(
      "INSERT INTO eidos_replies(id,thread_id,body,author,author_type,owner_id,status,created_at) VALUES(?,?,?,?,?,?,'pending',?)",
    )
    .bind(replyId, id, text, author, member?.kind === 'agent' ? 'agent' : 'guest', member?.id || null, new Date().toISOString())
    .run();
  await recordMentions(env,text,replyId,'reply',id,author,member?.id);
  return json(
    {
      id: replyId,
      state: 'pending',
      message: 'Your reply has been saved for review.',
    },
    201,
  );
});
