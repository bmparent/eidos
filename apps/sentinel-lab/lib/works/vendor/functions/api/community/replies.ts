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
  const text = clean(input.body, 3000),
    author = clean(input.author, 50),
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
  await publishedThread(env, id);
  await challenge(request, env, input.challenge, 'community');
  const replyId = crypto.randomUUID();
  await database
    .prepare(
      "INSERT INTO eidos_replies(id,thread_id,body,author,author_type,status,created_at) VALUES(?,?,?,?,'guest','pending',?)",
    )
    .bind(replyId, id, text, author, new Date().toISOString())
    .run();
  return json(
    {
      id: replyId,
      state: 'pending',
      message: 'Your reply has been saved for review.',
    },
    201,
  );
});
