import { clean, db, HttpError, reserve, type PlatformEnv } from './core';
import { sourceAnswer, knowledge } from './knowledge';
export interface Thread {
  id: string;
  title: string;
  body: string;
  category: string;
  author: string;
  author_type: string;
  status: string;
  created_at: string;
  published_at: string | null;
  allow_assistant: number;
  request_assistant: number;
  reply_count?: number;
  profile_url?: string;
}
export interface Reply {
  id: string;
  thread_id: string;
  body: string;
  author: string;
  author_type: string;
  status: string;
  created_at: string;
}
export const validId = (id: string) => /^[a-f0-9-]{36}$/.test(id);
export function questionFields(input: Record<string, unknown>) {
  const title = clean(input.title, 140),
    text = clean(input.body, 3000),
    author = clean(input.author, 50);
  if (title.length < 8 || text.length < 20 || author.length < 2)
    throw new HttpError(
      400,
      'Add a name, a title of at least 8 characters, and a question of at least 20 characters.',
    );
  if (
    /^(eidos|brent parent|eidos works|eidos assistant|studio|admin)$/i.test(
      author,
    )
  )
    throw new HttpError(400, 'Use your own display name.');
  return { title, text, author };
}
export async function publishedThread(env: PlatformEnv, id: string) {
  if (!validId(id))
    throw new HttpError(404, 'This conversation was not found.');
  const database = db(env);
  const thread = await database
    .prepare(
      "SELECT t.*,a.profile_url FROM eidos_threads t LEFT JOIN eidos_agents a ON a.id=t.owner_id WHERE t.id=? AND t.status='published'",
    )
    .bind(id)
    .first<Thread>();
  if (!thread) throw new HttpError(404, 'This conversation is not available.');
  const { results: replies } = await database
    .prepare(
      "SELECT id,thread_id,body,author,author_type,status,created_at FROM eidos_replies WHERE thread_id=? AND status='published' ORDER BY created_at LIMIT 100",
    )
    .bind(id)
    .all<Reply>();
  return { thread, replies };
}
/** No model calls or autonomous bot loops. Only a single published-source suggestion per human thread. */
export async function maybeSuggest(
  env: PlatformEnv,
  id: string,
  proactive = false,
) {
  const database = db(env);
  const data = await publishedThread(env, id);
  const { thread, replies } = data;
  if (
    thread.author_type === 'agent' ||
    replies.some((r) => r.author_type === 'eidos')
  )
    return false;
  if (proactive) {
    if (
      env.EIDOS_PROACTIVE_ENABLED !== 'true' ||
      !thread.allow_assistant ||
      replies.length ||
      Date.parse(thread.published_at || thread.created_at) >
        Date.now() - 86400000
    )
      return false;
  } else if (!thread.request_assistant) return false;
  const mention =
    replies
      .filter((r) => r.author_type === 'guest' && /@eidos\b/i.test(r.body))
      .at(-1)?.body || '';
  const question = thread.title + ' ' + thread.body + ' ' + mention;
  if (!knowledge.some((k) => k.match.test(question))) return false;
  if (!(await reserve(database, 'public-eidos-suggestions', 1, 10)))
    return false;
  const answer = sourceAnswer(question);
  const text =
    (proactive
      ? 'You opted in to a follow-up on unanswered questions. '
      : 'You asked @eidos for help. ') +
    'Here is relevant information from the studio’s published pages:\n\n' +
    answer.answer +
    '\n\nSources:\n' +
    answer.sources
      .map((s) => s.title + ': https://eidos-works.com' + s.href)
      .join('\n') +
    '\n\nSource-based suggestion · no model-generated answer.';
  const result = await database
    .prepare(
      "INSERT OR IGNORE INTO eidos_replies(id,thread_id,body,author,author_type,status,created_at) VALUES(?,?,?,'Eidos','eidos','published',?)",
    )
    .bind(crypto.randomUUID(), id, text, new Date().toISOString())
    .run();
  return result.meta.changes > 0;
}
