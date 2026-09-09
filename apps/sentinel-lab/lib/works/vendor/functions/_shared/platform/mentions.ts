import { db, type PlatformEnv } from './core';
// An email address or the tail of a longer username is not a mention.
export function mentionedUsernames(text: string) {
  return [
    ...new Set(
      [
        ...text.matchAll(
          /(?:^|[^a-z0-9_@])@([a-z][a-z0-9_]{2,23})(?![a-z0-9_])/gi,
        ),
      ].map((m) => m[1].toLowerCase()),
    ),
  ].slice(0, 10);
}
export async function recordMentions(
  env: PlatformEnv,
  text: string,
  sourceId: string,
  sourceKind: 'thread' | 'reply',
  threadId: string,
  sender: string,
  senderId = '',
) {
  if (env.EIDOS_ACCOUNTS_ENABLED !== 'true' && env.EIDOS_LOCAL_TEST !== 'true')
    return;
  const names = mentionedUsernames(text).filter((n) => n !== 'eidos');
  for (const name of names) {
    const target = await db(env)
      .prepare(
        'SELECT id FROM eidos_email_members WHERE username=? AND disabled=0 AND id<>?',
      )
      .bind(name, senderId)
      .first<{ id: string }>();
    if (target)
      await db(env)
        .prepare(
          'INSERT OR IGNORE INTO eidos_mentions(id,member_id,source_id,source_kind,thread_id,sender,created_at) VALUES(?,?,?,?,?,?,?)',
        )
        .bind(
          crypto.randomUUID(),
          target.id,
          sourceId,
          sourceKind,
          threadId,
          sender,
          new Date().toISOString(),
        )
        .run();
  }
}
export const inboxQuery = `SELECT n.id,n.thread_id,n.source_id,n.source_kind,n.sender,n.created_at,n.read_at,t.title FROM eidos_mentions n
 JOIN eidos_threads t ON t.id=n.thread_id AND t.status='published'
 LEFT JOIN eidos_replies r ON r.id=n.source_id AND n.source_kind='reply'
 WHERE n.member_id=? AND (n.source_kind='thread' OR r.status='published') ORDER BY n.created_at DESC LIMIT 100`;
