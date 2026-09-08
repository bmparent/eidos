import { db, guarded, HttpError, json } from '../../_shared/platform/core';
import { accountsReady } from '../../_shared/platform/memberAuth';
export const onRequestGet = guarded(async ({ request, env }) => {
  if (!accountsReady(request, env)) return json({ members: [] });
  const q = (new URL(request.url).searchParams.get('q') || '').toLowerCase();
  if (!/^[a-z][a-z0-9_]{1,23}$/.test(q))
    throw new HttpError(400, 'Enter at least two username characters.');
  const { results } = await db(env)
    .prepare(
      'SELECT username,kind,created_at FROM eidos_members WHERE disabled=0 AND substr(username,1,?)=? ORDER BY username LIMIT 8',
    )
    .bind(q.length, q)
    .all();
  return json({ members: results });
});
