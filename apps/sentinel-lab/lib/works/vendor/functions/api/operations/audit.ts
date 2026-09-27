import { admin, db, guarded, HttpError, json } from '../../_shared/platform/core';

export const onRequestGet = guarded(async ({ request, env }) => {
  await admin(request, env);
  const url = new URL(request.url);
  if ([...url.searchParams.keys()].some(key => key !== 'limit'))
    throw new HttpError(400, 'Invalid audit query.');
  const value = url.searchParams.get('limit') || '50';
  if (!/^(?:[1-9]|[1-9][0-9]|100)$/.test(value))
    throw new HttpError(400, 'Choose between 1 and 100 audit records.');
  const events = await db(env).prepare(
    'SELECT id,actor,method,path,event,created_at FROM eidos_owner_audit ORDER BY created_at DESC,id DESC LIMIT ?',
  ).bind(Number(value)).all();
  return json({ events: events.results });
});
