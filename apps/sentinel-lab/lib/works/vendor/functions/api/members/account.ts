import {
  body,
  clean,
  db,
  guarded,
  hash,
  HttpError,
  json,
  origin,
  reserve,
} from '../../_shared/platform/core';
import {
  accountsReady,
  cookieHeader,
  memberFromRequest,
  publicMember,
  randomToken,
  requireMember,
} from '../../_shared/platform/memberAuth';
import { inboxQuery } from '../../_shared/platform/mentions';

export const onRequestGet = guarded(async ({ request, env }) => {
  const member = await memberFromRequest(request, env);
  if (!member)
    return json({ member: null, ready: accountsReady(request, env) });
  const database = db(env);
  const [saved, inbox, keys] = await Promise.all([
    database
      .prepare(
        'SELECT slug,created_at FROM eidos_bookmarks WHERE member_id=? ORDER BY created_at DESC LIMIT 200',
      )
      .bind(member.id)
      .all(),
    database.prepare(inboxQuery).bind(member.id).all(),
    database
      .prepare(
        'SELECT id,last_four,created_at FROM eidos_member_keys WHERE member_id=? AND revoked=0 ORDER BY created_at DESC LIMIT 10',
      )
      .bind(member.id)
      .all(),
  ]);
  const isAgentKey = (request.headers.get('authorization') || '').startsWith(
    'Bearer ew_agent_',
  );
  return json({
    member: {
      ...publicMember(member),
      ...(!isAgentKey ? { email: member.email } : {}),
      newsletter: Boolean(member.newsletter),
    },
    saved: saved.results,
    inbox: inbox.results,
    keys: keys.results,
  });
});
export const onRequestPost = guarded(async ({ request, env }) => {
  const input = await body(request);
  const action = clean(input.action, 30);
  const agentAllowed = ['bookmark', 'read-mention'].includes(action);
  const apiKey = (request.headers.get('authorization') || '').startsWith(
    'Bearer ew_agent_',
  );
  if (!apiKey || !agentAllowed) origin(request);
  const member = await requireMember(request, env, agentAllowed);
  const database = db(env);
  if (!(await reserve(database, 'account-write:' + member.id, 1, 120, 3600)))
    throw new HttpError(429, 'Please wait before making more account changes.');
  if (action === 'preferences') {
    if (typeof input.newsletter !== 'boolean')
      throw new HttpError(400, 'Choose your delivery preference.');
    await database
      .prepare(
        'UPDATE eidos_members SET newsletter=?,newsletter_after=CASE WHEN newsletter=0 AND ?=1 THEN ? ELSE newsletter_after END WHERE id=?',
      )
      .bind(
        input.newsletter ? 1 : 0,
        input.newsletter ? 1 : 0,
        new Date().toISOString(),
        member.id,
      )
      .run();
    if (!input.newsletter)
      await database
        .prepare(
          "UPDATE eidos_mail_deliveries SET status='canceled',payload='' WHERE member_id=? AND status='pending'",
        )
        .bind(member.id)
        .run();
    return json({
      ok: true,
      message: input.newsletter
        ? 'You’ll receive new white papers and posts in a free daily email.'
        : 'Article emails are off. Your account and saved reading stay available.',
    });
  }
  if (action === 'bookmark') {
    const slug = clean(input.slug, 180);
    if (
      !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug) ||
      typeof input.saved !== 'boolean'
    )
      throw new HttpError(400, 'Choose a valid article.');
    if (input.saved) {
      const existing = await database
        .prepare(
          'SELECT slug FROM eidos_bookmarks WHERE member_id=? AND slug=?',
        )
        .bind(member.id, slug)
        .first();
      if (!existing) {
        const added = await database
          .prepare(
            'INSERT INTO eidos_bookmarks(member_id,slug,created_at) SELECT ?,?,? WHERE (SELECT COUNT(*) FROM eidos_bookmarks WHERE member_id=?)<200 ON CONFLICT DO NOTHING RETURNING slug',
          )
          .bind(member.id, slug, new Date().toISOString(), member.id)
          .first();
        if (!added)
          throw new HttpError(
            400,
            'Your reading list has 200 entries. Remove one to save another.',
          );
      }
    } else
      await database
        .prepare('DELETE FROM eidos_bookmarks WHERE member_id=? AND slug=?')
        .bind(member.id, slug)
        .run();
    return json({ ok: true });
  }
  if (action === 'read-mention') {
    await database
      .prepare('UPDATE eidos_mentions SET read_at=? WHERE member_id=? AND id=?')
      .bind(new Date().toISOString(), member.id, clean(input.id, 36))
      .run();
    return json({ ok: true });
  }
  if (action === 'create-key') {
    if (member.kind !== 'agent')
      throw new HttpError(403, 'Agent keys belong to agent accounts.');
    const key = 'ew_agent_' + randomToken(),
      id = crypto.randomUUID();
    const created = await database
      .prepare(
        'INSERT INTO eidos_member_keys(id,member_id,key_hash,last_four,created_at) SELECT ?,?,?,?,? WHERE (SELECT COUNT(*) FROM eidos_member_keys WHERE member_id=? AND revoked=0)<3 RETURNING id',
      )
      .bind(
        id,
        member.id,
        await hash(key),
        key.slice(-4),
        new Date().toISOString(),
        member.id,
      )
      .first();
    if (!created)
      throw new HttpError(
        400,
        'Revoke an existing key before creating another.',
      );
    return json(
      {
        id,
        key,
        message:
          'Copy this key now. It is shown once and grants access only to your account and community submissions.',
      },
      201,
    );
  }
  if (action === 'revoke-key') {
    await database
      .prepare(
        'UPDATE eidos_member_keys SET revoked=1 WHERE member_id=? AND id=?',
      )
      .bind(member.id, clean(input.id, 36))
      .run();
    return json({ ok: true });
  }
  if (action === 'signout-all') {
    await database
      .prepare('DELETE FROM eidos_member_sessions WHERE member_id=?')
      .bind(member.id)
      .run();
    const response = json({ ok: true });
    response.headers.set('set-cookie', cookieHeader(request, env, '', 0));
    return response;
  }
  throw new HttpError(400, 'Choose a valid account action.');
});
