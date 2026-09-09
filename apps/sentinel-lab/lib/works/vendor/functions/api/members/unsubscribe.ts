import {
  body,
  db,
  guarded,
  hash,
  HttpError,
  json,
  origin,
} from '../../_shared/platform/core';
/** A GET (including an email scanner) never changes a subscription. */
export const onRequestPost = guarded(async ({ request, env }) => {
  origin(request);
  const input = await body(request);
  if (typeof input.token !== 'string' || !/^[a-f0-9]{64}$/.test(input.token))
    throw new HttpError(
      400,
      'This unsubscribe link is invalid. You can also turn emails off in your account.',
    );
  const database = db(env);
  const row = await database
    .prepare(
      'SELECT member_id FROM eidos_unsubscribe_tokens WHERE token_hash=?',
    )
    .bind(await hash(input.token))
    .first<{ member_id: string }>();
  if (!row)
    throw new HttpError(
      400,
      'This unsubscribe link is invalid. You can also turn emails off in your account.',
    );
  await database.batch([
    database
      .prepare('UPDATE eidos_email_members SET newsletter=0 WHERE id=?')
      .bind(row.member_id),
    database
      .prepare(
        "UPDATE eidos_mail_deliveries SET status='canceled',payload='' WHERE member_id=? AND status='pending'",
      )
      .bind(row.member_id),
  ]);
  return json({
    ok: true,
    message:
      'You are unsubscribed. Your free account and saved reading list are still available.',
  });
});
