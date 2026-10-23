import {
  body,
  challenge,
  clean,
  db,
  fingerprint,
  guarded,
  hash,
  HttpError,
  json,
  local,
  origin,
  reserve,
  siteOrigin,
} from '../../_shared/platform/core';
import {
  cookieHeader,
  publicMember,
  randomToken,
  readSessionCookie,
  requireAccounts,
  username,
  type Member,
} from '../../_shared/platform/memberAuth';
import { sendMemberMail } from '../../_shared/platform/memberMail';

export const onRequestPost = guarded(async ({ request, env }) => {
  origin(request);
  requireAccounts(request, env);
  const input = await body(request);
  const database = db(env),
    action = clean(input.action, 20);
  if (action === 'logout') {
    await database
      .prepare('DELETE FROM eidos_member_sessions WHERE token_hash=?')
      .bind(await hash(readSessionCookie(request)))
      .run();
    const response = json({ ok: true });
    response.headers.set('set-cookie', cookieHeader(request, env, '', 0));
    return response;
  }
  if (action === 'verify') {
    const token = clean(input.token, 100);
    if (!/^[a-f0-9]{64}$/.test(token))
      throw new HttpError(
        400,
        'This sign-in link is invalid. Request a new one.',
      );
    const now = Math.floor(Date.now() / 1000);
    const link = await database
      .prepare(
        'UPDATE eidos_signin_links SET consumed=1 WHERE token_hash=? AND consumed=0 AND expires>? RETURNING email,username,kind,newsletter',
      )
      .bind(await hash(token), now)
      .first<{
        email: string;
        username: string | null;
        kind: 'person' | 'agent';
        newsletter: number;
      }>();
    if (!link)
      throw new HttpError(
        400,
        'This sign-in link has expired or was already used. Request a new one.',
      );
    let member = await database
      .prepare('SELECT * FROM eidos_email_members WHERE email=? AND disabled=0')
      .bind(link.email)
      .first<Member>();
    if (!member) {
      if (!link.username)
        throw new HttpError(400, 'Create an account to continue.');
      const taken = await database
        .prepare('SELECT id FROM eidos_email_members WHERE username=?')
        .bind(link.username)
        .first();
      if (taken)
        throw new HttpError(
          409,
          'That username has been claimed. Create an account with another username.',
        );
      const nowText = new Date().toISOString(),
        id = crypto.randomUUID();
      await database
        .prepare(
          'INSERT OR IGNORE INTO eidos_email_members(id,email,username,kind,created_at,newsletter,newsletter_after) VALUES(?,?,?,?,?,?,?)',
        )
        .bind(
          id,
          link.email,
          link.username,
          link.kind,
          nowText,
          link.newsletter,
          nowText,
        )
        .run();
      member = await database
        .prepare('SELECT * FROM eidos_email_members WHERE email=? AND disabled=0')
        .bind(link.email)
        .first<Member>();
      if (!member)
        throw new HttpError(
          409,
          'That username has been claimed. Create an account with another username.',
        );
    }
    const session = randomToken();
    const createdSession = await database
      .prepare(
        'INSERT INTO eidos_member_sessions(token_hash,member_id,expires) SELECT ?,?,? WHERE EXISTS(SELECT 1 FROM eidos_signin_links WHERE token_hash=? AND consumed=1) RETURNING token_hash',
      )
      .bind(await hash(session), member.id, now + 2592000, await hash(token))
      .first();
    if (!createdSession) throw new HttpError(401, 'Your account security changed. Request a new sign-in link.');
    const response = json({ member: publicMember(member) });
    response.headers.set('set-cookie', cookieHeader(request, env, session));
    return response;
  }
  if (!['signup', 'signin'].includes(action))
    throw new HttpError(400, 'Choose sign in or create account.');
  if (clean(input.website)) throw new HttpError(400, 'Please try again.');
  const email = clean(input.email, 260).toLowerCase();
  if (!/^[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+$/.test(email))
    throw new HttpError(400, 'Enter a valid email address.');
  const name = action === 'signup' ? username(input.username) : null;
  const kind = input.kind === 'agent' ? 'agent' : 'person';
  const visitor = await fingerprint(request, env);
  if (
    !(await reserve(database, 'signin-ip:' + visitor, 1, 8, 3600)) ||
    !(await reserve(
      database,
      'signin-email:' + (await hash(email)),
      1,
      3,
      3600,
    ))
  )
    throw new HttpError(
      429,
      'Please wait before requesting another sign-in email.',
    );
  await challenge(request, env, input.challenge, 'member');
  const existing = await database
    .prepare('SELECT id FROM eidos_email_members WHERE email=? AND disabled=0')
    .bind(email)
    .first();
  const generic = {
    message:
      'If this address can sign in, an email is on its way. Open the link within 15 minutes. Check spam if it does not arrive.',
  };
  if (action === 'signin' && !existing) return json(generic);
  if (
    name &&
    !existing &&
    (await database
      .prepare('SELECT id FROM eidos_email_members WHERE username=?')
      .bind(name)
      .first())
  )
    throw new HttpError(409, 'That username is taken. Please choose another.');
  const token = randomToken(),
    tokenHash = await hash(token);
  await database
    .prepare(
      'INSERT INTO eidos_signin_links(token_hash,email,username,kind,newsletter,expires) VALUES(?,?,?,?,?,?)',
    )
    .bind(
      tokenHash,
      email,
      name,
      kind,
      input.newsletter === true ? 1 : 0,
      Math.floor(Date.now() / 1000) + 900,
    )
    .run();
  const link = siteOrigin(env) + '/account/verify#token=' + token;
  if (local(request, env))
    return json({ ...generic, localVerificationUrl: link });
  await sendMemberMail(
    env,
    {
      to: email,
      subject: 'Your Eidos Works sign-in link',
      text: `Sign in to Eidos Works:\n\n${link}\n\nThis link works once and expires in 15 minutes. Only use it if you requested it. Your email is private; your chosen username and account type are public.\n\nThe complete archive is always free: ${siteOrigin(env)}/insights`,
    },
    'signin-' + tokenHash,
  );
  return json(generic);
});
