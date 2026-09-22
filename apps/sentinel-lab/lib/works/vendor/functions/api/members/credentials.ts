import { body, challenge, clean, db, fingerprint, guarded, hash, HttpError, json, local, origin, reserve, siteOrigin } from '../../_shared/platform/core';
import { publicMember, randomToken, requireMember, username, type Member } from '../../_shared/platform/memberAuth';
import { emailValue, passwordSession, passwordValue, requirePasswords } from '../../_shared/platform/memberCredentials';
import { sendMemberMail } from '../../_shared/platform/memberMail';

const invalidToken = () => new HttpError(400, 'This link has expired or was already used. Request a new one.');
type AuthToken = { email: string; username: string; member_id: string | null; kind: string; newsletter: number; password_hash: string | null };

export const onRequestGet = guarded(async ({ request, env }) => {
  requirePasswords(request, env);
  const member = await requireMember(request, env, false);
  const [password, google] = await Promise.all([
    db(env).prepare('SELECT updated_at FROM eidos_member_passwords WHERE member_id=?').bind(member.id).first(),
    db(env).prepare('SELECT linked_at FROM eidos_member_google WHERE member_id=?').bind(member.id).first(),
  ]);
  return json({ hasPassword: Boolean(password), googleLinked: Boolean(google) });
});

export const onRequestPost = guarded(async ({ request, env }) => {
  origin(request);
  const service = requirePasswords(request, env), input = await body(request), action = clean(input.action, 30), database = db(env);
  const visitor = await fingerprint(request, env);
  if (!(await reserve(database, 'password-ip:' + visitor, 1, 30, 900))) throw new HttpError(429, 'Too many attempts. Wait a few minutes and try again.');

  if (action === 'login') {
    const identifier = clean(input.identifier, 260).toLowerCase();
    if (!identifier || typeof input.password !== 'string' || new TextEncoder().encode(input.password).length > 512) throw new HttpError(401, 'The username, email or password is incorrect.');
    if (!(await reserve(database, 'password-login:' + await hash(identifier), 1, 10, 900))) throw new HttpError(429, 'Too many attempts. Wait a few minutes and try again.');
    await challenge(request, env, input.challenge, 'member');
    const member = await database.prepare('SELECT m.*,p.password_hash FROM eidos_email_members m LEFT JOIN eidos_member_passwords p ON p.member_id=m.id WHERE (m.email=? OR m.username=?) AND m.disabled=0').bind(identifier, identifier).first<Member & { password_hash: string | null }>();
    // Missing accounts perform the same KDF with a fixed dummy salt/hash.
    const valid = await service.verify(input.password, member?.password_hash || null);
    if (!member || !member.password_hash || !valid) throw new HttpError(401, 'The username, email or password is incorrect.');
    const response = json({ member: publicMember(member) });
    response.headers.set('set-cookie', await passwordSession(request, env, member.id, member.password_hash));
    return response;
  }

  if (action === 'signup' || action === 'request-reset') {
    if (clean(input.website)) throw new HttpError(400, 'Please try again.');
    const email = emailValue(input.email), name = action === 'signup' ? username(input.username) : null;
    const password = action === 'signup' ? passwordValue(input.password) : null;
    if (!(await reserve(database, 'password-mail:' + await hash(email), 1, 3, 3600))) throw new HttpError(429, 'Please wait before requesting another account email.');
    await challenge(request, env, input.challenge, 'member');
    const existing = await database.prepare('SELECT id,disabled FROM eidos_email_members WHERE email=?').bind(email).first<{ id: string; disabled: number }>();
    const generic = { message: action === 'signup' ? 'Check your email to finish creating your account. If you already have an account, sign in or reset your password.' : 'If this address can sign in, a password reset email is on its way. The link expires in 15 minutes.' };
    if ((action === 'signup' && existing) || (action === 'request-reset' && (!existing || existing.disabled))) return json(generic);
    if (name && await database.prepare('SELECT id FROM eidos_email_members WHERE username=?').bind(name).first()) throw new HttpError(409, 'That username is taken. Please choose another.');
    const encoded = password ? await service.hash(password) : null, token = randomToken(), tokenHash = await hash(token), purpose = action === 'signup' ? 'signup' : 'reset';
    await database.prepare('INSERT INTO eidos_member_auth_tokens(token_hash,purpose,member_id,email,username,kind,newsletter,password_hash,expires) VALUES(?,?,?,?,?,?,?,?,?)')
      .bind(tokenHash, purpose, existing?.id || null, email, name, input.kind === 'agent' ? 'agent' : 'person', input.newsletter === true ? 1 : 0, encoded, Math.floor(Date.now() / 1000) + 900).run();
    const link = siteOrigin(env) + (purpose === 'signup' ? '/account/verify#intent=signup&token=' : '/account/reset#token=') + token;
    if (local(request, env)) return json({ ...generic, localVerificationUrl: link });
    await sendMemberMail(env, { to: email, subject: purpose === 'signup' ? 'Confirm your Eidos Works account' : 'Reset your Eidos Works password', text: `${purpose === 'signup' ? 'Confirm your email to create your account' : 'Choose a new password'}:\n\n${link}\n\nThis link works once and expires in 15 minutes. Only continue if you requested it. Your password is never included in email.` }, 'credential-' + tokenHash);
    return json(generic);
  }

  if (action === 'verify-signup' || action === 'reset') {
    const token = clean(input.token, 100);
    if (!/^[a-f0-9]{64}$/.test(token)) throw invalidToken();
    const purpose = action === 'verify-signup' ? 'signup' : 'reset', now = Math.floor(Date.now() / 1000), tokenHash = await hash(token);
    const newPassword = action === 'reset' ? passwordValue(input.password) : null;
    const link = await database.prepare('UPDATE eidos_member_auth_tokens SET consumed=1 WHERE token_hash=? AND purpose=? AND consumed=0 AND expires>? RETURNING email,username,member_id,kind,newsletter,password_hash').bind(tokenHash, purpose, now).first<AuthToken>();
    if (!link) throw invalidToken();
    const encoded = newPassword ? await service.hash(newPassword) : link.password_hash;
    if (!encoded) throw invalidToken();
    const id = action === 'verify-signup' ? crypto.randomUUID() : link.member_id, nowText = new Date().toISOString();
    if (!id) throw invalidToken();
    if (action === 'verify-signup') {
      await database.batch([
        database.prepare('INSERT OR IGNORE INTO eidos_email_members(id,email,username,kind,created_at,newsletter,newsletter_after) SELECT ?,?,?,?,?,?,? WHERE EXISTS(SELECT 1 FROM eidos_member_auth_tokens WHERE token_hash=? AND consumed=1)').bind(id, link.email, link.username, link.kind, nowText, link.newsletter, nowText, tokenHash),
        database.prepare('INSERT INTO eidos_member_passwords(member_id,password_hash,updated_at) SELECT id,?,? FROM eidos_email_members WHERE id=?').bind(encoded, nowText, id),
        database.prepare('UPDATE eidos_member_auth_tokens SET consumed=2,password_hash=NULL WHERE token_hash=?').bind(tokenHash),
      ]);
      const member = await database.prepare('SELECT * FROM eidos_email_members WHERE id=? AND disabled=0').bind(id).first<Member>();
      if (!member) throw new HttpError(409, 'Account creation could not finish. Sign in to an existing account or try another username.');
      const response = json({ member: publicMember(member) });
      response.headers.set('set-cookie', await passwordSession(request, env, id, encoded));
      return response;
    }
    await database.batch([
      database.prepare('INSERT INTO eidos_member_passwords(member_id,password_hash,updated_at) SELECT m.id,?,? FROM eidos_email_members m WHERE m.id=? AND m.disabled=0 AND EXISTS(SELECT 1 FROM eidos_member_auth_tokens WHERE token_hash=? AND consumed=1) ON CONFLICT(member_id) DO UPDATE SET password_hash=excluded.password_hash,updated_at=excluded.updated_at').bind(encoded, nowText, id, tokenHash),
      database.prepare('DELETE FROM eidos_member_sessions WHERE member_id=? AND EXISTS(SELECT 1 FROM eidos_member_passwords WHERE member_id=? AND password_hash=?)').bind(id, id, encoded),
      database.prepare('UPDATE eidos_signin_links SET consumed=2 WHERE email=? AND EXISTS(SELECT 1 FROM eidos_member_passwords WHERE member_id=? AND password_hash=?)').bind(link.email, id, encoded),
      database.prepare('UPDATE eidos_member_auth_tokens SET consumed=2,password_hash=NULL WHERE email=? AND EXISTS(SELECT 1 FROM eidos_member_passwords WHERE member_id=? AND password_hash=?)').bind(link.email, id, encoded),
      database.prepare('UPDATE eidos_member_oauth_flows SET consumed=2 WHERE member_id=? AND EXISTS(SELECT 1 FROM eidos_member_passwords WHERE member_id=? AND password_hash=?)').bind(id, id, encoded),
    ]);
    if (!await database.prepare('SELECT member_id FROM eidos_member_passwords WHERE member_id=? AND password_hash=?').bind(id, encoded).first()) throw invalidToken();
    return json({ message: 'Your password has been reset and all devices have been signed out. Sign in with your new password.' });
  }

  if (action === 'change-password') {
    const member = await requireMember(request, env, false), password = passwordValue(input.password);
    const credential = await database.prepare('SELECT password_hash FROM eidos_member_passwords WHERE member_id=?').bind(member.id).first<{ password_hash: string }>();
    if (!credential) throw new HttpError(400, 'Use the password reset email to set your first password.');
    if (typeof input.currentPassword !== 'string' || new TextEncoder().encode(input.currentPassword).length > 512 || !await service.verify(input.currentPassword, credential.password_hash)) throw new HttpError(401, 'Your current password is incorrect.');
    const encoded = await service.hash(password);
    await database.batch([
      database.prepare('UPDATE eidos_member_passwords SET password_hash=?,updated_at=? WHERE member_id=? AND password_hash=?').bind(encoded, new Date().toISOString(), member.id, credential.password_hash),
      database.prepare('DELETE FROM eidos_member_sessions WHERE member_id=? AND EXISTS(SELECT 1 FROM eidos_member_passwords WHERE member_id=? AND password_hash=?)').bind(member.id, member.id, encoded),
      database.prepare('UPDATE eidos_signin_links SET consumed=2 WHERE email=? AND EXISTS(SELECT 1 FROM eidos_member_passwords WHERE member_id=? AND password_hash=?)').bind(member.email, member.id, encoded),
      database.prepare('UPDATE eidos_member_auth_tokens SET consumed=2,password_hash=NULL WHERE email=? AND EXISTS(SELECT 1 FROM eidos_member_passwords WHERE member_id=? AND password_hash=?)').bind(member.email, member.id, encoded),
      database.prepare('UPDATE eidos_member_oauth_flows SET consumed=2 WHERE member_id=? AND EXISTS(SELECT 1 FROM eidos_member_passwords WHERE member_id=? AND password_hash=?)').bind(member.id, member.id, encoded),
    ]);
    const response = json({ message: 'Password changed. Your other devices have been signed out.' });
    response.headers.set('set-cookie', await passwordSession(request, env, member.id, encoded));
    return response;
  }
  throw new HttpError(400, 'Choose a valid account action.');
});
