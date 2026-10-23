import { db, hash, HttpError, local, type PlatformEnv } from './core';
import { accountsReady, cookieHeader, randomToken } from './memberAuth';

export function passwordsReady(request: Request, env: PlatformEnv) {
  return accountsReady(request, env) && Boolean(env.EIDOS_PASSWORD_SERVICE) &&
    (env.EIDOS_PASSWORD_AUTH_ENABLED === 'true' || local(request, env));
}
export function requirePasswords(request: Request, env: PlatformEnv) {
  if (!passwordsReady(request, env)) throw new HttpError(503, 'Password sign-in is being connected. You can still use an email sign-in link.');
  return env.EIDOS_PASSWORD_SERVICE!;
}
export function passwordValue(value: unknown) {
  // Preserve every character, including spaces. Permit paste and password managers.
  if (typeof value !== 'string' || [...value].length < 15 || [...value].length > 128 || new TextEncoder().encode(value).length > 512)
    throw new HttpError(400, 'Use a password or passphrase of 15–128 characters.');
  if (/^(.)\1+$/.test(value) || ['passwordpassword', 'password123456789', '123456789012345', 'qwertyuiopasdfgh', 'letmeinletmeinletmein'].includes(value.toLowerCase()))
    throw new HttpError(400, 'Choose a less predictable password or passphrase.');
  return value;
}
export function emailValue(value: unknown) {
  if (typeof value !== 'string') throw new HttpError(400, 'Enter a valid email address.');
  const email = value.trim().toLowerCase();
  if (email.length > 260 || !/^[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+$/.test(email)) throw new HttpError(400, 'Enter a valid email address.');
  return email;
}
/** The credential predicate is checked when the session is inserted, after the KDF.
 * A concurrent reset cannot authenticate a password that was valid before it. */
export async function passwordSession(request: Request, env: PlatformEnv, memberId: string, encoded: string) {
  const session = randomToken();
  const created = await db(env).prepare('INSERT INTO eidos_member_sessions(token_hash,member_id,expires) SELECT ?,m.id,? FROM eidos_email_members m JOIN eidos_member_passwords p ON p.member_id=m.id WHERE m.id=? AND m.disabled=0 AND p.password_hash=? RETURNING token_hash')
    .bind(await hash(session), Math.floor(Date.now() / 1000) + 2592000, memberId, encoded).first();
  if (!created) throw new HttpError(401, 'Sign-in could not complete. Please try again.');
  return cookieHeader(request, env, session);
}
