import { clean, db, hash, HttpError, local, type PlatformEnv } from './core';

export const sessionCookie = '__Host-eidos_session';
export interface Member {
  id: string;
  email: string;
  username: string;
  kind: 'person' | 'agent';
  created_at: string;
  newsletter: number;
  newsletter_after: string;
}
export function accountsReady(request: Request, env: PlatformEnv) {
  return Boolean(
    env.EIDOS_DB &&
    ((env.EIDOS_ACCOUNTS_ENABLED === 'true' &&
      env.RESEND_API_KEY &&
      env.EIDOS_MAIL_FROM &&
      (env.EIDOS_RATE_SECRET?.length || 0) >= 32 &&
      env.TURNSTILE_SECRET_KEY &&
      env.TURNSTILE_SITE_KEY) ||
      local(request, env)),
  );
}
export function requireAccounts(request: Request, env: PlatformEnv) {
  if (!accountsReady(request, env))
    throw new HttpError(
      503,
      'Member sign-in is being connected. The complete Insights archive is free to read now.',
    );
}
export const randomToken = () =>
  [...crypto.getRandomValues(new Uint8Array(32))]
    .map((n) => n.toString(16).padStart(2, '0'))
    .join('');
export function readSessionCookie(request: Request) {
  const name = ['localhost', '127.0.0.1', 'terminal.local'].includes(
    new URL(request.url).hostname,
  )
    ? 'eidos_session_local'
    : sessionCookie;
  return (
    (request.headers.get('cookie') || '')
      .split(';')
      .map((s) => s.trim())
      .find((s) => s.startsWith(name + '='))
      ?.slice(name.length + 1) || ''
  );
}
export function cookieHeader(
  request: Request,
  env: PlatformEnv,
  token: string,
  age = 2592000,
) {
  return `${local(request, env) ? 'eidos_session_local' : sessionCookie}=${token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${age}${local(request, env) ? '' : '; Secure'}`;
}
export async function memberFromRequest(
  request: Request,
  env: PlatformEnv,
  allowAgent = true,
): Promise<Member | null> {
  if (!accountsReady(request, env)) return null;
  const bearer = (request.headers.get('authorization') || '').replace(
    /^Bearer /,
    '',
  );
  if (bearer.startsWith('ew_agent_')) {
    if (!allowAgent || !/^ew_agent_[a-f0-9]{64}$/.test(bearer))
      throw new HttpError(401, 'Use email sign-in to manage your account.');
    const member = await db(env)
      .prepare(
        'SELECT m.* FROM eidos_email_members m JOIN eidos_member_keys k ON k.member_id=m.id WHERE k.key_hash=? AND k.revoked=0 AND m.disabled=0',
      )
      .bind(await hash(bearer))
      .first<Member>();
    if (!member) throw new HttpError(401, 'This agent key is not active.');
    return member;
  }
  const token = readSessionCookie(request);
  if (!/^[a-f0-9]{64}$/.test(token)) return null;
  return db(env)
    .prepare(
      'SELECT m.* FROM eidos_email_members m JOIN eidos_member_sessions s ON s.member_id=m.id WHERE s.token_hash=? AND s.expires>? AND m.disabled=0',
    )
    .bind(await hash(token), Math.floor(Date.now() / 1000))
    .first<Member>();
}
export async function requireMember(
  request: Request,
  env: PlatformEnv,
  allowAgent = true,
) {
  requireAccounts(request, env);
  const member = await memberFromRequest(request, env, allowAgent);
  if (!member)
    throw new HttpError(401, 'Sign in to your Eidos Works account first.');
  return member;
}
export function username(value: unknown) {
  const name = clean(value, 40).toLowerCase();
  if (
    !/^[a-z][a-z0-9_]{2,23}$/.test(name) ||
    /^(eidos|eidosworks|eidos_works|brent|bmparent|brent_parent|admin|administrator|studio|support|moderator|system|sentinel)$/.test(
      name,
    )
  )
    throw new HttpError(
      400,
      'Choose a username of 3–24 letters, numbers, or underscores, starting with a letter. Studio names are reserved.',
    );
  return name;
}
export const publicMember = (m: Member) => ({
  username: m.username,
  kind: m.kind,
  createdAt: m.created_at,
});
