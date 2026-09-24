import { db, hash, HttpError, local, type PlatformEnv } from './core';
import { accountsReady, cookieHeader, randomToken } from './memberAuth';

export function googleReady(request: Request, env: PlatformEnv) {
  return Boolean(accountsReady(request, env) && env.EIDOS_GOOGLE_VERIFY && env.EIDOS_GOOGLE_CLIENT_ID && env.EIDOS_GOOGLE_CLIENT_SECRET &&
    env.EIDOS_GOOGLE_REDIRECT_URI === new URL('/api/members/google', request.url).href);
}
export function flowCookie(request: Request, env: PlatformEnv, name: 'oauth' | 'onboard', value: string, age = 600) {
  return `${local(request, env) ? 'eidos_' + name + '_local' : '__Host-eidos_' + name}=${value}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${age}${local(request, env) ? '' : '; Secure'}`;
}
export function readFlowCookie(request: Request, env: PlatformEnv, name: 'oauth' | 'onboard') {
  const prefix = (local(request, env) ? 'eidos_' + name + '_local' : '__Host-eidos_' + name) + '=';
  const value = (request.headers.get('cookie') || '').split(';').map(s => s.trim()).find(s => s.startsWith(prefix))?.slice(prefix.length) || '';
  return /^[a-f0-9]{64}$/.test(value) ? value : '';
}
export async function googleSession(request: Request, env: PlatformEnv, memberId: string, subject: string, flowHash?: string) {
  const token = randomToken();
  const result = await db(env).prepare('INSERT INTO eidos_member_sessions(token_hash,member_id,expires) SELECT ?,m.id,? FROM eidos_email_members m JOIN eidos_member_google g ON g.member_id=m.id WHERE m.id=? AND g.subject=? AND m.disabled=0 AND (? IS NULL OR EXISTS(SELECT 1 FROM eidos_member_oauth_flows WHERE state_hash=? AND consumed=1)) RETURNING token_hash')
    .bind(await hash(token), Math.floor(Date.now() / 1000) + 2592000, memberId, subject, flowHash || null, flowHash || null).first();
  if (!result) throw new HttpError(401, 'Google sign-in could not complete. Please start again.');
  return cookieHeader(request, env, token);
}
