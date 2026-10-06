import { body, db, fingerprint, guarded, hash, HttpError, json, origin, reserve } from '../../_shared/platform/core';
import { memberFromRequest, randomToken, requireMember, username, type Member } from '../../_shared/platform/memberAuth';
import { emailValue } from '../../_shared/platform/memberCredentials';
import { flowCookie, googleReady, googleSession, readFlowCookie } from '../../_shared/platform/memberGoogle';

export const onRequestPost = guarded(async ({ request, env }) => {
  origin(request);
  if (!googleReady(request, env)) throw new HttpError(503, 'Google sign-in is being connected. Use your password or an email link.');
  const input = await body(request), database = db(env), visitor = await fingerprint(request, env);
  if (!(await reserve(database, 'google-ip:' + visitor, 1, 20, 3600))) throw new HttpError(429, 'Please wait before trying Google sign-in again.');
  if (input.action === 'complete-signup') {
    const name = username(input.username), token = readFlowCookie(request, env, 'onboard');
    if (!token) throw new HttpError(401, 'Start Google sign-in again.');
    const tokenHash = await hash(token);
    const ticket = await database.prepare('SELECT subject,email FROM eidos_member_google_signup WHERE token_hash=? AND consumed=0 AND expires>?').bind(tokenHash, Math.floor(Date.now() / 1000)).first<{ subject: string; email: string }>();
    if (!ticket) throw new HttpError(401, 'Start Google sign-in again.');
    if (await database.prepare('SELECT id FROM eidos_email_members WHERE username=?').bind(name).first()) throw new HttpError(409, 'That username is taken. Please choose another.');
    // One atomic ticket claim. Uniqueness constraints prevent duplicate email/sub identities.
    const claimed = await database.prepare('UPDATE eidos_member_google_signup SET consumed=1 WHERE token_hash=? AND consumed=0 AND expires>? RETURNING subject').bind(tokenHash, Math.floor(Date.now() / 1000)).first();
    if (!claimed) throw new HttpError(401, 'Start Google sign-in again.');
    const id = crypto.randomUUID(), now = new Date().toISOString();
    try {
      await database.batch([
        database.prepare('INSERT INTO eidos_email_members(id,email,username,kind,created_at,newsletter,newsletter_after) VALUES(?,?,?,?,?,?,?)').bind(id, ticket.email, name, 'person', now, input.newsletter === true ? 1 : 0, now),
        database.prepare('INSERT INTO eidos_member_google(subject,member_id,linked_at) VALUES(?,?,?)').bind(ticket.subject, id, now),
        database.prepare('UPDATE eidos_member_google_signup SET consumed=2 WHERE token_hash=?').bind(tokenHash),
      ]);
    } catch { throw new HttpError(409, 'An account identity was claimed while you were signing in. Start Google sign-in again, or sign in to your existing Eidos account to link it.'); }
    const response = json({ ok: true });
    response.headers.append('set-cookie', await googleSession(request, env, id, ticket.subject));
    response.headers.append('set-cookie', flowCookie(request, env, 'onboard', '', 0));
    return response;
  }
  if (!['start', 'link'].includes(String(input.action))) throw new HttpError(400, 'Choose Google sign-in or account linking.');
  const member = input.action === 'link' ? await requireMember(request, env, false) : null;
  if (member) {
    const password = await database.prepare('SELECT password_hash FROM eidos_member_passwords WHERE member_id=?').bind(member.id).first<{ password_hash: string }>();
    if (password && (typeof input.currentPassword !== 'string' || new TextEncoder().encode(input.currentPassword).length > 512 || !await env.EIDOS_PASSWORD_SERVICE?.verify(input.currentPassword, password.password_hash))) throw new HttpError(401, 'Enter your current password to link Google.');
  }
  const state = randomToken(), browser = randomToken(), nonce = randomToken(), verifier = randomToken();
  await database.prepare('INSERT INTO eidos_member_oauth_flows(state_hash,browser_hash,nonce,verifier,member_id,expires) VALUES(?,?,?,?,?,?)')
    .bind(await hash(state), await hash(browser), nonce, verifier, member?.id || null, Math.floor(Date.now() / 1000) + 600).run();
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier)));
  const pkce = btoa(String.fromCharCode(...digest)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  const authorize = new URL('https://accounts.google.com/o/oauth2/v2/auth');
  authorize.search = new URLSearchParams({ client_id: env.EIDOS_GOOGLE_CLIENT_ID!, redirect_uri: env.EIDOS_GOOGLE_REDIRECT_URI!, response_type: 'code', scope: 'openid email', state, nonce, code_challenge: pkce, code_challenge_method: 'S256', prompt: 'select_account' }).toString();
  const response = json({ authorizeUrl: authorize.href });
  response.headers.set('set-cookie', flowCookie(request, env, 'oauth', browser));
  return response;
});

/** Google returns to this same-origin relay route. Never log codes, tokens or provider bodies. */
export const onRequestGet = guarded(async ({ request, env }) => {
  const redirect = (status: string) => new Response(null, { status: 303, headers: { location: new URL('/account?oauth=' + status, request.url).href, 'cache-control': 'no-store', 'referrer-policy': 'no-referrer', 'set-cookie': flowCookie(request, env, 'oauth', '', 0) } });
  let response: Response;
  let claimedState: string | undefined;
  try {
    if (!googleReady(request, env)) throw Error('Unavailable');
    const url = new URL(request.url), state = url.searchParams.get('state') || '', code = url.searchParams.get('code') || '', browser = readFlowCookie(request, env, 'oauth');
    if (!/^[a-f0-9]{64}$/.test(state) || !browser || !code || code.length > 4096 || url.searchParams.has('error')) throw Error('Invalid callback');
    const database = db(env), stateHash = await hash(state);
    const flow = await database.prepare('UPDATE eidos_member_oauth_flows SET consumed=1 WHERE state_hash=? AND browser_hash=? AND consumed=0 AND expires>? RETURNING nonce,verifier,member_id')
      .bind(stateHash, await hash(browser), Math.floor(Date.now() / 1000)).first<{ nonce: string; verifier: string; member_id: string | null }>();
    if (!flow) throw Error('Invalid flow');
    claimedState = stateHash;
    const result = await fetch('https://oauth2.googleapis.com/token', { method: 'POST', headers: { 'content-type': 'application/x-www-form-urlencoded' }, body: new URLSearchParams({ client_id: env.EIDOS_GOOGLE_CLIENT_ID!, client_secret: env.EIDOS_GOOGLE_CLIENT_SECRET!, code, grant_type: 'authorization_code', redirect_uri: env.EIDOS_GOOGLE_REDIRECT_URI!, code_verifier: flow.verifier }), redirect: 'error', signal: AbortSignal.timeout(12000) });
    const data = await result.json() as { id_token?: unknown };
    if (!result.ok || typeof data.id_token !== 'string') throw Error('Token exchange failed');
    const identity = await env.EIDOS_GOOGLE_VERIFY!(data.id_token, flow.nonce);
    const email = emailValue(identity.email);
    const linked = await database.prepare('SELECT m.* FROM eidos_email_members m JOIN eidos_member_google g ON g.member_id=m.id WHERE g.subject=?').bind(identity.subject).first<Member & { disabled: number }>();
    let member: (Member & { disabled?: number }) | null = linked;
    if (flow.member_id) {
      const current = await memberFromRequest(request, env, false);
      if (!current || current.id !== flow.member_id || (linked && linked.id !== current.id)) return redirect('collision');
      const sameEmail = await database.prepare('SELECT id FROM eidos_email_members WHERE email=?').bind(email).first<{ id: string }>();
      if (sameEmail && sameEmail.id !== current.id) return redirect('collision');
      member = current;
    } else if (!member) {
      const byEmail = await database.prepare('SELECT * FROM eidos_email_members WHERE email=?').bind(email).first<Member & { disabled: number }>();
      // Google is authoritative for Gmail or verified Workspace domains only.
      // Third-party Google emails must first prove control through Eidos email login.
      if (!identity.authoritativeEmail) return redirect(byEmail ? 'link-required' : 'signup-required');
      member = byEmail;
    }
    if (member?.disabled) throw Error('Disabled account');
    if (member) {
      try { await database.prepare('INSERT INTO eidos_member_google(subject,member_id,linked_at) SELECT ?,?,? WHERE EXISTS(SELECT 1 FROM eidos_member_oauth_flows WHERE state_hash=? AND consumed=1) ON CONFLICT(subject) DO NOTHING').bind(identity.subject, member.id, new Date().toISOString(), stateHash).run(); }
      catch { return redirect('collision'); }
      response = redirect('success');
      response.headers.append('set-cookie', await googleSession(request, env, member.id, identity.subject, stateHash));
    } else {
      const ticket = randomToken();
      await database.prepare('INSERT INTO eidos_member_google_signup(token_hash,subject,email,expires) VALUES(?,?,?,?)').bind(await hash(ticket), identity.subject, email, Math.floor(Date.now() / 1000) + 600).run();
      response = redirect('choose-username');
      response.headers.append('set-cookie', flowCookie(request, env, 'onboard', ticket));
    }
  } catch { response = redirect('error'); }
  finally {
    if (claimedState) await db(env).prepare("UPDATE eidos_member_oauth_flows SET consumed=2,verifier='',nonce='' WHERE state_hash=?").bind(claimedState).run();
  }
  return response;
});
