import { createRemoteJWKSet, jwtVerify, type JWTVerifyGetKey } from 'jose';

const googleKeys = createRemoteJWKSet(new URL('https://www.googleapis.com/oauth2/v3/certs'), { timeoutDuration: 5000, cooldownDuration: 30000 });
export function googleVerifier(clientId: string | undefined, keys: JWTVerifyGetKey = googleKeys) {
  if (!clientId) return undefined;
  return async (token: string, nonce: string) => {
    const { payload } = await jwtVerify(token, keys, {
      issuer: ['https://accounts.google.com', 'accounts.google.com'], audience: clientId,
      algorithms: ['RS256'], requiredClaims: ['sub', 'email', 'email_verified', 'nonce', 'iat', 'exp'], maxTokenAge: '10m', clockTolerance: 5,
    });
    if (payload.nonce !== nonce || payload.email_verified !== true || typeof payload.sub !== 'string' || !/^[\x21-\x7e]{1,255}$/.test(payload.sub) || typeof payload.email !== 'string' || (payload.azp !== undefined && payload.azp !== clientId)) throw Error('Invalid Google identity');
    return { subject: payload.sub, email: payload.email.toLowerCase(), authoritativeEmail: /@gmail\.com$/i.test(payload.email) || (typeof payload.hd === 'string' && payload.hd.length > 0) };
  };
}
