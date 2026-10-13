import { createRemoteJWKSet, jwtVerify, type JWTVerifyGetKey } from 'jose';

type AccessSettings = {
  EIDOS_ACCESS_TEAM_DOMAIN?: string;
  EIDOS_ACCESS_AUD?: string;
  EIDOS_OWNER_EMAIL?: string;
};

const keysets = new Map<string, JWTVerifyGetKey>();

/** Validate the signed Access application token, never just its forwarded email header. */
export async function verifyAccessOwner(
  token: string,
  settings: AccessSettings,
  keys?: JWTVerifyGetKey,
) {
  const domain = settings.EIDOS_ACCESS_TEAM_DOMAIN?.toLowerCase();
  const audience = settings.EIDOS_ACCESS_AUD;
  const owner = settings.EIDOS_OWNER_EMAIL?.trim().toLowerCase();
  if (!domain || !/^[a-z0-9-]+\.cloudflareaccess\.com$/.test(domain) ||
      !audience || !/^[a-zA-Z0-9_-]{16,128}$/.test(audience) ||
      !owner || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(owner)) {
    throw Error('Owner Access is not configured.');
  }
  if (token.length < 64 || token.length > 8192) throw Error('Invalid Access token.');
  let keyset = keys || keysets.get(domain);
  if (!keyset) {
    keyset = createRemoteJWKSet(new URL(`https://${domain}/cdn-cgi/access/certs`), {
      timeoutDuration: 5000,
      cooldownDuration: 30000,
    });
    keysets.set(domain, keyset);
  }
  const { payload } = await jwtVerify(token, keyset, {
    issuer: `https://${domain}`,
    audience,
    algorithms: ['RS256'],
    requiredClaims: ['iss', 'aud', 'iat', 'exp', 'email'],
    maxTokenAge: '24h',
    clockTolerance: 30,
  });
  if (typeof payload.email !== 'string' || payload.email.toLowerCase() !== owner)
    throw Error('Unrecognized owner.');
  return owner;
}
