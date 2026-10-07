import { randomBytes, timingSafeEqual } from "node:crypto";
import { guidedStore, hash, LabError, type GuidedStore } from "./store";

export type Principal = { id: string; credential: "pilot" | "member" | "operator" | "connector"; source?: string };
const same = (a: string, b: string) => timingSafeEqual(Buffer.from(hash(a), "hex"), Buffer.from(hash(b), "hex"));

export async function principal(request: Request, store = guidedStore(), allowConnector = false): Promise<Principal> {
  const bearer = request.headers.get("Authorization")?.match(/^Bearer (\S+)$/)?.[1] || "";
  const cookieName = process.env.EIDOS_GUIDED_LOCAL === "1" && !process.env.VERCEL ? "eidos_lab_local" : "__Host-eidos_lab";
  const token = bearer || (request.headers.get("Cookie") || "").split(";").map(s => s.trim()).find(s => s.startsWith(cookieName + "="))?.slice(cookieName.length + 1) || "";
  if (token.startsWith("eidos_ingest_")) {
    if (!allowConnector) throw new LabError(403, "This credential can ingest only into its assigned source.");
    await store.initialize();
    const result = await store.client.execute({ sql: "SELECT owner,source FROM sentinel_guided_keys WHERE scope=? AND digest=? AND revoked=0 AND expires>?", args: [store.scope, hash(token), Date.now()] });
    if (!result.rows[0]) throw new LabError(401, "Connector credential is expired or revoked.");
    return { id: String(result.rows[0].owner), source: String(result.rows[0].source), credential: "connector" };
  }
  if (token.startsWith("eidos_session_")) {
    const session = await store.get("sessions", "session", hash(token)).catch(() => null);
    if (session && session.expires > Date.now()) {
      // Revalidate the underlying pilot grant on every request so revocation is immediate.
      const grants = activeGrants();
      if (session.credential === "pilot" && !grants.some(g => g.id === session.grantId && g.sha256 === session.grantHash))
        throw new LabError(401, "Your pilot access has expired or been revoked.");
      if (session.credential === "operator" && session.grantHash !== hash(process.env.EIDOS_OPERATOR_TOKEN || ""))
        throw new LabError(401, "Operator access changed. Sign in again.");
      return { id: session.owner, credential: session.credential };
    }
  }
  const expected = process.env.EIDOS_OPERATOR_TOKEN?.trim();
  if (token && expected && same(token, expected)) return { id: "operator", credential: "operator" };
  const grant = activeGrants().find(g => g.sha256 === hash(token));
  if (token && grant) return { id: `pilot:${grant.id}`, credential: "pilot" };
  // Reuse the existing Works member-session tables if installed. Sessions are
  // accepted only by their hash; no account creation or access policy is changed.
  const works = (request.headers.get("Cookie") || "").match(/(?:^|;\s*)(?:__Host-eidos_session|eidos_session_local)=([a-f0-9]{64})(?:;|$)/)?.[1];
  if (works || /^ew_agent_[a-f0-9]{64}$/.test(bearer)) {
    const agent = bearer.startsWith("ew_agent_");
    const sql = agent ? "SELECT m.id FROM eidos_email_members m JOIN eidos_member_keys k ON k.member_id=m.id WHERE k.key_hash=? AND k.revoked=0 AND m.disabled=0" :
      "SELECT m.id FROM eidos_email_members m JOIN eidos_member_sessions s ON s.member_id=m.id WHERE s.token_hash=? AND s.expires>? AND m.disabled=0";
    const result = await store.client.execute({ sql, args: agent ? [hash(bearer)] : [hash(works!), Math.floor(Date.now() / 1000)] }).catch(() => null);
    if (result?.rows[0]) return { id: `member:${result.rows[0].id}`, credential: "member" };
  }
  throw new LabError(401, "Sign in with an approved Lab key or an existing Eidos member credential.");
}

function activeGrants(): { id: string; sha256: string; expiresAt: string }[] {
  try {
    const values = JSON.parse(process.env.EIDOS_TEST_ACCESS_GRANTS || "[]");
    return Array.isArray(values) ? values.filter(g => /^[\w-]{3,80}$/.test(g.id) && /^[a-f0-9]{64}$/.test(g.sha256) && Date.parse(g.expiresAt) > Date.now()) : [];
  } catch { return []; }
}

export async function signIn(request: Request, store = guidedStore()) {
  const user = await principal(request, store);
  const raw = request.headers.get("Authorization")?.slice(7) || "";
  // Member credentials remain backed by their existing cookie/session store;
  // bearer member keys are not copied into a second non-revocable session.
  if (user.credential === "member") return { user, cookie: null };
  const grant = activeGrants().find(g => `pilot:${g.id}` === user.id);
  const expires = Math.min(Date.now() + 8 * 3600000, grant ? Date.parse(grant.expiresAt) : Infinity);
  const token = "eidos_session_" + randomBytes(32).toString("base64url");
  await store.put("sessions", "session", hash(token), { owner: user.id, credential: user.credential,
    grantId: grant?.id, grantHash: hash(raw), expires });
  const local = process.env.EIDOS_GUIDED_LOCAL === "1" && !process.env.VERCEL;
  return { user, cookie: `${local ? "eidos_lab_local" : "__Host-eidos_lab"}=${token}; HttpOnly; Path=/; SameSite=Strict; Max-Age=${Math.floor((expires - Date.now()) / 1000)}${local ? "" : "; Secure"}` };
}

export function requireSameOrigin(request: Request) {
  const origin = request.headers.get("Origin");
  const requestURL = new URL(request.url);
  // Next may normalize its internal request URL to localhost. The HTTP Host
  // still identifies the browser-facing origin; do not trust forwarded-host.
  const host = request.headers.get("Host") || requestURL.host;
  if (origin && (new URL(origin).host !== host || new URL(origin).protocol !== requestURL.protocol))
    throw new LabError(403, "Cross-origin writes are not allowed.");
  if (!request.headers.has("Authorization") && !origin && request.headers.get("Sec-Fetch-Site") === "cross-site")
    throw new LabError(403, "Cross-site request rejected.");
}
