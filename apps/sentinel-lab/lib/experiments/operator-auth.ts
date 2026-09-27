import { createHash, timingSafeEqual } from "node:crypto";

type TestAccessGrant = {
  id: string;
  sha256: string;
  expiresAt: string;
};

function activeTestAccessGrants(now = Date.now()) {
  const value = process.env.EIDOS_TEST_ACCESS_GRANTS?.trim();
  if (!value) return [];
  try {
    const parsed = JSON.parse(value) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((grant): grant is TestAccessGrant => {
      if (!grant || typeof grant !== "object") return false;
      const candidate = grant as Record<string, unknown>;
      const expiresAt =
        typeof candidate.expiresAt === "string"
          ? Date.parse(candidate.expiresAt)
          : Number.NaN;
      return (
        typeof candidate.id === "string" &&
        /^[a-zA-Z0-9_-]{3,80}$/.test(candidate.id) &&
        typeof candidate.sha256 === "string" &&
        /^[a-f0-9]{64}$/.test(candidate.sha256) &&
        Number.isFinite(expiresAt) &&
        expiresAt > now
      );
    });
  } catch {
    return [];
  }
}

function digest(value: string) {
  return createHash("sha256").update(value, "utf8").digest();
}

export function isOperatorAuthConfigured() {
  return Boolean(
    process.env.EIDOS_OPERATOR_TOKEN?.trim() || activeTestAccessGrants().length,
  );
}

export function authorizeOperator(request: Request) {
  const expected = process.env.EIDOS_OPERATOR_TOKEN?.trim();
  const testGrants = activeTestAccessGrants();
  if (!expected && !testGrants.length)
    throw new Error("OPERATOR_AUTH_NOT_CONFIGURED");
  const header = request.headers.get("Authorization") || "";
  const supplied = header.startsWith("Bearer ") ? header.slice(7).trim() : "";
  const suppliedHash = digest(supplied);
  const operatorMatch = expected
    ? timingSafeEqual(digest(expected), suppliedHash)
    : false;
  const testGrantMatch = testGrants.some((grant) =>
    timingSafeEqual(Buffer.from(grant.sha256, "hex"), suppliedHash),
  );
  if (!supplied || (!operatorMatch && !testGrantMatch))
    throw new Error("OPERATOR_AUTH_REQUIRED");
}
