import { createHash, timingSafeEqual } from "node:crypto";

export function isOperatorAuthConfigured() {
  return Boolean(process.env.EIDOS_OPERATOR_TOKEN?.trim());
}

export function authorizeOperator(request: Request) {
  const expected = process.env.EIDOS_OPERATOR_TOKEN?.trim();
  if (!expected) throw new Error("OPERATOR_AUTH_NOT_CONFIGURED");
  const header = request.headers.get("Authorization") || "";
  const supplied = header.startsWith("Bearer ") ? header.slice(7).trim() : "";
  const expectedHash = createHash("sha256").update(expected, "utf8").digest();
  const suppliedHash = createHash("sha256").update(supplied, "utf8").digest();
  if (!supplied || !timingSafeEqual(expectedHash, suppliedHash)) throw new Error("OPERATOR_AUTH_REQUIRED");
}
