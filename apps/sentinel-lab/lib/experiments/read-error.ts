import { dispatchDiagnostic } from "./dispatch-diagnostics.js";

export function experimentReadError(error: unknown, operation: "status" | "artifact") {
  const message = error instanceof Error ? error.message : "";
  const known: Record<string, number> = {
    "Experiment job not found.": 404,
    "Artifact not found.": 404,
    "Invalid experiment job ID.": 400,
    OPERATOR_AUTH_REQUIRED: 401,
    OPERATOR_AUTH_NOT_CONFIGURED: 503,
    RUNNER_NOT_CONFIGURED: 503,
    SANDBOX_SESSION_TRANSITIONING: 503,
  };
  const isKnown = Object.hasOwn(known, message);
  const status = isKnown ? known[message] : 502;
  const diagnosticId = crypto.randomUUID();
  const code = isKnown ? message : `EXPERIMENT_${operation.toUpperCase()}_UNAVAILABLE`;
  console.error(JSON.stringify({ event: `experiment_${operation}_failed`, diagnosticId, code, status, diagnostic: dispatchDiagnostic(error) }));
  return Response.json({
    error: code,
    detail: status >= 500 ? "The experiment service could not complete this lookup. Retry the status check or download; this lookup does not establish that the engine failed." : code,
    diagnosticId,
    retryable: status >= 500,
  }, { status, headers: { "Cache-Control": "no-store", ...(status >= 500 ? { "Retry-After": "5" } : {}) } });
}
