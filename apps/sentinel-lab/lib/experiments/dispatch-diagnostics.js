const STAGE_FAILURES = {
  operator_auth: {
    error: "OPERATOR_AUTH_FAILED",
    status: 401,
    detail: "The operator credential could not be verified.",
    retryable: false,
  },
  request_parse: {
    error: "INVALID_EXPERIMENT_REQUEST",
    status: 400,
    detail: "The experiment dispatch body is not valid JSON.",
    retryable: false,
  },
  request_validation: {
    error: "INVALID_EXPERIMENT_REQUEST",
    status: 400,
    detail: "The experiment dispatch request did not satisfy the locked schema.",
    retryable: false,
  },
  sandbox_capacity: {
    error: "SANDBOX_CAPACITY_CHECK_FAILED",
    status: 502,
    detail: "The Lab could not inspect the current Sandbox capacity before dispatch.",
    retryable: true,
  },
  sandbox_allocation: {
    error: "SANDBOX_ALLOCATION_FAILED",
    status: 502,
    detail: "Vercel Sandbox did not allocate compute for the pinned experiment.",
    retryable: true,
  },
  sandbox_bootstrap: {
    error: "SANDBOX_BOOTSTRAP_FAILED",
    status: 502,
    detail: "The Sandbox was allocated, but the pinned runner could not be started.",
    retryable: true,
  },
  runner_dispatch: {
    error: "RUNNER_DISPATCH_FAILED",
    status: 502,
    detail: "The configured execution backend did not accept the pinned experiment.",
    retryable: true,
  },
};

const EXACT_FAILURES = {
  IDEMPOTENCY_KEY_REQUIRED: {
    error: "IDEMPOTENCY_KEY_REQUIRED", status: 400, retryable: false,
    detail: "Refresh the Lab and lock your settings to create a stable launch retry key.",
  },
  IDEMPOTENCY_LOCK_CONFLICT: {
    error: "IDEMPOTENCY_LOCK_CONFLICT", status: 409, retryable: false,
    detail: "This launch key belongs to different settings. Restore its original lock or explicitly prepare a new experiment.",
  },
  ADMISSION_RESERVATION_EXPIRED: {
    error: "ADMISSION_RESERVATION_EXPIRED", status: 409, retryable: false,
    detail: "The launch reservation expired before allocation. Prepare a new experiment; this retry key cannot start another job.",
  },
  OPERATOR_AUTH_NOT_CONFIGURED: {
    error: "OPERATOR_AUTH_NOT_CONFIGURED",
    status: 503,
    detail: "Operator authentication is not configured for this deployment.",
    retryable: false,
  },
  OPERATOR_AUTH_REQUIRED: {
    error: "OPERATOR_AUTH_REQUIRED",
    status: 401,
    detail: "Provide the configured operator credential before dispatching a real-data run.",
    retryable: false,
  },
  RUNNER_CAPACITY_OCCUPIED: {
    error: "RUNNER_CAPACITY_OCCUPIED",
    status: 429,
    detail: "The configured real-data job capacity is occupied. Retry after the active job reaches a terminal state.",
    retryable: true,
  },
  RUNNER_NOT_CONFIGURED: {
    error: "RUNNER_NOT_CONFIGURED",
    status: 503,
    detail: "No resource-qualified execution backend is configured for this deployment.",
    retryable: false,
  },
};

function errorMessage(error) {
  if (error instanceof Error) return error.message || error.name;
  if (typeof error === "string") return error;
  return "Experiment dispatch failed.";
}

function rootCause(error) {
  let current = error;
  const seen = new Set();
  while (current instanceof Error && current.cause instanceof Error && !seen.has(current.cause)) {
    seen.add(current);
    current = current.cause;
  }
  return current;
}

export function redactDispatchDiagnostic(value) {
  return String(value)
    .replace(/(Bearer\s+)[^\s,;]+/gi, "$1[REDACTED]")
    .replace(/\bKGAT_[A-Za-z0-9_-]+\b/g, "[REDACTED_KAGGLE_TOKEN]")
    .replace(/((?:api[_-]?key|authorization|password|secret|token)\s*[=:]\s*)[^\s,;&]+/gi, "$1[REDACTED]")
    .replace(/(https?:\/\/[^/\s:@]+:)[^@\s]+@/gi, "$1[REDACTED]@")
    .replace(/[\r\n]+/g, " ")
    .slice(0, 2_000);
}

function providerFailure(message) {
  if (/\b(?:401|403)\b|unauthori[sz]ed|forbidden|oidc|authentication failed/i.test(message)) {
    return {
      error: "SANDBOX_AUTH_FAILED",
      status: 503,
      detail: "Vercel Sandbox rejected the deployment identity used for compute allocation.",
      retryable: false,
    };
  }
  if (/\b429\b|quota|billing|payment|required plan|resource limit|rate.?limit/i.test(message)) {
    return {
      error: "SANDBOX_RESOURCE_UNAVAILABLE",
      status: 503,
      detail: "Vercel Sandbox compute is unavailable under the current resource or account limits.",
      retryable: true,
    };
  }
  if (/timed?\s*out|timeout|abort(?:ed)?|etimedout/i.test(message)) {
    return {
      error: "SANDBOX_DISPATCH_TIMEOUT",
      status: 504,
      detail: "The Sandbox dispatch operation exceeded its response deadline.",
      retryable: true,
    };
  }
  return null;
}

/**
 * @param {string} stage
 * @param {unknown} cause
 * @param {{ jobId?: string }} [context]
 */
export class DispatchStageError extends Error {
  constructor(stage, cause, context = {}) {
    super(errorMessage(cause), cause instanceof Error ? { cause } : undefined);
    this.name = "DispatchStageError";
    this.stage = stage;
    this.jobId = context.jobId;
  }
}

/**
 * @template T
 * @param {string} stage
 * @param {() => Promise<T>} operation
 * @param {{ jobId?: string }} [context]
 * @returns {Promise<T>}
 */
export async function withDispatchStage(stage, operation, context = {}) {
  try {
    return await operation();
  } catch (error) {
    if (error instanceof DispatchStageError) throw error;
    throw new DispatchStageError(stage, error, context);
  }
}

/**
 * @param {unknown} error
 * @param {{ diagnosticId?: string, stage?: string }} [context]
 */
export function normalizeDispatchFailure(error, { diagnosticId, stage = "runner_dispatch" } = {}) {
  const failureStage = error instanceof DispatchStageError ? error.stage : stage;
  const jobId = error instanceof DispatchStageError ? error.jobId : undefined;
  const cause = rootCause(error);
  const message = errorMessage(cause);
  let normalized = EXACT_FAILURES[message];

  if (!normalized && /^[A-Z0-9_]+_NOT_CONFIGURED$/.test(message)) {
    normalized = {
      error: message,
      status: 503,
      detail: `Required execution setting ${message.replace(/_NOT_CONFIGURED$/, "")} is not configured.`,
      retryable: false,
    };
  }

  if (!normalized && failureStage === "request_validation") {
    normalized = {
      ...STAGE_FAILURES.request_validation,
      detail: redactDispatchDiagnostic(message),
    };
  }

  if (!normalized && failureStage.startsWith("sandbox_")) normalized = providerFailure(message);
  normalized ||= STAGE_FAILURES[failureStage] || STAGE_FAILURES.runner_dispatch;

  return {
    ...normalized,
    diagnosticId,
    stage: failureStage,
    ...(jobId ? { jobId, statusUrl: `/api/experiments/${jobId}` } : {}),
  };
}

/** @param {unknown} error */
export function dispatchDiagnostic(error) {
  const cause = rootCause(error);
  const code = cause && typeof cause === "object" && "code" in cause ? String(cause.code) : undefined;
  const status = cause && typeof cause === "object" && "status" in cause ? Number(cause.status) : undefined;
  const statusCode = cause && typeof cause === "object" && "statusCode" in cause ? Number(cause.statusCode) : undefined;
  return {
    type: cause instanceof Error ? cause.name : typeof cause,
    message: redactDispatchDiagnostic(errorMessage(cause)),
    ...(code ? { providerCode: redactDispatchDiagnostic(code) } : {}),
    ...(Number.isFinite(status) ? { providerStatus: status } : {}),
    ...(Number.isFinite(statusCode) ? { providerStatusCode: statusCode } : {}),
  };
}
