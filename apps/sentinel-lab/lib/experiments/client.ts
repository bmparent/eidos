export class LabRequestError extends Error {
  constructor(message: string, public status: number, public jobId?: string) { super(message); }
}

export async function requestJson<T>(url: string, init: RequestInit = {}, timeout = 30_000): Promise<T> {
  const signal = init.signal ? AbortSignal.any([init.signal, AbortSignal.timeout(timeout)]) : AbortSignal.timeout(timeout);
  let response: Response;
  try {
    response = await fetch(url, { ...init, cache: "no-store", signal });
  } catch (error) {
    if (signal.aborted && !init.signal?.aborted) throw new LabRequestError("The request timed out. A launched job may still be running; check its receipt before launching again.", 0);
    throw error;
  }
  const body = await response.json().catch(() => null);
  if (!response.ok || !body || typeof body !== "object") {
    throw new LabRequestError(body?.detail || body?.error || `The server returned HTTP ${response.status} without a readable receipt. Try checking status again.`, response.status, typeof body?.jobId === "string" ? body.jobId : undefined);
  }
  return body as T;
}

export const JOB_ID = /^rd-[a-f0-9]{12}-[a-f0-9]{8}$/;

export function downloadJson(value: unknown, filename: string) {
  const href = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(href), 0);
}
