export const MAX_JSON_REQUEST_BYTES = 12_000

const responseHeaders = {
  'content-type': 'application/json; charset=utf-8',
  'cache-control': 'no-store, max-age=0',
  'x-content-type-options': 'nosniff',
  'x-robots-tag': 'noindex, nofollow',
}

export function json(value: unknown, init: ResponseInit = {}) {
  return new Response(JSON.stringify(value), {
    ...init,
    headers: { ...responseHeaders, ...(init.headers ?? {}) },
  })
}

export class RequestBodyError extends Error {
  constructor(
    readonly status: number,
    readonly publicMessage: string,
  ) {
    super(publicMessage)
  }
}

export async function readTextBodyLimited(request: Request, maxBytes: number) {
  const declaredLength = Number(request.headers.get('content-length') ?? 0)
  if (Number.isFinite(declaredLength) && declaredLength > maxBytes) {
    throw new RequestBodyError(413, 'The request is too large.')
  }

  if (!request.body) return ''
  const reader = request.body.getReader()
  const decoder = new TextDecoder('utf-8', { fatal: false })
  const chunks: string[] = []
  let bytes = 0

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      bytes += value.byteLength
      if (bytes > maxBytes) {
        await reader.cancel('request-too-large')
        throw new RequestBodyError(413, 'The request is too large.')
      }
      chunks.push(decoder.decode(value, { stream: true }))
    }
    chunks.push(decoder.decode())
    return chunks.join('')
  } finally {
    reader.releaseLock()
  }
}

export async function readJsonBody(request: Request, maxBytes = MAX_JSON_REQUEST_BYTES): Promise<unknown> {
  const contentType = request.headers.get('content-type')?.toLowerCase() ?? ''
  if (!contentType.includes('application/json')) {
    throw new RequestBodyError(415, 'This endpoint accepts JSON requests only.')
  }

  const body = await readTextBodyLimited(request, maxBytes)

  try {
    return JSON.parse(body) as unknown
  } catch {
    throw new RequestBodyError(400, 'The request body is not valid JSON.')
  }
}

export function optionsResponse() {
  return new Response(null, {
    status: 204,
    headers: {
      'access-control-allow-methods': 'GET, POST, OPTIONS',
      'access-control-allow-headers': 'content-type, stripe-signature',
      'access-control-max-age': '86400',
    },
  })
}

export function publicFailure(message: string, status = 500) {
  return json({ ok: false, message }, { status })
}

export async function withTimeout<T>(
  milliseconds: number,
  operation: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), milliseconds)
  try {
    return await operation(controller.signal)
  } finally {
    clearTimeout(timeout)
  }
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

export function cleanString(value: unknown, maxLength: number) {
  return typeof value === 'string' ? value.trim().replace(/\s+/g, ' ').slice(0, maxLength) : ''
}
