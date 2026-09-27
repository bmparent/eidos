import { cleanString, isRecord, withTimeout } from './http'
import type { SnapshotEnv, SnapshotRecord } from './types'

const STRIPE_TIMEOUT_MS = 15_000
const STRIPE_SIGNATURE_TOLERANCE_SECONDS = 300

export interface StripeCheckoutResult {
  id: string
  url: string
}

export interface StripeEvent {
  id: string
  type: string
  data: {
    object: Record<string, unknown>
  }
}

export function snapshotPriceCents(env: SnapshotEnv) {
  const parsed = Number.parseInt(env.SNAPSHOT_PRICE_CENTS ?? '500', 10)
  return Number.isFinite(parsed) && parsed >= 50 && parsed <= 100_000 ? parsed : 500
}

export function isStripeConfigured(env: SnapshotEnv) {
  return Boolean(cleanString(env.STRIPE_SECRET_KEY, 500) && cleanString(env.STRIPE_WEBHOOK_SECRET, 500))
}

async function validateConfiguredPrice(env: SnapshotEnv, secretKey: string, priceId: string) {
  const response = await withTimeout(STRIPE_TIMEOUT_MS, (signal) =>
    fetch(`https://api.stripe.com/v1/prices/${encodeURIComponent(priceId)}`, {
      headers: { authorization: `Bearer ${secretKey}` },
      signal,
    }),
  )
  if (!response.ok) throw new Error('stripe-price-unavailable')

  const price = (await response.json()) as unknown
  if (!isRecord(price)) throw new Error('stripe-price-invalid')
  const valid =
    price.id === priceId &&
    price.active === true &&
    price.type === 'one_time' &&
    cleanString(price.currency, 12).toLowerCase() === 'usd' &&
    price.unit_amount === snapshotPriceCents(env) &&
    (price.recurring === null || price.recurring === undefined)
  if (!valid) throw new Error('stripe-price-invalid')
}

function publicSiteUrl(env: SnapshotEnv) {
  const fallback = new URL('https://eidos-works.com')
  try {
    const candidate = new URL(cleanString(env.PUBLIC_SITE_URL, 500) || fallback.toString())
    const local = candidate.hostname === 'localhost' || candidate.hostname === '127.0.0.1'
    if (candidate.protocol !== 'https:' && !(local && candidate.protocol === 'http:')) return fallback.origin
    return candidate.origin
  } catch {
    return fallback.origin
  }
}

export async function createStripeCheckout(
  env: SnapshotEnv,
  record: SnapshotRecord,
): Promise<StripeCheckoutResult> {
  const secretKey = cleanString(env.STRIPE_SECRET_KEY, 500)
  if (!secretKey || !isStripeConfigured(env)) throw new Error('stripe-not-configured')

  const siteUrl = publicSiteUrl(env)
  const values = new URLSearchParams()
  values.set('mode', 'payment')
  values.set('payment_method_types[0]', 'card')
  values.set('client_reference_id', record.requestId)
  values.set('customer_email', record.intake.email)
  values.set('success_url', `${siteUrl}/snapshot/success?token=${encodeURIComponent(record.resultToken)}&session_id={CHECKOUT_SESSION_ID}`)
  values.set('cancel_url', `${siteUrl}/snapshot/start?request=${encodeURIComponent(record.requestId)}&payment=cancelled`)
  values.set('metadata[snapshot_request_id]', record.requestId)
  values.set('metadata[snapshot_result_token]', record.resultToken)
  values.set('payment_intent_data[metadata][snapshot_request_id]', record.requestId)
  values.set('line_items[0][quantity]', '1')

  const priceId = cleanString(env.STRIPE_PRICE_ID_SNAPSHOT, 240)
  if (priceId) {
    await validateConfiguredPrice(env, secretKey, priceId)
    values.set('line_items[0][price]', priceId)
  } else {
    values.set('line_items[0][price_data][currency]', 'usd')
    values.set('line_items[0][price_data][unit_amount]', String(snapshotPriceCents(env)))
    values.set('line_items[0][price_data][product_data][name]', 'Eidos Snapshot')
    values.set(
      'line_items[0][price_data][product_data][description]',
      'AI-assisted homepage concept with practical UX, SEO, and AI-search-readiness recommendations.',
    )
  }

  const response = await withTimeout(STRIPE_TIMEOUT_MS, (signal) =>
    fetch('https://api.stripe.com/v1/checkout/sessions', {
      method: 'POST',
      headers: {
        authorization: `Bearer ${secretKey}`,
        'content-type': 'application/x-www-form-urlencoded',
        'idempotency-key': `eidos-snapshot-checkout-${record.requestId}`,
      },
      body: values.toString(),
      signal,
    }),
  )

  if (!response.ok) throw new Error('stripe-checkout-failed')
  const body = (await response.json()) as unknown
  if (!isRecord(body)) throw new Error('stripe-checkout-invalid')
  const id = cleanString(body.id, 240)
  const url = cleanString(body.url, 1200)
  if (!/^cs_[A-Za-z0-9_]+$/.test(id)) throw new Error('stripe-checkout-invalid')
  try {
    const checkoutUrl = new URL(url)
    if (
      checkoutUrl.protocol !== 'https:' ||
      !(checkoutUrl.hostname === 'stripe.com' || checkoutUrl.hostname.endsWith('.stripe.com'))
    ) {
      throw new Error('stripe-checkout-invalid')
    }
  } catch {
    throw new Error('stripe-checkout-invalid')
  }
  return { id, url }
}

function bytesToHex(bytes: ArrayBuffer) {
  return [...new Uint8Array(bytes)].map((value) => value.toString(16).padStart(2, '0')).join('')
}

function constantTimeEqual(left: string, right: string) {
  if (left.length !== right.length) return false
  let mismatch = 0
  for (let index = 0; index < left.length; index += 1) {
    mismatch |= left.charCodeAt(index) ^ right.charCodeAt(index)
  }
  return mismatch === 0
}

function parseStripeSignature(header: string) {
  let timestamp = 0
  const signatures: string[] = []

  for (const segment of header.split(',')) {
    const [key, value] = segment.trim().split('=', 2)
    if (key === 't') timestamp = Number.parseInt(value ?? '', 10)
    if (key === 'v1' && /^[a-f\d]{64}$/i.test(value ?? '')) signatures.push((value ?? '').toLowerCase())
  }
  return { timestamp, signatures }
}

export async function verifyStripeSignature(rawBody: string, signatureHeader: string, secret: string) {
  const { timestamp, signatures } = parseStripeSignature(signatureHeader)
  const now = Math.floor(Date.now() / 1000)
  if (!timestamp || !signatures.length || Math.abs(now - timestamp) > STRIPE_SIGNATURE_TOLERANCE_SECONDS) return false

  const encoder = new TextEncoder()
  const key = await crypto.subtle.importKey(
    'raw',
    encoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  )
  const digest = await crypto.subtle.sign('HMAC', key, encoder.encode(`${timestamp}.${rawBody}`))
  const expected = bytesToHex(digest)
  return signatures.some((signature) => constantTimeEqual(signature, expected))
}

export function parseStripeEvent(value: unknown): StripeEvent | null {
  if (!isRecord(value) || !isRecord(value.data) || !isRecord(value.data.object)) return null
  const id = cleanString(value.id, 240)
  const type = cleanString(value.type, 160)
  if (!/^evt_[A-Za-z0-9_]+$/.test(id) || !type) return null
  return { id, type, data: { object: value.data.object } }
}
