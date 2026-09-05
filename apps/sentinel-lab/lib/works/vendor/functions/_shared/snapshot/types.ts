export const PRIMARY_GOALS = [
  'more leads',
  'better trust',
  'modernize design',
  'improve storefront conversions',
  'improve local SEO',
  'improve AI/search readiness',
] as const

export const STYLE_PREFERENCES = [
  'clean premium',
  'bold modern',
  'warm local business',
  'high-end editorial',
  'tech-forward',
  'playful storefront',
] as const

export type PrimaryGoal = (typeof PRIMARY_GOALS)[number]
export type StylePreference = (typeof STYLE_PREFERENCES)[number]

export interface SnapshotIntake {
  websiteUrl: string
  businessName: string
  industry: string
  primaryGoal: PrimaryGoal
  stylePreference: StylePreference | ''
  biggestIssue: string
  email: string
  consent: true
}

export type SnapshotPriority = 'high' | 'medium' | 'low'

export interface SnapshotOpportunity {
  title: string
  whyItMatters: string
  suggestedFix: string
  priority: SnapshotPriority
}

export interface SnapshotReport {
  overallImpression: string
  businessTypeGuess: string
  primaryConversionGoal: string
  uiUxOpportunities: SnapshotOpportunity[]
  seoOpportunities: SnapshotOpportunity[]
  suggestedHomepageStructure: Array<{
    section: string
    purpose: string
    sampleCopy: string
  }>
  suggestedTitleTag: string
  suggestedMetaDescription: string
  aiSearchReadiness: Array<{
    title: string
    recommendation: string
  }>
  nextStepRecommendation: {
    label: string
    reason: string
    cta: string
  }
  imagePrompt: string
}

export interface CapturedPage {
  finalUrl: string
  title: string
  metaDescription: string
  headings: string[]
  linkLabels: string[]
  visibleText: string
}

export type SnapshotStatus =
  | 'created'
  | 'checkout_created'
  | 'paid'
  | 'processing'
  | 'complete'
  | 'failed'

export interface SnapshotRecord {
  version: 1
  requestId: string
  resultToken: string
  status: SnapshotStatus
  intake: SnapshotIntake
  createdAt: string
  updatedAt: string
  paidAt?: string
  processingStartedAt?: string
  completedAt?: string
  stripeSessionId?: string
  stripeCheckoutUrl?: string
  captureNote?: string
  report?: SnapshotReport
  imageStored?: boolean
  imageMediaType?: 'image/jpeg' | 'image/png' | 'image/webp'
  imageBase64?: string
  imageNote?: string
  publicError?: string
}

export interface KVNamespaceLike {
  get(key: string): Promise<string | null>
  put(
    key: string,
    value: string,
    options?: {
      expirationTtl?: number
    },
  ): Promise<void>
  delete(key: string): Promise<void>
}

export interface SnapshotEnv {
  SNAPSHOT_STORE?: KVNamespaceLike
  SNAPSHOT_PUBLIC_ENABLED?: string
  SNAPSHOT_DEV_BYPASS_PAYMENT?: string
  SNAPSHOT_PRICE_CENTS?: string
  PUBLIC_SITE_URL?: string
  PUBLIC_SNAPSHOT_EMAIL?: string

  STRIPE_SECRET_KEY?: string
  STRIPE_WEBHOOK_SECRET?: string
  STRIPE_PRICE_ID_SNAPSHOT?: string

  OPENAI_SNAPSHOT_API_KEY?: string
  OPENAI_API_KEY?: string
  OPENAI_TEXT_MODEL?: string
  OPENAI_IMAGE_MODEL?: string

  GOOGLE_APPS_SCRIPT_WEBHOOK_URL?: string
  COMMAND_CENTER_SHARED_SECRET?: string
}

export interface PagesFunctionContext {
  request: Request
  env: SnapshotEnv
  waitUntil(promise: Promise<unknown>): void
}
