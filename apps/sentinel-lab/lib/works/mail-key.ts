/** The owner-selected alias is limited to the isolated Playground TEST preview. */
export function worksMailKey(source: Record<string, string | undefined>): string | undefined {
  if (source.RESEND_API_KEY) return source.RESEND_API_KEY;
  if (source.VERCEL_ENV !== 'preview' || source.EIDOS_PLAYGROUND_DELIVERY_MODE !== 'test' ||
      source.PUBLIC_SITE_URL !== 'https://eidosworks-playground-test-20261015.pages.dev') return undefined;
  return source.RESEND_playground_preview;
}
