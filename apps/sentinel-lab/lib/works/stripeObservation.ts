type Order = { id: string; status: string; payment_intent?: string | null; archive_present?: number | null };
type StripeObject = { id?: string; status?: string; payment_intent?: string | null; amount_refunded?: number };

async function getStripe(path: string, key: string, fetcher: typeof fetch) {
  const response = await fetcher(`https://api.stripe.com/v1/${path}`, {
    headers: { authorization: `Bearer ${key}`, accept: 'application/json' },
    signal: AbortSignal.timeout(7000),
  });
  if (!response.ok) throw new Error(`stripe_http_${response.status}`);
  return response.json() as Promise<{ data?: StripeObject[]; has_more?: boolean }>;
}

/** Read-only, bounded TEST observation. Never return the credential or customer data. */
export async function observeStripeTest(key: string | undefined, kits: Order[], playground: Order[], fetcher: typeof fetch = fetch, now = new Date()) {
  const observedAt = now.toISOString();
  const base = { source: 'Stripe TEST API', environment: 'test', observedAt,
    staleAfter: new Date(now.getTime() + 300000).toISOString() };
  if (!key?.startsWith('sk_test_') && !key?.startsWith('rk_test_'))
    return { ...base, status: 'not_configured', errorCode: 'test_key_required', data: null };
  try {
    const all = [...kits.map(x => ({ ...x, kind: 'kit' })), ...playground.map(x => ({ ...x, kind: 'playground' }))];
    const references = [...new Set(all.map(x => x.payment_intent).filter((x): x is string => Boolean(x && /^pi_[A-Za-z0-9]+$/.test(x))))].slice(0, 30);
    const [intents, refunds, disputes] = await Promise.all([
      Promise.all(references.map(ref => getStripe(`payment_intents/${encodeURIComponent(ref)}`, key, fetcher))),
      getStripe('refunds?limit=100', key, fetcher),
      getStripe('disputes?limit=100', key, fetcher),
    ]);
    const byId = new Map(intents.map(x => { const item = x as StripeObject; return [item.id, item.status] as const; }));
    const refunded = new Set((refunds.data || []).filter(x => x.status === 'succeeded').map(x => x.payment_intent));
    const disputed = new Set((disputes.data || []).map(x => x.payment_intent));
    const rows = all.map(order => {
      const ref = order.payment_intent || null;
      const providerStatus = ref ? byId.get(ref) || 'unknown' : 'no_intent_reference';
      const reversalMissing = (order.status === 'refunded' || order.status === 'revoked') && ref &&
        !refunded.has(ref) && !disputed.has(ref) && !refunds.has_more && !disputes.has_more;
      const mismatch = (order.status === 'paid' && providerStatus !== 'succeeded') || reversalMissing;
      return { id: order.id, kind: order.kind, ledgerStatus: order.status, providerStatus,
        refundObserved: ref && refunded.has(ref) ? 'observed' : refunds.has_more ? 'not_in_window' : 'not_observed',
        disputeObserved: ref && disputed.has(ref) ? 'observed' : disputes.has_more ? 'not_in_window' : 'not_observed',
        archivePresent: order.kind === 'playground' ? Boolean(order.archive_present) : null,
        mismatch: Boolean(mismatch) };
    });
    return { ...base, status: references.length < new Set(all.map(x => x.payment_intent).filter(Boolean)).size || refunds.has_more || disputes.has_more ? 'partial' : 'healthy', errorCode: null,
      data: { rows, referencedIntents: references.length, totalLedgerRows: all.length,
        checkedIntents: intents.length, refundWindow: { limit: 100, hasMore: Boolean(refunds.has_more) },
        disputeWindow: { limit: 100, hasMore: Boolean(disputes.has_more) }, webhookDelivery: 'unavailable',
        mismatchCount: rows.filter(x => x.mismatch).length } };
  } catch (error) {
    return { ...base, status: 'unavailable', errorCode: error instanceof Error && /^stripe_http_\d+$/.test(error.message) ? error.message : 'stripe_read_failed', data: null };
  }
}
