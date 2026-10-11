# Paid templates TEST candidate

This additive candidate extends the existing Eidos Works shop and its existing `eidos_orders` / Stripe-event ledger. Cinematic Starter keeps its $29 price, URL, original ZIP, receipt behavior and refund handling. Eidos Brain research files are untouched.

New template sales require `EIDOS_TEMPLATE_TEST_ENABLED=true`, a Stripe test key, shop configuration, the private archive service, a non-production site origin, and an exact edition ID in `EIDOS_TEMPLATE_VERIFIED_EDITIONS`. The canonical `eidos-works.com` host and all its subdomains fail closed. Draft server prices are $99 developer source / $129 WordPress. These are TEST candidate prices, not approved live prices. Builder pilots always remain unavailable through this candidate API.

## Packaging and installation gate

Store current and retained immutable buyer ZIPs in this server-only directory:

```text
apps/sentinel-lab/private/templates/<switchboard|tideglass|matter|nightjar>/<version>/<developer|wordpress>.zip
```

Current WordPress packages and Switchboard developer source use version 1.0.1; Nightjar developer source uses 1.0.3. Tideglass and Matter developer source remain at 1.0.0. The archive importer validates each package's explicit version and refuses to overwrite different bytes at an existing path. The preview bundle includes all eight current packages and every retained version, including the six replaced 1.0.0 archives and Nightjar developer 1.0.1/1.0.2. The actually purchased Switchboard developer 1.0.0 archive remains unchanged. Existing order snapshots continue to retrieve their original version; a catalog update does not rewrite purchases.

Never put paid ZIPs in Next `public/` or the website's public demo tree. Next file tracing includes these files in the private Works API function bundle. The `templateArchives` adapter only accepts allowlisted product/version/edition paths. It reads bytes and computes SHA-256. A purchase snapshots the path, hash, price, edition and version in `eidos_template_orders`. If a later deployment changes the bytes at that path, old orders fail closed; restore the original archive, then publish the update at a new version path. Retain every sold version in future bundles.

Before adding an edition to `EIDOS_TEMPLATE_VERIFIED_EDITIONS`, retain its written buyer instructions, clean-install result, content-edit result, browser/main-action result, archive checksum and declared platform version. WordPress packages do not become available merely because a ZIP exists. Squarespace and GoDaddy previews do not establish native installation.

## Database and deployment

Apply the additive `0004_paid_templates.sql` migration only to the confirmed validation database for preview. It creates side tables and indexes; it does not replace or delete legacy orders, accounts or purchases. Run from the repository root:

```powershell
npm --prefix apps/sentinel-lab run works:migrate
npm --prefix apps/sentinel-lab run lint -- --incremental false
npm --prefix apps/sentinel-lab test
npm --prefix apps/sentinel-lab run build
```

`works:migrate` consumes existing dedicated Works database configuration. Inspect the destination before running it; never assume a default database is a preview database. For this task, the available authorized provider snapshot names `eidos-works-validation-vercel-icfg-j5glcbpipda5txwa0a43dmjx.aws-us-east-1.turso.io`. A provider connection or an old environment file alone does not prove the new deployment is correctly paired.

Migration must precede deployment, including when the new template flag is off: the additive shop wrappers look up the new side tables when distinguishing legacy orders. This candidate uses `EIDOS_DATABASE_BINDING_PREFIX=EIDOS_TEMPLATES_TEST` and explicitly named URL/token bindings for that validation database. Reapplying updated migration 0004 creates the additional payment-failure table idempotently and preserves existing rows.

Deploy only as a preview under the existing merchant/project. Keep production checkout, production price records, production domains and production migrations unchanged. API state must confirm the actual backend URL, source SHA, database hostname and site origin used by the final preview.

## API contract

| Route | Request | Result |
| --- | --- | --- |
| GET `/api/shop/catalog` | none | TEST readiness and per-edition availability; builder pilots unavailable |
| POST `/api/shop/checkout` | `productId`, `editionId`, `acceptTerms`, `challenge`, optional 64-hex `attempt` | Stripe TEST checkout URL |
| POST `/api/shop/status` | `receipt` | purchase state, product/edition/version, guide, mail state, expiring download token |
| POST `/api/shop/download` | `receipt`, `downloadToken` | private purchased ZIP, original SHA-256 response header |
| POST `/api/shop/recover` | `email`, `challenge` | identical generic response for matching/nonmatching addresses |
| POST `/api/shop/redeem` | one-time `token` | fresh receipt, rotating the old receipt |

Product IDs are `<slug>-developer-v1` or `<slug>-wordpress-v1`; edition IDs are `developer` / `wordpress`. A slug plus matching edition is also accepted. Client price, amount, version, archive key, currency and Stripe price-ID fields are rejected. Unknown products and mismatched editions are rejected. An `attempt` identifies one checkout; retry the same attempt after an uncertain response. It cannot switch product, edition or account.

New template success redirects go to `/shop/start#receipt=...`. The receipt is never treated as payment proof. New receipt access expires after 24 hours; recovery links last 30 minutes and are single-use. Status issues a download token valid for ten minutes, bound to the exact order. Requests remain POST and responses use private/no-store headers. Existing Cinematic Starter receipts keep their established behavior.

Status distinguishes financial ledger `status` from display `paymentState`. Pending attempts report `pending`, `expired` after checkout expiry, or terminal `failed` after `checkout.session.async_payment_failed`. Failed attempts retain their order receipt but never receive download access or receipt email; a later/replayed paid event does not revive that failed attempt. Completed/refunded orders report their existing paid/refunded state.

## Fulfillment and recovery

The existing signed Stripe webhook handles new template sessions. It checks TEST mode, server-snapshotted amount/currency, payment status, order/session/client reference, product, edition and version. Unpaid/delayed sessions remain pending; only a later paid event fulfills. Async failure is journaled without entitlement. New template Checkout Sessions use configured merchant payment methods and do not enable methods in the Dashboard. Their request pins the existing paired shop contract version `2025-06-30.basil`; legacy Cinematic requests retain their established version behavior. Any enabled delayed methods still need hosted acceptance.

The isolated candidate may explicitly set `EIDOS_TEMPLATE_TEST_CHALLENGE=true` with Cloudflare's documented public TEST widget/secret. The code still calls siteverify and requires its TEST-result metadata, exact candidate Pages hostname, TEST Stripe key, and checkout/recovery action. Cloudflare TEST responses use `example.com` and omit action; that exception cannot apply to canonical hosts, other actions or live keys. `EIDOS_LOCAL_TEST` is never set remotely. These keys establish integration flow, not live bot protection. See [Cloudflare's testing documentation](https://developers.cloudflare.com/turnstile/troubleshooting/testing/).

The existing Resend transactional service sends setup/recovery mail independently of newsletter consent. Provider message IDs and mail status are saved; no credentials or response bodies are exposed to buyers. Receipt mail has stable content and provider idempotency across webhook retries. If mail fails, paid state remains durable and the webhook returns a retryable error. The browser can still recover using its unexpired receipt. Recovery responses avoid disclosing whether another address owns a purchase. A signed-in buyer cannot use another account's purchase receipt or recovery link; guest capability links remain available as the existing shop supports guest purchases.

Refunds/disputes revoke future retrieval while retaining the order snapshot and financial/event trail. A late paid event cannot restore a revoked payment. Previously downloaded files cannot be reclaimed. Existing approved refund policy must be used; this code does not decide whether a refund is owed or create refunds automatically.

Legacy paired hooks share the Stripe event ledger. A signed TEST refund/dispute whose payment intent matches a stored template purchase therefore reconciles the template side table even when an older hook already recorded that event. This repeat operation is idempotent, is bound to the stored payment intent, and cannot grant access. LIVE-labelled events for matching template purchases are rejected. Unknown payment intents cannot select a template order through metadata.

## Validation and limits

`tests/works-templates.test.ts` uses real in-memory libSQL with controlled Stripe/Resend responses. It checks server prices, TEST/production guard, unsupported editions, idempotent attempts, payment signatures and mismatches, delayed state, duplicate webhook, stable email retries, archive immutability, expired links, single-use recovery, account and order isolation, refund retention, late-paid events, and byte-for-byte legacy Cinematic ZIP preservation.

These are local controlled tests. They do not prove a real hosted Stripe checkout, provider webhook delivery, received inbox message, production behavior, WordPress installation, Squarespace transfer or GoDaddy editor coexistence. Retain separate hosted receipts before claiming those gates passed.

Rollback: redeploy the prior paired preview revision or set `EIDOS_TEMPLATE_TEST_ENABLED=false`. Preserve the additive schema, paid archives and order rows. Do not drop tables or remove old archives to roll back UI changes. No production promotion is authorized by this candidate.
