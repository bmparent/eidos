# October 22, 2026 dependency security update

Brent requested checking Eidos Works after the October 21 PHP disclosure.
Baseline main is `34c3d21509dfa1017705244e0d42a6eb7bc7dff9`; Vercel's production
alias `eidos-sentinel-lab.vercel.app` is observed at that source in deployment
`dpl_8Uu5eVF597W6tqiTSGWoesomjqZy`. Provider metadata identifies Next.js/Node
24.x. The Works API route declares `runtime = 'nodejs'`. The complete repository
tree contains no PHP source or Composer manifest/lockfile. No direct exposure
to PHP2Shell (CVE-2026-109618/109619) was identified in this application.

The lockfile separately contained four vulnerable package entries: Next.js
(critical), sharp, source-map-js and undici (high). Next.js is updated from
16.3.4 to 16.3.8, and transitive packages resolve to sharp 0.35.5,
source-map-js 1.2.2 and undici 7.30.1. `npm audit --package-lock-only` reports
zero known advisories after the update. No forced audit upgrade was used.

The critical ImageResponse advisory requires attacker-controlled SVG values in
the Node `next/og` implementation. No ImageResponse/next/og use was found here.
The image optimizer SSRF advisory requires remotePatterns, which this app does
not configure. Updating avoids retaining the affected package versions even
where no exploitable application path was identified.

References:

- https://www.php.net/archive/2026.php
- https://github.com/vercel/next.js/security/advisories/GHSA-vcvr-r3jv-pc5j
- https://github.com/vercel/next.js/security/advisories/GHSA-cjq9-62q9-8jv4

Local validation: typecheck (`npm run lint`), 16 JavaScript plus 41 TypeScript
tests, and `npm run build` pass. Existing tests include relay authentication,
request boundaries, account ownership and payment entitlement enforcement.
The production alias's health route returns 200; a direct anonymous member
request returns 401 with no-store/nosniff. No live writes, payment/email/model
calls or attempted exploitation were performed.

The paired site review covers its root application, Quote Desk and Wellway.
Candidate SHAs, exact-head CI and deployment receipts are recorded on the
security PRs. No feature flags, bindings, secrets, DB schemas, account IDs or
payment records change; pending account/Playground/Observatory branches remain
separate. Local validation is not production deployment or comprehensive
penetration testing. Provider internals and unrelated research applications
are outside this review. A clean registry scan cannot rule out unknown bugs.

The prior production SHA above is the rollback identity; prefer a forward
correction to returning to affected dependencies. Existing held account and
payment acceptance gates are not accepted by this dependency-only change.

## Same-day advisory follow-up

A fresh registry audit after the morning release flagged Next.js 16.3.8 for
GHSA-2464-fhr4-w9xw (image optimization) and GHSA-p3m5-59mp-jv5r (Server
Actions denial of service), published October 22 at 15:52 and 15:50 UTC.
The official GitHub advisory records identify 16.4.1 as patched; this follow-up
pins the available 16.4.2 release and retains sharp 0.35.5. The already-patched
sharp version satisfies the image advisory's documented workaround; an
exploitable application path has not been established by the package scan.

Baseline for this follow-up is production main
`a69daf7afdc3f8856dd2a1b03e76d9d755aee3b2`. Only the Next.js manifest/lockfile
and this record change. No held Playground/account code, provider setting,
feature flag, credential or database migration is included.

Fresh install, TypeScript check, all 16 JavaScript and 41 TypeScript tests,
and the Next production build pass. The lockfile audit now returns zero known
advisories. Exact-head CI, deployment and production readback remain separate
requirements recorded on the follow-up PR.

- https://github.com/vercel/next.js/security/advisories/GHSA-2464-fhr4-w9xw
- https://github.com/vercel/next.js/security/advisories/GHSA-p3m5-59mp-jv5r
