# Target probes and durable Sentinel integration — 2026-10-14

The existing owner-private Live Connections Observatory is updated at
https://sentinel-evidence-observatory.bmparent.chatgpt.site (Sites version 15).

This work observes the public URL/IP selected by the operator. It does not attach
Eidos Works traffic, firewall, account or application-log feeds.

## Exact deployed pair

- Runtime source: `f44cf3de72e19fdd0b1cabfa4ae2dafa75dc71ff` on
  `codex/observatory-target-sensors-20261014`, based on PR73
  `5cabf9a612597b279c0705f07aaaa32794dc7548`.
- Immutable Node collector: `f0bf5d067c8ecd44bc4576ba21a60c3e5a314e96`,
  deployment `dpl_DtAG2U9HnHRRVApoUxz2WaM7nCBb` at
  https://eidos-sentinel-23vpq50ej-1brentbm-1876s-projects.vercel.app.
  Later runtime changes concern Worker integration/UI; the deployed collector
  route and its imported collection boundary are unchanged.
- Sites source `f49c165538c04bc5c155d4a49c28e476c0b88470`, version **15**,
  deployment `appgdep_6acfc77a3b1881918b1b9b6f6dd31b2d`, environment revision **6**.
- Compiled Worker SHA-256:
  `c2b4d750882039364f48bf31a5015c45c397e69723c7d51ff8bc9f03b848ff22`.
- Original private audience, model key, collector secret, original collector URL,
  recorded capture and applied migrations were preserved. The new preview has
  the same collector credential scoped only to its own Preview branch. The
  original collector binding is preserved independently for rollback.

## Acceptance ledger

| Requirement | Observed result | State |
|---|---|---|
| Public DNS, TCP, TLS and HTTP measurements | Actual one.one.one.one responses, public numeric address pinning, certificate identity/expiry, separate stage timings | Pass |
| Literal IP behavior | Actual HTTPS 1.1.1.1 measurement; DNS timing explicitly not applicable | Pass |
| Optional page fingerprint | Actual HTTP 200, complete 56,614-byte response fingerprint; content not retained | Pass |
| Optional redirects | Actual HTTP one.one.one.one 301 followed by HTTPS 200, one separately validated redirect | Pass |
| Durable per-metric calibration | New v2 profile reached 13 samples; retained history held 26 HTTPS target observations at readback | Pass |
| Duplicate, race, missing and late-data handling | Real SQLite transactional tests, compare-and-swap race, missing timings excluded, separate failure ratio | Pass, local tests |
| Sol uses new evidence | Actual exact-model completed response; 757 input / 25 output tokens; accepted result and identical cached repeat | Pass |
| Private and request boundaries | Anonymous capability access withheld, cross-origin observation 403, private target 400 | Pass |
| Served source | app.js, target panel, styles and unchanged Brain module match source SHA-256 | Pass |
| Node companion | Loopback origin/Host and durable API tests; path parsing and command-boundary tests | Pass, local tests |
| Actual ICMP/traceroute in this host | Both system commands absent; actual runtime returns unavailable | Unavailable |
| Rendered desktop/mobile interactions | Control-browser capability unavailable; no browser or physical-device acceptance claimed | Not tested |

The full functional hosted sequence spans the v2 integration deployment and the
final UI wording deployment. Final served asset hashes match version 15. Endpoint
and model algorithms are identical across those two deployments.

## Checks

- **136 Node tests** passed, including the existing Observatory regression suite.
- **13 existing literal-IP collector tests** passed.
- **30 Python tests** passed.
- Next.js production build and TypeScript passed, including the new target route.
- Canonical Brain parity passed 954 frames (receipt retained); core math unchanged.
- Compiled Worker ESM validation and syntax/whitespace checks passed.
- Five deterministic synthetic integration scenarios compare raw Sentinel flags,
  the operational output and the same simple measured-deviation baseline.

## Measured-deviation gate

A quiet-stream replay exposed frequent internal Sentinel changes on steady and
small-jitter inputs. Those are internal numerical dynamics, not evidence of an
unusual network. The v2 adapter retains raw scores while an operational numerical
flag requires both a Sentinel change and departure from the preceding 16-sample
median beyond the greater of six scaled MADs and a minimum timing/ratio bound.

The two quiet scenarios produced **0 operational flags** across 88 evaluated
samples each. Each controlled step produced eight flagged transition frames;
rolling baseline adaptation subsequently accepts the new level. The simple
measured-deviation baseline matched those operational results. These finite
checks do **not** demonstrate added Sentinel benefit or security detection quality.
Explicit DNS/certificate/status/page events remain separately labeled observations.

## Preserved failures and limits

The first bounded-page test exposed an end callback overwriting a truncated-body
flag; the callback now preserves completed cleanup state. A timer boundary exposed
an existing timeout classification race; explicit deadline state now retains the
failed reservation classification. Their behavior regressions pass.

Local Next build initially rejected an out-of-root dependency symlink. Local
dependencies were copied without that symlink; the explicit monorepo root supports
the shared collection-boundary import. Vercel request validation rejected the
first Preview environment write before its branch existed and the literal
`target=preview` deployment argument; corrected native calls succeeded. No access
policy was relaxed. A tool interruption lost an intermediate proof export; the
final run saved progress incrementally and its complete final receipt is retained.

Collection remains foreground-only. Hidden tabs pause. Retained analysis is not
an unattended scheduler. No attacks, internal traffic, visitor streams, general
prediction skill, calibrated incident probability or real-device acceptance is
claimed. Network path commands require the operator's Node 24+ local companion
and installed operating-system tools. The existing Python/Qwen launcher remains.

## Evidence and rollback

Receipts, test logs, source identities, actual redirect check, local path runtime
result and the synthetic comparison are in `evidence/2026-10-14-target-probes/`.
Evidence contains bounded public test measurements, not credentials or page bodies.

Rollback: deploy existing Sites version **12**,
`appgprj_6ace6f7f4a5481918adb34a5cb366de3~appgver_99a1705f061c819181c2cd6534b30e6c`.
It ignores the new target-probe flags and uses the preserved original collector
binding. Migration 0003 is additive and can remain. The original recorded capture
and separate Showcase candidate are preserved. Main and unrelated release gates
were not merged or promoted by this private app update.
