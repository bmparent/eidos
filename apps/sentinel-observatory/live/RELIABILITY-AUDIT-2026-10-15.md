# Observatory reliability audit — 15 October 2026

The Observatory is a useful private endpoint-inspection prototype. It is not yet
qualified as a dependable unattended monitoring or security product. The first
hardening pass below is deployed privately as Sites version 16; this audit does
not promote the Lab's research gates or assert Sentinel superiority.

## What was actually inspected

- Owner-private Site `appgprj_6ace6f7f4a5481918adb34a5cb366de3`, initially version 15.
- All five retained model-job rows, all marked completed with no stored provider
  error code. A completed job is an execution receipt, not a quality verdict.
- All **54** retained endpoint observations across five canonical targets.
  Forty-three contain an HTTP status; eleven contain a connection timeout.
  Reading an operator's existing target records did not attach a Works data feed.
- Three v2 interpretation receipts revalidated through the deployed API, with
  `cached=true` and **zero new provider calls** during this initial audit.
- Current Worker source, Node collector contract, target storage/Brain adapter,
  foreground client, model controller, and Sentinel Lab experiment contracts.
- Recent Site Worker errors: cancelled requests, a missing favicon and expected
  input/origin denials; these do not demonstrate an OpenAI provider failure.
- The reported browser error was not captured in a screenshot or client log.
  The reproduced issues below are confirmed; they may not exhaust what Brent saw.

## Findings and disposition

| Priority | Confirmed finding | Effect | Disposition |
|---|---|---|---|
| P1 | A previously successful v2 ranking fails present-context revalidation because its prompt hash differs | UI suggests rejected model output even though the saved provider run completed | New semantic prompt version v3; explicit `cached_context_changed` reason; facts rebuilt and no silent paid retry |
| P1 | A stable v2 target receipt selected `normal_variation`, whose wording referred to observed changes despite no detected change | Plausible but misleading interpretation | Change explanations are eligible only when a change exists; ordinary responses receive a coverage explanation |
| P1 | Model context omitted DNS/TCP/TLS/HTTP stage states and elapsed failure time | Generic timeout explanation misses the valuable distinction between connection failure and a received HTTP error | Shared factual explanation derives only from retained stages; model receives bounded stage enums/timing; deterministic next check is always supplied |
| P1 | The HTTP response field displayed `timeout` while the timeout had no HTTP status and no response latency | Looks like an HTTP timeout response and hides the measured 12-second wait | HTTP status now says no response; separate probe elapsed and failure-stage explanation |
| P1 | One user-triggered Sol receipt described an Observatory service request, not an entered endpoint | Accurate source summary, poor answer to the monitoring task | Advanced sources/service diagnostics moved behind an explicit disclosure; service-request receipts cannot invoke interpretation from that panel |
| P2 | Incomplete sensor-response ingestion could race its deadline, recording a request-only success | Deadline test failed on the fresh checkout | Incomplete/invalid response reads propagate failure; the existing regression now passes |
| P2 | New Node target-relay receipts were relabeled as Cloudflare HEAD transport | Incorrect provenance in the service diagnostics | Preserve the new relay transport allowlist on write/read |
| P2 | Provider HTTP failures shared one generic code | Credential, rate-limit and service faults hard to distinguish | Bounded distinct codes/messages; provider bodies remain excluded |

The new facts distinguish **a connection timeout** from a received HTTP 408,
successful first-hop redirects from destination responses, and calibration from
an established baseline. They never infer that a firewall blocked the request,
that a public IP must host a web server, or that a failed probe proves an outage.

## Important limits still open

1. **One hosted observation point, foreground collection.** Checks pause in a
   hidden tab and stop when the session closes. This cannot promise continuous
   uptime coverage, incident alerts or target-wide availability.
2. **Collector detail has gaps.** Retained successful TLS records can lose the
   protocol string after socket cleanup. On a later failure, some successful
   partial TLS/network detail is not retained. The explanation keeps the stage
   unknown where that evidence is absent. Repair and deploy the Node collector
   with real failure fixtures before a commercial reliability claim.
3. **One resolved address is tried.** Hostnames select one public address. An
   unhealthy address is not evidence that all addresses are unreachable. Address
   retry and two independent collectors need explicit per-attempt provenance.
4. **HTTP is a narrow check.** A successful HEAD response does not verify page
   rendering, login or checkout. A public IP need not provide HTTP. HEAD 405/501
   needs an explicit method-coverage explanation, not an outage conclusion.
5. **Interpretation utility is unscored.** Sol ranks exactly two code-supplied
   possibilities. It cannot invent facts/citations, but eligibility and ranking
   can still be unhelpful. The two-item requirement can force a weak second note.
   Large concurrent-change contexts can fall back under the 6,000-byte request
   cap; budget/error fallbacks need their own user-experience checks.
6. **No demonstrated Sentinel advantage.** The prior quiet/step evaluation's
   operational flags matched its simple measured-deviation baseline. Numerical
   parity and 145 unit/integration checks do not establish detection usefulness.
7. **Browser acceptance remains open.** This managed environment has no supported
   control-browser capability. No rendered desktop/mobile or physical-device
   acceptance is claimed. Source/hash and HTTP checks are separate evidence.
8. **Source is private operational evidence.** User-selected endpoint records and
   sensor captures are not automatically suitable for portfolio publication.
   Public results should use controlled fixtures or approved, redacted cases.

## One platform, two workspaces

Recommendation: retain **Sentinel Observatory** as the practical experience and
**Sentinel Lab** as the experiment workbench under one Sentinel platform. Share
tested components and evidence contracts, rather than maintaining two unrelated
apps or turning every research control into a customer feature.

| Surface | Main question | Appropriate behavior |
|---|---|---|
| Observatory | What did this endpoint do, what changed, and what should I check next? | Simple target setup, actual stage measurements, freshness, calibrated change notes, next action, retained evidence |
| Lab | Does this method improve on simpler alternatives, and under which conditions? | Locked data, engine/profile selection, replay, fault injection, competing baselines, blind evaluation, reproducible artifacts |
| Eidos Works public Lab/portfolio | What has been demonstrated and what remains uncertain? | Approved recordings, exact versions, methodology, baseline comparison, limitations and reproducible case notes |

These are different engines today: the Observatory uses a portable scalar
32-reservoir Sentinel V3 profile, while the Lab can dispatch the full isolated
Torch Eidos 0.4.7.02 workflow. A shared name does not make their capabilities equal.
Use a versioned adapter contract recording engine identity and evidence class.
Promote a tested adapter into the Observatory only after its own acceptance.

The shared contract should cover target and observation-point identity,
measurement stages/units/missing values, sample times, collector health,
engine/profile version, past-only baseline and evidence references,
interpretation role/context version, and experiment export/replay. Keep customer
tenant/auth/storage and research execution credentials separate.

No research/Lab core or Works production route is changed by this audit.

## Next validation experiment — proposed, not a completed benchmark

Begin controlled tests now to establish reliability; public claims follow their
results. Freeze a protocol before tuning, then hold back cases for final review.

1. Build at least 60 labeled cases across at least 12 families: ordinary service,
   DNS failure, unsafe DNS rejection, TCP timeout/refusal, TLS identity/expiry,
   delayed headers, HEAD rejection, HTTP application errors, redirects, changing
   page content, missing/stale evidence, and collector/model/cache failures.
2. Require zero invented statuses/timings/causes and zero wrong-source attachment
   on those cases; every factual claim must resolve to the correct retained
   evidence. Include cancellation, cache and multi-target interaction cases.
3. Blindly score whether the explanation helps a reviewer choose the appropriate
   next check. Compare deterministic diagnosis alone with Sol-ranked notes; an
   accepted JSON response is not the success metric.
4. Compare Sentinel plus the operational gate against the gate alone, EWMA and
   CUSUM on identical locked streams. Report false alarms, event precision/recall,
   detection delay, missing-data behavior and confidence intervals, including
   negative results. Do not tune against the final held-back cases.
5. Run an authorized multi-day pilot from two independent observation points,
   recording coverage and failures. Continuous monitoring requires a separately
   implemented scheduler, bounded concurrency, health checks and alert receipts.
6. Publish a redacted case study only with its exact source/engine/configuration,
   input and artifact hashes, baseline results and limitations. Research results
   remain research results until the commercial acceptance passes.

## Change and validation record

- Backend runtime commit: `49e765ccd77e3d66d2fe9a6946d65215eff5b16b`, branch
  `codex/observatory-reliability-audit-20261015`, based on PR77 head
  `1a618d61a696aa93bf7c6b677866bc63e5523d8f`.
- Sites version **16**, source `4c9c2fd349e232cdcdf61d881320f0f4ec587bb5`,
  deployment `appgdep_6ad0ee0ddf6081919b0b0570a43afede`, environment revision **6**.
- Worker SHA-256 `0fa2cca2691684d6058d197dbab0871a1c413f98743cfc10cefde03605e1133b`.
- Same immutable Node collector as version 15; no key, environment, audience,
  database migration, unattended task, Brain math or collector route changed.
- **145 Node checks** and **30 Python checks** pass. Final changed interpretation
  paths also pass their focused rerun. The initial fresh-checkout deadline
  failure is retained as evidence rather than hidden by rerunning until green.
- All 54 historical observations pass the new safe preparation rules; all eleven
  known TCP timeout records retain their failure stage. No endpoint URL enters
  the prepared model context.
- ESM artifact validation, JavaScript syntax and whitespace checks pass.
- Hosted final API/model verification passed: one v3 provider call, 722 input / 24 output tokens, accepted stage-specific result, identical cached repeat, served source hashes matching, and anonymous capability access withheld. The receipt is recorded in
  `evidence/2026-10-15-reliability/verification.json`.
- Original recorded capture SHA-256 is unchanged:
  `a96bada34c90cc6a13cf93682f1b99be030470020407a63367794016b364bb8f`.
- Rollback: redeploy Sites version **15**,
  `appgprj_6ace6f7f4a5481918adb34a5cb366de3~appgver_41b84267493c8191a0791182f4c4f339`.
  Existing additive storage and original collector bindings remain compatible.

Decision: **private prototype hardening delivered; commercial/security readiness
not established**. The next work should be the reliability corpus and measured
usefulness experiment, followed by shared contracts between the two workspaces.
