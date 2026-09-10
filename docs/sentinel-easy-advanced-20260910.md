# Sentinel Easy / Advanced: implementation and release gates

Date: 2026-09-10. Base: `1eb0a1769896a59e7637823af06e18781cb1fe77`, draft PR #52 (`codex/sentinel-guided-analysis-20260908`). This change is stacked on that candidate, not an independent production release.

## What this patch changes

- `/` becomes an Easy-first four-stage journey: Add data -> Check its meaning -> Run experiment -> Understand results. An in-place Easy/Advanced switch preserves selections and uses the same owner-scoped API and worker.
- `/advanced` preserves the previous GuidedLab component, charts, deterministic evidence questions, saved comparisons and scoped telemetry. `/research` remains unchanged. No historical data, research receipts, engine source, Works account code or commerce configuration is changed.
- Easy accepts the existing upload and public-source workflows. Kaggle catalog search is visible on the data-entry screen and distinguishes live results from curated fixtures. Links are constructed only from validated owner/dataset references.
- **Kaggle is not yet one-click ingestion in Easy.** The explicit supported bridge is: open the dataset, download a supported file within the pilot limit, upload it. The existing versioned Kaggle research importer remains available. Do not label the catalog action as importing data.
- Confirmation keeps chronology, time zone and entity interpretation explicit. Labels, identifiers and constants are excluded from suggested numeric features. No random file-order chronology or silent sample is introduced.
- Advanced currently exposes supported mapping and run rules: selected measurements, units, session grouping, missing-value handling, ordering policy, named target, forecast horizon and late-match window. **Arbitrary custom threshold expressions and business-rule alerts are not implemented in this patch.**
- Result summaries validate dataset identity and input hash, retain missing metrics as missing, disclose baseline losses and separate unusualness from consequence. Source references resolve against original record IDs/passages, with raw source timestamps and byte hashes. Broken references remain visible as unverified.
- Finding cards distinguish observed evidence, possible relevance, unknown cause, uncertainty and next action. The presentation does not invent a causal diagnosis or domain-specific story. Forecasts are explicitly described as analysis issuances/historical replay, not proof of real-time future prediction.
- Session-scoped private state is cleared on sign-out; outstanding requests are aborted and stale responses are fenced. Submit exclusion and immutable-request retry keys help avoid duplicate UI submissions. The existing server remains responsible for authentication, bounded parsing, ownership, admission and idempotency. Client preflight is not a security boundary.

## Local verification in this session

The complete repository could not be cloned in this container: outbound GitHub DNS resolution failed. Source inspection and publication used the GitHub connector. Local checks used Node 22.16.0 and the installed TypeScript compiler; the repository requires Node >=24 for its actual application runtime.

Observed locally:

1. 25 pure-policy unit tests pass. Coverage includes label exclusion, the feature limit, immutable mapping copies, temporal prerequisites, confirmed targets, nonfinite/fractional/empty horizon rejection, missing metrics, original record identity, ambiguous/unresolved references, document passages, no manufactured timestamps, no-findings != safety, invalid results, source hash mismatch, baseline losses/ties, non-invented accuracy, safe Kaggle links, provider rejection wording and request-key canonicalization.
2. Standalone strict TypeScript checking passes for `lib/guided/experience.ts`.
3. TypeScript syntax transpilation passes for the new component and both route files. CSS parses with PostCSS.

The tests were transpiled to an external temporary directory for the local Node runtime; the committed `.test.ts` uses the repository's existing `tsx` test stage. No new dependencies are introduced.

**NOT RUN here:** the full repository lint/build/test suite; rendered Next.js/browser validation; production sign-in; real upload -> provider -> Torch -> durable result; Kaggle authenticated import; provider budget recovery; full final-SHA compute acceptance. Browser plugin not available; Python Playwright is installed, but the actual React/Next dependencies and runnable full checkout are not present. A syntax check is not rendered UI evidence. No screenshot or simulated result is claimed as a live acceptance receipt.

Keep the PR draft. Do not merge or promote based only on these local checks or a Ready preview.

## Existing release blockers that remain in force

The inspected #52 release guide reports HTTP 402 `payment_required` during new Vercel Sandbox creation. It does not establish the presently exhausted meter or its reset date, and this session did not retry provider allocation or change billing. An authorized operator must verify current access/quota, restore allocation if necessary, and repeat the final source SHA on the hosted path.

The same guide records failed frozen operational qualification for all four Eidos variants. The default synthetic final MAE was 0.9866 versus persistence 1.0326, but precision was 0.60 and false alerts/day 16.7033 against a declared maximum of 1. The hosted service fixture favored persistence (2.2658 vs Eidos 2.8254). These are existing archived report values, not new experiments run in this session. Zero research gates advance.

A functioning user journey and a useful detector are separate acceptance decisions. Neither may be substituted for the other.

## Next implementation: complete the easy data bridge and safe advanced rules

### Kaggle ingestion

Reuse the existing authenticated provider path rather than teaching the generic public-URL importer to scrape arbitrary Kaggle pages. Before allocation, show the exact dataset reference, pinned version, selected file, listed size and available license/terms metadata. Ask the user to select one supported file; reject archives that exceed compressed/expanded byte and file-count limits. Keep keys server-side, owner-scoped and redacted. Private/terms-gated datasets require their normal authorization, not bypasses. Persist a source receipt (provider, ref, version, filename, download time, original byte hash) before parsing into the same guided dataset store. No whole-archive downloads or silent truncation to fit 2 MB. Explicitly handle missing keys, permissions, 404, 429, interrupted downloads and expired links. A catalog result is never proof of a loaded dataset.

### Structured business-rule alerts

Do not add a text box that pretends unsupported settings are executed. Implement a bounded, versioned rule schema first: allowed numeric comparisons; named feature and unit; entity/session scope; minimum duration in event-time seconds; missing/gap policy; optional cooldown. No `eval`, generated code, shell commands, SQL, arbitrary regex or arbitrary remote fetches. Validate before enqueue and again in the worker. Hash the exact rules into the immutable run request/receipt. Report rule-triggered findings separately from model anomaly findings, with original record IDs, times, measured value, declared limit and rule ID/version. A user threshold crossing is not a model discovery, a calibrated probability or proof of cause. Unsupported rule fields must fail rather than be ignored.

### Plain-English findings and forecasts

Use a versioned evidence contract for each finding: observed phenomenon, source/time/entity, actual and reference values with units, detector/version, scoring rule, evidence IDs, competing hypotheses with supporting/contradicting evidence, uncertainty and next action. Unknown cause is a valid answer. A generic list of possible explanations is not a diagnosis. Keep deterministic summaries available without any LLM. A later language-model explanation must operate on a bounded redacted evidence bundle and may not alter scores, invent citations or follow instructions contained in uploaded data.

For predictions, require a named target, issue time, horizon, unit, target time, predictor identity, pre-observation commitment and outcome state. Distinguish prospective issuance from historical replay. When quality or time prerequisites fail, say "No reliable forecast available" rather than manufacturing a narrative. Identify any selected persistence/other baseline by its real method name.

## Engine work: qualify usefulness without reusing the consumed final set

Preserve all historical successes, failures and frozen artifacts. Do not tune on the previously consumed final evaluation and present it as a fresh holdout.

1. Choose a specific operational task and intended action (for example, identifying sustained machine/service degradation from numeric telemetry). Define event labels, entity boundaries, operating exposure, alert burden and meaningful lead time before development. The example is a candidate task, not a claim of deployed coverage.
2. Freeze a new versioned development protocol. Isolate labels and later periods. Compute preprocessing and score thresholds from permitted past data only. Issue each prediction before its target is exposed. Select predictors using resolved historical losses only. Include tests that changing future values cannot change earlier issuances.
3. Compare on the same causal timeline and target units against persistence, an appropriate seasonal baseline when justified, robust residual/rule baselines and the existing Eidos causal profile. Do not add a weaker baseline just to obtain a win. Preserve failure cases and interval widths/coverage, not just MAE.
4. Reduce alert burden on development data using explicit incident grouping, temporal persistence and threshold policies. Keep raw point alerts alongside grouped incidents. Do not hide misses or reduce displayed false positives by changing the evaluation definition after seeing final results.
5. Before an untouched final run, freeze acceptance values and resource budgets. At minimum report event precision/recall, false alerts per unit of operating time, detection delay, target error, empirical interval coverage/width, abstention rate, runtime and memory. Report uncertainty from suitable time/entity blocks, not an independence assumption for adjacent samples. Where labels are absent, do not claim measured precision/recall.
6. Use one untouched final evaluation, archive raw inputs/code/config/outputs with hashes, and retain an explicit pass/fail/inconclusive decision. A baseline winning is a legitimate outcome. Experimental mechanisms remain separately labeled until their declared gates pass.

## Required integration and hosted acceptance

Run under the repository's supported Node version, from a full checkout of the exact PR head:

```sh
npm ci --prefix apps/sentinel-lab --no-audit --no-fund
npm test --prefix apps/sentinel-lab
npm run lint --prefix apps/sentinel-lab
npm run build --prefix apps/sentinel-lab
```

Run the existing runner and browser matrix from `docs/sentinel-guided-release.md`. Preserve old detailed-workspace coverage at `/advanced` and add Easy-root coverage rather than silently dropping tests that used the previous root selectors. Required scenarios: authenticated CSV and document flows, each supported adapter, time/missing/entity confirmation, mode-switch state preservation, a real completed run, exact source drill-down, zero findings, baseline loss, invalid/missing output, network interruption, repeat click, queued recovery, failed provider creation, cancel fencing, reload/reopen, account A/B isolation, sign-out during polling and late responses, revoked grants, desktop and narrow mobile, keyboard navigation, reduced motion, downloads and feedback. The Kaggle direct-import cases stay explicitly not implemented until the bridge above exists.

Use only an isolated preview namespace and authorized test accounts. Do not copy production customer data into QA. Record exact Git SHA, job IDs, source/result hashes, actual engine identity, browser receipts and every blocked/failed check. Do not change paid plans, relax proof gates, delete Drive artifacts, migrate unrelated accounts/commerce, or promote production as part of testing. Production release requires the existing separate authorization after these gates pass.
