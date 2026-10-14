# Coordinated Observatory evidence views — October 13, 2026

Status: **local and isolated hosted fixture acceptance passed; actual sensor/model acceptance unavailable**. This is a presentation milestone, not evidence of detection quality or accepted GPT-6.1 Sol execution. Base: draft PR73, `00d49ff2e8ef74931503fb7a27046857228d0b04`. Candidate branch: `codex/showcase-observatory-20261013`.

The existing application is `apps/sentinel-observatory/live`. The Lab's separate reservoir trajectory component is not the active network Observatory. The original owner-private Site, its durable sensor, protected literal-IP collector and interpretation configuration remain unchanged.

## Completed task

Import evidence, select a record, inspect its topology, switch to its time series, narrow the time window or host, read provenance and coverage in the drawer, then download the retained records and selection as JSON. Time, host and record filters drive all three views. A selected receipt remains inspectable if a filter hides it, with an explicit explanation. Fixed time bounds survive repeated source receipts and view changes.

Record counts are deduplicated by stable normalized evidence ID and bounded at 2,000. The series contains only occupied record bins; an empty interval means unknown coverage, not zero network traffic. Imported history, explicit synthetic fixtures, receiving, partial, stale, connecting, paused, reconnecting and unauthorized sources have distinct descriptions. Model failure preserves source facts.

`/evidence-fixture` is an explicitly synthetic view demonstration, separate from `/` and from real collection. It uses reserved documentation addresses and an example DNS name, invokes no provider, and exports its fixture provenance. Nothing is inserted into the sensor store.

## Validation and evidence

Commands run from the repository root:

```sh
node --test apps/sentinel-observatory/live/collector.test.mjs apps/sentinel-observatory/live/brain.test.mjs apps/sentinel-observatory/live/browser-interpreter.test.mjs apps/sentinel-observatory/live/security-events.test.mjs apps/sentinel-observatory/live/hosted-sensor.test.mjs apps/sentinel-observatory/live/sol-interpreter.test.mjs apps/sentinel-observatory/live/evidence-view.test.mjs
python -m unittest discover -s apps/sentinel-observatory/live -p 'test_*.py' -v
node apps/sentinel-observatory/live/verify_brain_parity.mjs
python apps/sentinel-observatory/live/build_worker.py --output artifacts/showcase-observatory-20261013/worker/index.mjs
python apps/sentinel-observatory/live/local_server.py --port 8770
node apps/sentinel-observatory/live/browser-evidence-qa.mjs
```

The browser script accepts `EIDOS_PLAYWRIGHT_MODULE` when Playwright is supplied by the host runtime and `EIDOS_BROWSER_CHANNEL=chrome` for installed Chrome. Browser plugin was not listed; regular Playwright used installed Chrome after the bundled runtime lacked its expected headless-shell revision. No browser or project dependency was installed.

- 122 Node cases passed, including seven new coordination/retention/source-state cases.
- Canonical parity passed: 954 frames, 41,022 comparisons at the existing tolerances. Numerical engine source is unchanged.
- Python: 27 passed, one existing symlink rejection case could not run because Windows denied symlink creation (`WinError 1314`). The original failing log is retained. This is an environment limit, not a passing case.
- Browser: inspect `artifacts/showcase-observatory-20261013/browser-acceptance.json` for the final exact checks and browser version. Desktop 1440×1000 and phone-width 390×844, reduced motion, GPU explicitly disabled. These are browser-width tests, not physical-device or accessibility certification.
- Download: `fixture-export.json`; visual receipts: `observatory-desktop-fixture.png`, `observatory-phone-fixture.png`.
- Worker build and source hashes: `build-receipt.json`; syntax and ESM build validation are retained with the candidate.

To use the export independently, open `fixture-export.json` in a JSON viewer. It includes all retained normalized records, per-source mode, interpretations, and the selected event/window/host/kind/view. It is a portable receipt, not a native Suricata/Zeek log, cryptographic signature, or live sensor connection.

## Proof Logic + Meaning

### Goal reached

The local coordinated evidence inspection gate passed. Isolated hosted fixture acceptance passed on the exact deployment below. Actual sensor/model acceptance remains unavailable on this independent candidate. No scientific proof-readiness score advances.

### Previous state

The active network panel already preserved source records, topology, imports, bounded streaming and interpretation. Its list filter did not constrain the topology, there was no coordinated time-series/window control, and repeated IDs inside a single batch could survive the first append.

### Technical logic utilized

Presentation derives from a single retained event set and shared selection. Stable-ID deduplication precedes sorting and bounded retention. All views use the same intersection of time, host and record kind. Source status separates successful feed refresh from source event freshness. The detector, original measurements, model catalogue, usage limits, credentials and sensor collection are untouched.

### Math / scoring logic

`R = newest_2000(unique_by_evidence_id(receipts))`.

`V = {e in R | start <= time(e) <= end AND host_matches(e) AND kind_matches(e)}`.

Each occupied time bin reports `count_b = sum(1 for e in V if bin(e) = b)`. Empty bins are omitted because missing evidence cannot establish zero traffic. Recorded security-event counts preserve the source label; they do not estimate attack probabilities. The unchanged engine passed its existing numeric parity tolerances (`1e-7 + 1e-8 * abs(reference)`).

### Philosophical meaning

Explanation before automation; restraint before alarm. A connected UI is useful only when the operator can revisit the underlying receipt and see the limits of its source.

### Why this is better

Topology, time series, list and drawer now agree on the selected evidence and scope. Duplicate delivery cannot inflate visible counts, filters no longer imply different populations, and missing sources retain an explicit unknown state.

### North-star connection

Eidos Brain is a self-monitoring streaming intelligence codec: it learns live streams, compresses predictable behavior, preserves meaningful anomalies, monitors internal state and emits human-readable incident receipts. This milestone strengthens inspectable receipts and reproducible operator interpretation. It does not newly prove stream learning, compression, anomaly preservation or advantage over detectors/compressors.

### Evidence

The test logs, parity receipt, browser receipt, exported fixture, screenshots and Worker hashes are under `artifacts/showcase-observatory-20261013`. The journal and plain-language analysis are under `docs/proof_runs/2026-10-13`.

### Remaining uncertainty

No new actual sensor observation, provider inference, intrusion-quality assessment, packet capture, two-account acceptance, physical-device performance or production promotion is claimed. The original live sensor/model acceptance records remain separate. Hosted candidate source bindings and the environment-limited Python case remain explicit gates.

## Rollback and follow-up

Rollback source identity: `00d49ff2e8ef74931503fb7a27046857228d0b04`. Close the isolated candidate without changing the original private Site or its persisted evidence. Review on the exact hosted candidate, then independently authorize any later promotion. Next narrow step: attach an isolated real sensor scope and verify source transitions with actual observations; provision model access only through the established credential workflow.

## Isolated hosted receipt

Owner-private candidate: https://sentinel-showcase-candidate-20261013.bmparent.chatgpt.site; demonstration: `/evidence-fixture`. Site source `6342acf48e95acc3c399db117f0e08456bba7a2d`; deployment `appgdep_6acec682295c8191ad967510bf5fcf01`; Worker SHA256 `368ba1729acd3a5dd932227d1717e41026acd3b3bec1c15cda97150dd7befbf4`. New project `appgprj_6acec3e0cc4c8191899f69048a9e7eb8` has owner-only custom access, one allowed owner and zero external visitors. It is a separate hosted private Site, not a promotion of the existing Observatory.

Seven actual hosted rendered checks passed in Chrome155.0.8059.39: fixture rendering, synchronized selection, fixed window/view changes,390px reduced-motion/GPU-disabled layout, explicit unavailable sensor/model bindings, unauthenticated denial (HTTP401), and no runtime errors. Exact result and screenshots are `hosted-browser-acceptance.json`, `hosted-observatory-desktop.png`, and `hosted-observatory-phone.png`. The sensor endpoint returned503 with unavailable storage; GPT-6.1Sol is not configured. No database, model secret or IP-relay credential was copied.

Two failed harness attempts remain preserved: the first omitted the required Bearer prefix in the Sites authorization header; the second expected403 although the current auth layer correctly returned401. Both were test-harness defects and were corrected without changing access policy or product source. Site packaging initially failed because Windows selected WSL bash; the same pushed source was successfully packaged with Git Bash using the supported package script.
