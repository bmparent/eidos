# Controlled Observatory reliability proof

This is a local engineering harness, separate from customer operation and the
Lab's sealed research proof. The production collector rejects private targets.
Only this test directory maps logical public names to loopback sockets. Its
public fixture key is not an operating credential and is excluded from both
the Worker build and the Next collector dependency trace.

From a clean repository root with Node 24+ and Python 3.11+:

```sh
node apps/sentinel-observatory/live/fault-harness/freeze.mjs
node --test --test-concurrency=1 apps/sentinel-observatory/live/*.test.mjs
python -m unittest discover -s apps/sentinel-observatory/live -p 'test_*.py' -v
node apps/sentinel-observatory/live/fault-harness/run.mjs --out artifacts/observatory/corpus
node apps/sentinel-observatory/live/fault-harness/compare-detectors.mjs --out artifacts/observatory/detectors
node apps/sentinel-observatory/live/fault-harness/serve.mjs
```

Open `http://127.0.0.1:8788`. The banner identifies synthetic evidence; the demo
refuses unregistered names/paths and does not call a real model. Each run writes
raw cases or streams, scores, failures and code/input hashes. DNS answers and
blackhole sockets are explicit simulations; HTTP/TLS/refusal tests use actual
local sockets. Normal socket deadlines allow real TLS setup on loaded hosts;
the production limit remains twelve seconds across addresses/hops/methods.

The corpus contains 72 cases across twelve families. Initial evaluation failures
have been inspected: subsequent runs are regression evidence, not a fresh final
held-back evaluation. Do not edit locked inputs or tune thresholds against those
cases; create a new version for future qualification. Wall-clock timings/UUIDs
vary. The scoring tolerance and detector configuration are versioned separately.

`browser-qa.mjs` uses an existing Playwright installation. Pass
`--playwright-module /absolute/path/to/playwright` and optional
`--browser /absolute/path/to/browser` when it is not installed in this project;
no browser dependency is required for the numerical/collector tests. Browser QA
checks 1440x1000 and 390x844, reduced motion, stage rendering, pause, keyboard
focus, reload/history, export and explicit HEAD→GET coverage. The two viewports
use distinct logical destinations to respect the collector cooldown.

`verify-hosted.mjs` reads the selected Site's service credential from hidden
stdin, never a command argument or file. By default it checks served hashes,
anonymous denial, boundary/schema rejection and fabricated evidence denial.
`--public-test example.com` adds exactly one bounded public-example observation
and retained-history readback. Model calls remain zero. Its optional browser
path sends authorization only to the exact Site origin and checks its empty
authenticated page; it is separate from synthetic interactive QA.

The assembler produces reports, journal, action-review packet, acceptance matrix,
progress SVG/HTML, pilot plan and checksum-verified mounted Drive copy:

```sh
python apps/sentinel-observatory/live/fault-harness/assemble-proof.py --out artifacts/observatory --corpus artifacts/observatory/corpus --detectors artifacts/observatory/detectors --browser artifacts/observatory/browser
```

Use `--dry-run-drive` to create an explicit skipped-copy receipt. The available
blinded packet is a single deterministic arm; a genuine Sol-ranked corpus arm
and independently validated rubric/human scores are still required. No model
self-grading or simulated provider output qualifies explanation usefulness.

## Proof Logic + Meaning

The local gate strengthens reproducible measurements and readable explanations.
Negotiated TLS is captured before cleanup; partial stages and bounded address
attempts remain separate evidence. Every attempt deadline is at most the shared
deadline. TP/(TP+FP), TP/(TP+FN), false events per scored benign target-day and
whole-stream bootstrap intervals expose detector failures. Sentinel plus the
gate matched the gate alone; no engine policy was changed. Restraint before
alarm and revisitable truth guide this work. See the measured report and raw
receipts for before/after evidence. Unattended scheduling, tenants, incidents,
delivery, compression/generalization, real pilot coverage and commercial value
remain unproven.
