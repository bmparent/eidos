Preserve negotiated TLS and completed DNS/TCP/TLS measurements when a later probe stage fails. Add at most three validated-address attempts under one shared deadline, explicit HEAD 405/501 coverage with opt-in GET fallback, retained attempt provenance, and consistent stage explanations. Unknown TLS/peer outcomes stay null.

The test-only socket harness has 72 labeled cases across 12 families, locked input/scoring hashes, raw measurements, storage/export/replay checks, and causal gate/Sentinel/EWMA/CUSUM comparisons. Its loopback adapter and fixture credentials are excluded from the Worker and the production collector dependency trace. Core reservoir/RLS/Sentinel policy and Lab gates are unchanged.

Validation: 152 Node tests; clean-checkout 151-test suite; 31 Python passed and one Windows symlink-privilege skip; 954 parity frames / 41,022 comparisons; Next build and fixture-separation trace; desktop/mobile interactive local QA. Private Sites version 17 passes 16 hosted checks plus authenticated desktop/mobile page QA. One real public-example probe returned HTTP 200 with TLS 1.3 and authoritative retention. A separate manual gpt-6.1-sol call was accepted (736 input / 24 output tokens); its identical repeat was cached with the same provider receipt. This establishes execution/cache behavior, not usefulness. No new model key or automatic loop was created.

The first corpus result was 70/72; repairs and clean-checkout regression are 72/72. Original evaluation cases are now consumed. On 24 fixed evaluation streams, Sentinel plus gate and the gate alone both had TP/FP/FN 24/12/2, precision 0.667 and recall 0.923. False incidents, other baselines and stream-cluster intervals are retained. No superiority or commercial-readiness claim.

Runtime commit: `4a9bd8b9da652a1d6587412316e18d97d35d76f8`. Sites source: `0295a79cfea528efa8295a766c64c5d2b3f4524c`, environment revision 8. Immutable Node collector: `dpl_7iTnN6c4qSwsHN5w4U9EtQ7ShyZP`, same runtime commit. Existing collector credential was reused only on this preview branch; owner-only audience, recorded capture, database and unrelated releases remain intact.

This PR is stacked on draft PR79; PR79/PR77 and this PR remain unmerged. Tenant ownership/budgets, durable scheduling, independent observers, incidents/outbox, backup/restore, capacity soak, seven-day pilot, independently scored Sol usefulness and actual commercial costs remain open.

## Proof Logic + Meaning

The collector engineering gate passed; commercial qualification is partial. Previously, cleanup and later-stage errors erased useful measured context. Monotonic capture and additive attempt contracts preserve it: attempt deadline <= shared deadline, and precision = TP/(TP+FP), recall = TP/(TP+FN). The result represents restraint before alarm and revisitable truth. It strengthens reproducible measurement and readable evidence in the self-monitoring streaming-codec north star. It does not prove compression, domain generalization, attacks, unattended operation or Sol usefulness.

Measured report, exact commands, raw observations, failure history, acceptance matrix, journal, progress artifacts and mounted Drive receipt are under `artifacts/proof_runs/2026-10-18/observatory-m1/`. See `apps/sentinel-observatory/live/fault-harness/README.md` and `OPERATIONS-M1-2026-10-18.md` for reproduction, private deployment and rollback.
