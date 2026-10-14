"""Validate and document this finite acceptance run, then mirror its receipts.

Run from the repository root. This never reads API credentials or changes the
model, hosted database, access policy, or deployment.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd().resolve()
OUT = ROOT / "apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance"
DOCS = ROOT / "docs/proof_runs/2026-10-13"
PROGRESS = ROOT / "artifacts/progress/sentinel-observatory-acceptance-20261013"
RUN = "sentinel-observatory-sol-acceptance-20261013"
MARKER = f"<!-- {RUN} -->"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def js(path, value):
    write(path, json.dumps(value, indent=2, ensure_ascii=False))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dated_document(path, content):
    if path.exists():
        previous = path.read_text(encoding="utf-8")
        prefix = previous.split(MARKER, 1)[0].rstrip()
        write(path, prefix + "\n\n" + MARKER + "\n" + content)
    else:
        write(path, MARKER + "\n" + content)


def main():
    assert Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()).resolve() == ROOT
    provider = read("actual-sol-provider-receipt.json")
    response = provider["response"]
    p = response["analysis"]["provenance"]
    external = read("final-external-state.json")
    browser = read("browser-checks.json")
    local = read("local-after-fixes/local-checks.json")
    parity = read("local-after-fixes/brain-parity.stdout.log")
    concurrency = read("hosted-concurrency-cooldown.json")
    source = read("authoritative-input-comparison.json")
    final_export = read("sensor-ui-export-final.json")
    assert p["model_ran"] is True and p["output_accepted"] is True
    assert p["provider_model"] == p["model_requested"] == "gpt-6.1-sol"
    assert p["execution_verified"] and response["cached"]
    assert source["event_id"] == response["source_event_id"] == p["input_evidence_ids"][0]
    assert source["linked_evidence_id"] == response["evidence_id"]
    assert all(f["evidence_ids"] == [source["event_id"]] for f in response["analysis"]["facts"])
    assert all(h["support"] == [source["event_id"]] for h in response["analysis"]["hypotheses"])
    assert source["status"] == 301 and source["network"]["tls_verified"]
    assert concurrency["first"]["body"]["analysis"]["provenance"]["output_accepted"]
    assert concurrency["concurrent_duplicate"]["status"] == 409
    assert concurrency["different_evidence_during_cooldown"]["status"] == 429
    assert external["access"]["owner_only"] and external["environment"]["revision"] == 5
    assert external["jobs"]["count"] == 2 and external["jobs"]["total_reserved_usd"] == 0.10
    assert all(c["passed"] for c in read("hosted-negative-after-key.json")["checks"])
    assert all(c["status"] == "passed" for c in local["checks"])
    assert parity["status"] == "PASS" and parity["field_comparisons"] == 41022
    assert browser["authenticated_owner"] and not browser["console"]
    unresolved = [c for c in browser["checks"] if c["status"] != "passed" and c["check"] != "sensor_control_focus_before_fix"]
    assert not unresolved, unresolved
    core_paths = ["apps/sentinel-observatory/live/brain.mjs", "apps/sentinel-observatory/live/collector.mjs", "apps/sentinel-observatory/live/sol-interpreter.mjs", "apps/sentinel-observatory/live/hosted-sensor.mjs", "apps/sentinel-observatory/live/migrations"]
    assert not subprocess.check_output(["git", "diff", "00d49ff2e8ef74931503fb7a27046857228d0b04", "HEAD", "--", *core_paths])
    stamp = datetime.now(timezone.utc).isoformat()
    deployment = read("deployment-sol-configured.json")
    product, site = deployment["product_runtime_sha"], deployment["sites_source_sha"]
    native_counts = {}
    for event in final_export["events"]:
        native_counts[event["kind"]] = native_counts.get(event["kind"], 0) + 1
    limits = read("capabilities-configured.json")["hosted_interpretation"]["limits"]
    config = {"run_id": RUN, "task_date": "2026-10-13", "task_timezone": "America/New_York", "model": "gpt-6.1-sol", "reasoning_effort": "low", "strict_structured_output": True, "manual_only": True, "provider_store": False, "limits": limits, "private_site_project_id": deployment["project_id"], "secret_names_only": [e["key"] for e in external["environment"]["entries"]]}
    js(OUT / "config.json", config)
    skip = "Real OS symlink creation is skipped on this Windows host (WinError 1314). Relative/directory rejection and symlink-metadata rejection before open pass."
    tests = {"node_passed": 115, "python_ran": 30, "python_passed": 29, "python_skipped": 1, "python_skip_reason": skip, "parity_frames": 954, "parity_comparisons": 41022, "provider_failure_tests": "Isolated fixtures only; no real credentials modified", "compiled_worker": "ESM validation passed", "rendered_browser": "Authenticated desktop 1440x900 and narrow mobile 390x844 passed"}
    metrics = {"provider_attempts": 2, "reserved_usd": 0.10, "actual_billed_usd": None, "billing_note": "Reservations and code estimates are not a provider invoice.", "first_usage": p["usage"], "second_usage": concurrency["first"]["body"]["analysis"]["provenance"]["usage"], "native_normalized_records": len(final_export["events"]), "native_record_kinds": native_counts, "raw_control_enforcement_records": native_counts.get("policy_event", 0), "merged_incident_count": None, "calibrated_precision": None, "raw_precision": None, "recall": None, "false_positive_events": None, "false_positives_per_10k": None, "unknown_reason": "This finite app acceptance has no labeled attack/benign truth set or accepted incident ledger. Control records and experimental changes are not attack labels."}
    js(OUT / "run_manifest.json", {"schema": "eidos.observatory.acceptance.v1", "run_id": RUN, "timestamp_utc": stamp, "status": "passed", "scope": "Finite owner-private Sol and rendered application acceptance", "product_runtime_sha": product, "sites_source_sha": site, "deployment_id": deployment["id"], "environment_revision": 5, "core_behavior_changed": False, "tests": tests, "metrics": metrics, "access": external["access"], "drive_manifest": "drive_manifest.json", "expected_artifact_omissions": {"pytest_results.xml": "The existing Python tests use unittest; exact results and the skip are retained in logs. Node JUnit XML is saved.", "compressed_streams": "No compression experiment was requested.", "crash_scan": "No new crash-resistance milestone is claimed.", "raw_provider_request": "Excluded to minimize retained provider context; hashes, bounded authoritative input comparison, IDs, usage and accepted output receipts are saved."}})
    commands = """node --test --test-reporter=junit apps/sentinel-observatory/live/collector.test.mjs apps/sentinel-observatory/live/brain.test.mjs apps/sentinel-observatory/live/browser-interpreter.test.mjs apps/sentinel-observatory/live/security-events.test.mjs apps/sentinel-observatory/live/hosted-sensor.test.mjs apps/sentinel-observatory/live/sol-interpreter.test.mjs
python -m unittest discover -s apps/sentinel-observatory/live -p 'test_*.py' -v
node apps/sentinel-observatory/live/verify_brain_parity.mjs
python apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/run_local_checks.py --out apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/local-after-fixes
node --check apps/sentinel-observatory/live/web/network-panel.mjs
git diff --check"""
    meaning = f"""## Proof Logic + Meaning

### Goal reached
**passed**: finite private Observatory credential activation, actual manual Sol execution, authoritative cache/control checks, and authenticated rendered desktop/mobile acceptance. This is the attached task's app gate.

### Previous state
The deployed app had no OpenAI key, no actual Sol receipt, and no rendered browser acceptance. The live pause control was replaced by each feed update and lost keyboard focus. Two Python checks relied on a tiny wall-clock window or unavailable Windows symlink privilege.

### Technical logic utilized
The server resolves a retained evidence ID, sanitizes bounded scalar metadata, reserves one durable attempt, calls exact `gpt-6.1-sol`, and validates the completed provider identity, usage and eligible structured ranking. Code rebuilds facts and citations from the authoritative record. The linked HTTP ID and canonical edge ID share the same cache. Polling and normal selection are read-only with respect to model jobs. The UI fix preserves the pause/resume button's DOM identity while updating freshness text.

### Math / scoring logic
`accepted = completed_provider_response AND exact_model AND valid_usage AND eligible_ranking AND authoritative_input_ids`.
`maximum_new_attempts = min(20, floor(0.50 / 0.05)) = 10 per UTC day`.
`observed_reservations = 2 * 0.05 = $0.10`; repeat requests for the first evidence created no second job.
Canonical parity uses `abs(actual-reference) <= 1e-7 + 1e-8 * abs(reference)` with exact categorical agreement: 954 frames and 41,022 comparisons passed. No Sentinel scoring or threshold changed. Precision, recall and false-positive rates are `null`: no labeled security truth set was evaluated.

### Philosophical meaning
Explanation before automation; restraint before alarm. Reproducibility is truth that can be revisited. A successful model call is separated from the truth of a security diagnosis.

### Why this is better
Before: credential and rendered acceptance were pending; focus vanished during feed refresh. After: real provider IDs and usage exist, cached aliases reuse them, live duplicate/cooldown requests reject safely, and the focused control survives an observed real feed update. Initial test failures and the remaining OS skip remain visible.

### How this moves Eidos closer to the north-star goal
Eidos Brain is a self-monitoring streaming intelligence codec. It learns live streams, compresses predictable behavior, preserves meaningful anomalies, monitors its own internal state, and emits human-readable incident receipts. This milestone strengthens reproducible operation and evidence-linked explanations. It does not evaluate streaming learning quality, compression value, anomaly-preservation quality, or superiority to other detectors/compressors.

### Evidence
`actual-sol-provider-receipt.json`, `authoritative-input-comparison.json`, `hosted-concurrency-cooldown.json`, `hosted-negative-after-key.json`, `final-external-state.json`, `browser-checks.json`, both final UI exports, screenshots, `local-after-fixes/node-tests.xml`, Python logs, parity receipt, `deployment-sol-configured.json`, and `drive_manifest.json`.

### Remaining uncertainty
Ranking usefulness and detection quality remain unproven. No labeled-domain, compression, GPU, physical-phone or alternate-browser acceptance is claimed. Provider outages and daily-dollar exhaustion were tested through isolated fixtures rather than deliberately damaging credentials or spending the remaining budget. {skip} Mounted Drive readback does not independently establish cloud synchronization. Eido remains an evaluation candidate.
"""
    report = f"""# Sentinel Observatory — private Sol acceptance

Result: **passed within the finite private-app acceptance scope**.

App: https://sentinel-evidence-observatory.bmparent.chatgpt.site
Draft PR: https://github.com/bmparent/eidos/pull/73

## Exact accepted deployment

- Product runtime: `{product}`.
- Sites source: `{site}`.
- Deployment: `{deployment['id']}`; succeeded at environment revision **5**.
- Saved version: `{deployment['version_id']}`.
- Worker SHA-256: `{deployment['worker_sha256']}`.
- Existing `OBSERVATORY_DB`, collector secrets and owner-only policy revision 1 were preserved. No external viewers were added. Applied migrations and the recorded capture were preserved.
- Final live JavaScript assets match the normalized inspected source. `/recorded` preserves the original recorded HTML; Cloudflare adds its challenge-platform script before the closing body. The original source file is unchanged, and no hosting protection was disabled. See `live-source-verification.json` and `source-invariants.json`.

## Actual provider and control acceptance

The key named **Sentinel Observatory** was created for **Personal / eidos_brain** through the connected OpenAI Developers encrypted-key workflow and bound only as server secret `OPENAI_API_KEY`. The picker did not appear; the owner selected the exact project through the connector's target list and an explicit chat choice. No API-key file or plaintext key log was produced.

The authenticated UI manually requested `{response['evidence_id']}`. The server resolved canonical `{response['source_event_id']}` and accepted a completed `gpt-6.1-sol` response: `{p['provider_response_id']}`, 390 input / 54 output / 444 total tokens, low reasoning, `model_ran:true`, `output_accepted:true`. All seven facts and both conditional explanations cite the retained canonical record. Its HTTP 301, 41.98 ms, verified TLS and 176/291 socket counters match the saved comparison receipt. No cause or attack was invented.

Repeating the same evidence in the UI and selecting its native alias returned the cached result. The job ledger remained at one job until a separately declared concurrency test intentionally created the second attempt. That test returned 200 for its completed provider response, **409** for the concurrent duplicate, and **429** for different evidence during cooldown. Two completed jobs reserve **$0.10** in total; this is not an invoice. The actual first cache gate passed without a second provider call.

With the real key configured, unknown IDs returned 404, imported IDs 400, extra client observations 400, and cross-origin requests 403. Configured polling and ordinary record selection left the model-job table empty before the first manual invocation. Provider errors, redirects, incomplete/malformed output, wrong model, timeout/cancellation, global request count and daily-dollar guards pass isolated regression tests with fixture credentials; live credentials were never deleted or altered for failure testing.

## Rendered browser acceptance and repairs

Authenticated owner Chromium browser checks passed at **1440x900** and **390x844**: connect/pause/resume, retained paused history after reload, measured evidence field, graph panning and endpoint selection, request/check/control filters, selected-evidence interpretation, cache and missing-key fallback, source age versus feed freshness, citations, normalized inspector, JSON downloads, dialog Escape and keyboard focus. Relevant application console warnings/errors were absent. The endpoint and sensor feed were left paused; the temporary viewport override was reset.

Fixed: sensor polling replaced the pause/resume button and dropped focus; it now retains DOM identity across real freshness updates. Collapsed mobile line breaks no longer join words/sentences. The deadline test now verifies watchdog scheduling/expiration deterministically. Symlink tests separate ordinary-path checks, a real-OS permission skip, and rejection of symlink metadata before opening.

## Validation and preserved failures

- 115 Node tests passed; Node JUnit XML retained.
- Python: 30 run, 29 passed, one explicit skip. {skip}
- Canonical numerical parity: 954 frames / 41,022 comparisons passed.
- Syntax, whitespace and exact compiled Worker ESM validation passed.
- The initial Python deadline and symlink failures are retained in the top-level local logs; the successful rerun is in `local-after-fixes/`.
- Packaging first selected unavailable WSL Bash (`HCS_E_CONNECTION_TIMEOUT`). The same supported Sites source helper succeeded using installed Git Bash and local archive handling. A short-lived source credential was renewed; no hosting audience or database migration was changed to work around this.
- The native table viewer truncated `result_json`; the full validated receipt was saved through authoritative cached API readback. The table ledger independently confirms job count, status, reservations and tokens.

Repo-root commands:

```text
{commands}
```

Actual Sites source packaging used the supported `site-workflow.mjs` helper, exact Worker generation and `scripts/validate-artifact.mjs`. The source helper returned the pushed source SHA and unchanged archive used for private deployment. No manual PYTHONPATH changes were used.

## Evidence boundaries and metrics

The attached sensor records this Observatory's incoming requests, outbound HEAD checks and known control enforcement. Functional-check labels from retained records remain visible. The final UI export has {len(final_export['events'])} normalized source records with counts `{json.dumps(native_counts, sort_keys=True)}`. These counts are not confirmed attacks or false-positive labels. No new demo incidents or simulated live measurements were added. Raw, merged and calibrated security-quality metrics are unknown where no ground truth exists; see `run_manifest.json` for explicit `null` values and reasons.

The unchanged literal-IP path still pins the public address and verifies TLS. DNS-name Worker checks cannot pin subsequent DNS resolution. The native sensor does not capture target visitors, LAN packets or provider firewall events. Broader intrusion evidence requires authorized Suricata/Zeek or application logs. Eido remains an evaluation candidate, not an accepted loaded checkpoint.

## Artifact locations and archive

Local: `apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/`.
Journal and analysis: `docs/proof_runs/2026-10-13/`.
Scoped visual progress: `artifacts/progress/sentinel-observatory-acceptance-20261013/`.
Drive disposition and checksums: `drive_manifest.json`. The mirror copies to a new dated run directory when the configured mounted Drive is writable. Cloud synchronization is a separate, unverified state.

{meaning}

## Recommended next step

Review conditional ranking usefulness against an explicitly labeled, authorized evidence set before making any detector-quality or broader Brain value claim. No additional feature or scientific expansion was implemented.
"""
    write(OUT / "acceptance_report.md", report)
    write(ROOT / "apps/sentinel-observatory/live/SOL-AUDIT-2026-10-13.md", report)
    analysis = "# Plain-language test analysis — 2026-10-13\n\nThe existing private Observatory now uses its own server-side Sol credential. A real owner-selected record produced a completed provider result, and repeats reused it. A second intentional test checked concurrency and cooldown. Facts still come from the stored measurements; Sol only orders predefined conditional explanations.\n\nThe actual browser exposed a disappearing keyboard focus control and mobile sentence spacing; both were repaired and verified on the deployed app. Local regressions passed with one clearly reported Windows OS-permission skip. The original failures remain saved.\n\nLocal logs, provider receipts, measured input comparison, screenshots, JSON exports, deployment and access receipts, ledgers and progress visuals are saved in this run folder. Google Drive copy status is recorded in drive_manifest.json. Detection accuracy, compression value, physical-phone behavior and cloud-sync completion remain unknown.\n\n" + meaning
    write(OUT / "plain_language_test_analysis.md", analysis)
    dated_document(DOCS / "plain_language_test_analysis.md", analysis)
    journal = f"""# Codex Journal — 2026-10-13

## What happened today
Continued the existing private Observatory and draft PR #73. Completed dedicated key activation, actual Sol/cache/control checks and authenticated desktop/mobile acceptance.

## What was accomplished
Saved real provider and authoritative evidence receipts; verified safe concurrency/cooldown/rejections; repaired sensor-control focus and mobile spacing; preserved numerical behavior, recorded capture, owner-only access and applied migrations.

## Tests and commands run
115 Node tests passed. Python ran 30: 29 passed and one explicit Windows symlink skip. Numerical parity passed 954 frames / 41,022 comparisons. Rendered 1440x900 and 390x844 checks, Worker ESM validation, syntax and whitespace passed.

```text
{commands}
```

## Problems encountered
The secure picker did not surface; the owner explicitly selected Personal / eidos_brain from connector-discovered targets. Initial Python failures were retained and repaired as test portability issues. WSL packaging failed; installed Git Bash completed the supported packaging helper. The native table viewer truncated a result JSON value; authoritative cached API readback supplied the complete sanitized receipt. Browser selector waits sometimes expired before asynchronous output completed; the existing operation was inspected rather than reissued as a new provider attempt.

## What changed
`web/network-panel.mjs`, `web/index.html`, two local boundary-test files, scoped `.gitattributes` and `.gitignore` receipt policies, audit/handoff/README, this run's receipts and dated proof documentation/progress artifacts. Runtime `OPENAI_API_KEY` was added as a secret; environment revision is 5. The narrow log exception keeps every manifested receipt available after a clean checkout.

## What did not change
Core numerical Sentinel behavior, RLS/reservoir/scoring/threshold policy, collector and Sol algorithms, cost controls, applied migrations, owner-only access, recorded evidence and protected relay configuration were untouched. Eido was not loaded or accepted.

{meaning}

## Artifacts generated
This run folder includes config, run manifest, provider/deployment/access receipts, original and corrected test logs, Node XML, parity result, browser checks/screenshots/exports, analysis, journal, ledgers and visual progress. The expected-omission reasons are explicit in run_manifest.json.

## Google Drive archive status
See drive_manifest.json for the actual root, dated run directory, copied files, skipped files, checksums and copy result. Only mounted readback is verified; remote cloud synchronization is unknown.

## Thoughts on improvement
Keep application acceptance separate from scientific detection and compression proof. Keep the one OS-permission skip visible.

## Where to improve next
A narrowly scoped, labeled usefulness review of the conditional rankings.

## Anything that stands out
Zero model jobs existed after configured polling and ordinary selection. The first cached UI repeat kept one job. The separate real concurrency test produced a second completed job and safe 409/429 responses. Two jobs reserve $0.10; actual billing is unknown.

## End-of-task summary
1. Files changed: narrow UI/test fixes, audit/handoff/README, receipt byte policy, proof artifacts and dated docs.
2. Core behavior changed: no.
3. Tests added/skipped: split symlink coverage plus metadata rejection; one real-OS symlink skip (WinError 1314).
4. Repo-root commands run: listed above; exact argv/logs in local-checks.json.
5. Artifacts generated: manifests, logs/XML, provider/cache/control/deployment receipts, screenshots/exports, analysis, journal and progress/logic ledgers.
6. Local artifact folder: apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/.
7. Drive status: exact final copy result and file list in drive_manifest.json; remote sync unknown.
8. Plain-language analysis: written locally and included in the mirror set.
9. Codex journal: written locally and included in the mirror set.
10. Known limitations: one OS symlink skip; no physical-phone, alternate-browser, detector-quality or compression-value claim.
11. Follow-up not implemented: labeled ranking usefulness review and broader scientific proof.
12. Proof Logic + Meaning: written in audit, analysis, journal and logic ledger.
13. Specific math: acceptance conjunction, per-evidence deduplication, UTC reservation cap and canonical parity tolerance.
14. Philosophical meaning: explanation before automation; restraint before alarm; revisitable truth.
15. Why better: real execution/caching receipts and rendered acceptance replace previously pending gates; focus survives polling.
16. North-star contribution: reproducible operation and evidence-linked explanations.
17. Evidence: provider receipt, authoritative input comparison, browser-checks, exports/screenshots, local logs/parity, final-external-state and deployment receipt.
18. Remaining uncertainty: ranking usefulness, security ground truth, compression benefit, actual billed spend and remote cloud synchronization.
"""
    write(OUT / "codex_journal.md", journal)
    dated_document(DOCS / "codex_journal.md", journal)
    handoff = ROOT / "apps/sentinel-observatory/live/CODEX-HANDOFF-SOL.md"
    old = handoff.read_text(encoding="utf-8")
    historical = old.split("## Historical handoff", 1)[-1] if "## Historical handoff" in old else old
    write(handoff, f"# Sentinel Observatory continuation completed\n\nThe finite private Sol/provider/cache/control and authenticated desktop/mobile acceptance passed on 2026-10-13. Preserve the working server secret; do not provision another key for this completed handoff.\n\nProduct `{product}`; Sites `{site}`; deployment `{deployment['id']}`; environment revision 5. Read [SOL-AUDIT-2026-10-13.md](./SOL-AUDIT-2026-10-13.md) and [the acceptance report](./evidence/2026-10-13-codex-acceptance/acceptance_report.md) for receipts, test scope and remaining scientific limits.\n\n## Historical handoff\n{historical}")
    readme = ROOT / "apps/sentinel-observatory/live/README.md"
    text = readme.read_text(encoding="utf-8").replace("Credential provisioning and the remaining actual-provider/browser checks are documented in [CODEX-HANDOFF-SOL.md](./CODEX-HANDOFF-SOL.md).", "Actual private Sol/provider/cache/control and authenticated desktop/mobile acceptance are documented in [SOL-AUDIT-2026-10-13.md](./SOL-AUDIT-2026-10-13.md). [CODEX-HANDOFF-SOL.md](./CODEX-HANDOFF-SOL.md) preserves the completed continuation and historical handoff. Scientific ranking and detection usefulness remain unproven.")
    write(readme, text)
    gates = [
        ("Secure dedicated credential and private deployment", 15, "secret-binding.json"),
        ("Actual accepted exact-model provider response", 25, "actual-sol-provider-receipt.json"),
        ("Authoritative citations and durable cache", 15, "authoritative-input-comparison.json"),
        ("Safe controls and provider-failure regressions", 15, "hosted-concurrency-cooldown.json"),
        ("Authenticated desktop/mobile rendered acceptance", 20, "browser-checks.json"),
        ("Reproducible proof package and journal", 10, "run_manifest.json"),
    ]
    progress = {"scope": "Finite Sentinel Observatory private-app acceptance", "status": "passed", "acceptance_gates": [{"name": n, "weight": w, "status": "passed", "evidence": e} for n, w, e in gates], "scoped_acceptance_score": sum(w for _, w, _ in gates), "scoped_maximum": 100, "overall_brain_proof_readiness_score": None, "overall_reason": "No project-wide weighted proof-gate audit was performed; application acceptance does not accept global research gates.", "north_star": "Eidos Brain is a self-monitoring streaming intelligence codec.", "metrics": metrics, "unproven": ["Ranking usefulness", "Labeled detection quality and false-positive rates", "Compression/anomaly preservation", "Physical-phone and alternate-browser behavior", "Remote Drive synchronization"], "broader_gates": [{"name": n, "weight": w, "status": "unknown", "gate_pass": None, "reason": "Outside this finite app acceptance; existing global proof evidence was not audited."} for n, w in [("reproducible baseline", 15), ("labeled-domain proof", 15), ("precision/false-positive discipline", 15), ("Sentinel calibration/generalization", 15), ("compression/anomaly preservation", 10), ("incident cards/explanation", 10), ("domain demos", 10), ("one-command reproducibility/final report", 10)]]}
    js(PROGRESS / "eidos_progress_meter.json", progress)
    evidence_prefix = "../../../apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/"
    rows = "\n".join(f"| {n} | {w} | passed | [{e}]({evidence_prefix}{e}) |" for n, w, e in gates)
    write(PROGRESS / "eidos_progress_meter.md", f"# Sentinel Observatory proof progress\n\nScoped app acceptance: {progress['scoped_acceptance_score']}/100, calculated from the explicit accepted gates below. Overall Brain proof readiness: **unknown**; no project-wide gate audit was performed.\n\n| Gate | Weight | Status | Receipt |\n| --- | ---: | --- | --- |\n{rows}\n\nPrecision, recall, false-positive rates and compression value remain unknown. Raw native controls remain visible in the run manifest.\n\n" + meaning)
    svg_rows = "".join(f'<rect x="28" y="{135+i*50}" width="824" height="38" rx="5" fill="#132e2d"/><text x="42" y="{160+i*50}" fill="#eaf6f0" font-size="14">{html.escape(n)}</text><text x="780" y="{160+i*50}" fill="#a9e8cd" font-size="13">passed</text>' for i, (n, _, _) in enumerate(gates))
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="880" height="520" viewBox="0 0 880 520" role="img" aria-label="Finite private app acceptance passed; overall Brain proof readiness unknown"><rect width="880" height="520" fill="#0a141d"/><g font-family="Arial,sans-serif"><text x="28" y="44" fill="#eef7f5" font-size="25">Sentinel Observatory · proof receipts</text><text x="28" y="76" fill="#a9e8cd" font-size="17">Scoped app acceptance: 100/100 · six evidenced gates</text><text x="28" y="104" fill="#b1bfcb" font-size="14">Overall Brain proof readiness: unknown · no global gate audit</text>{svg_rows}<text x="28" y="468" fill="#b1bfcb" font-size="13">Two intentional provider attempts · $0.10 reserved · no detection-quality claim</text><text x="28" y="496" fill="#b1bfcb" font-size="13">Explanation before automation. Evidence that can be revisited.</text></g></svg>'
    write(PROGRESS / "eidos_progress_meter.svg", svg)
    cards = "".join(f'<li><strong>{html.escape(n)}</strong><span>passed · weight {w}</span><code>{e}</code><a href="{evidence_prefix}{e}">Open receipt</a></li>' for n, w, e in gates)
    dashboard = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sentinel Observatory proof progress</title><style>body{background:#0a141d;color:#edf5f2;font:16px/1.65 system-ui;margin:0;padding:36px}main{max-width:1000px;margin:auto}h1{font-weight:550;line-height:1.2}p{color:#b7c7cc}ul{padding:0;list-style:none;display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}li{background:#11232d;border:1px solid #29414b;padding:20px;border-radius:9px}span,code{display:block;font-size:13px;overflow-wrap:anywhere}span{color:#a9e8cd}code{color:#b8c8d0;margin-top:12px}a{color:#a9e8cd}</style><main><p>EIDOS · FINITE PRIVATE APP ACCEPTANCE</p><h1>Proof with its limits visible.</h1><p>Scoped acceptance 100/100. Overall Brain proof readiness is unknown: no global gate audit was performed.</p><ul>' + cards + '</ul><p>115 Node checks passed. Python: 29 passed, one OS-permission skip. 954 parity frames / 41,022 comparisons. Two real provider attempts; $0.10 reserved. Model usefulness, detector precision, recall, false-positive rates and compression value remain unproven.</p><p><a href="eidos_progress_meter.md">Readable evidence ledger</a> · <a href="eidos_progress_meter.json">Machine-readable progress</a></p></main></html>'
    write(PROGRESS / "eidos_progress_dashboard.html", dashboard)
    write(PROGRESS / "eidos_progress_readme.md", "# Progress scope\n\nThis is a finite private-app acceptance meter generated from this run's actual receipts. It does not update or accept global Eidos Brain scientific gates. Broader weighted gate statuses and gate_pass values are unknown/null because those existing project-wide receipts were not audited in this task. SVG renders without JavaScript; the HTML is a static local evidence view.\n\nEvidence root: apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/.\n")
    js(PROGRESS / "proof_logic_ledger.json", {"run_id": RUN, "status": "passed", "scope": progress["scope"], "equations": ["accepted = completed AND exact_model AND valid_usage AND eligible_ranking AND authoritative_ids", "maximum_new_attempts = min(20, floor(0.50 / 0.05)) = 10", "observed_reservations = 2 * 0.05 = 0.10", "parity_tolerance = 1e-7 + 1e-8 * abs(reference)"], "evidence": [e for _, _, e in gates], "unknown_metrics": [k for k, v in metrics.items() if v is None], "philosophical_meaning": "Explanation before automation; restraint before alarm; revisitable truth."})
    write(PROGRESS / "proof_logic_ledger.md", "# Proof logic ledger\n\n" + meaning)
    for path in PROGRESS.iterdir():
        destination = OUT / "progress" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in (".md", ".html"):
            write(destination, path.read_text(encoding="utf-8").replace(evidence_prefix, "../"))
        else:
            shutil.copy2(path, destination)
    js(OUT / "packaging-recovery.json", {"initial_failure": "WSL Bash HCS_E_CONNECTION_TIMEOUT while packaging", "repair": "Used installed Git Bash and TAR_OPTIONS=--force-local for the same supported Sites source helper", "source_sha": site, "artifact_worker_sha256": deployment["worker_sha256"], "hosting_policy_changed": False, "migrations_changed": False})
    drive = {"drive_copy_attempted": False, "drive_copy_success": False, "drive_root": "unknown", "drive_run_dir": "unknown", "reason": "No writable configured Drive root", "files_considered": [], "files_copied": [], "files_skipped": [], "timestamp_utc": stamp, "remote_sync_verified": False}
    selected = os.environ.get("EIDOS_PROOF_DRIVE_DIR") or os.environ.get("EIDOS_ARTIFACT_ROOT")
    if selected and Path(selected).is_dir():
        root = Path(selected).resolve()
        destination = root / "Eidos_Brain_Proof_Phase" / "2026-10-13" / RUN
        if destination.exists():
            existing = destination / "run_manifest.json"
            assert existing.is_file() and json.loads(existing.read_text(encoding="utf-8"))["run_id"] == RUN, "Refusing to overwrite an unrelated Drive folder"
        destination.mkdir(parents=True, exist_ok=True)
        assert destination.resolve().is_relative_to(root)
        drive.update(drive_copy_attempted=True, drive_root=str(root), drive_run_dir=str(destination))
        try:
            for path in OUT.rglob("*"):
                if path.is_file() and path.name not in ("drive_manifest.json", "file_manifest.json"):
                    relative = path.relative_to(OUT).as_posix()
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    assert digest(path) == digest(target), f"Mirror checksum mismatch: {relative}"
                    drive["files_considered"].append(relative)
                    drive["files_copied"].append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
            drive.update(drive_copy_success=True, reason="Mounted Drive copy and SHA-256 readback verified. Remote cloud synchronization is not independently confirmed.")
        except (OSError, AssertionError) as error:
            drive.update(reason=f"Mirror incomplete: {type(error).__name__}: {error}")
    js(OUT / "drive_manifest.json", drive)
    disposition = (f"Artifacts copied to `{drive['drive_run_dir']}` with SHA-256 readback; cloud synchronization remains unverified." if drive["drive_copy_success"] else f"Drive copy incomplete or skipped: {drive['reason']}. Repo-local receipts remain available.")
    for path in (OUT / "codex_journal.md", OUT / "plain_language_test_analysis.md", OUT / "acceptance_report.md"):
        write(path, path.read_text(encoding="utf-8") + "\n\n## Final Google Drive disposition\n\n" + disposition)
    dated_document(DOCS / "codex_journal.md", (OUT / "codex_journal.md").read_text(encoding="utf-8"))
    dated_document(DOCS / "plain_language_test_analysis.md", (OUT / "plain_language_test_analysis.md").read_text(encoding="utf-8"))
    write(ROOT / "apps/sentinel-observatory/live/SOL-AUDIT-2026-10-13.md", (OUT / "acceptance_report.md").read_text(encoding="utf-8"))
    files = [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(OUT.rglob("*")) if p.is_file() and p.name not in ("file_manifest.json", "drive_manifest.json")]
    js(OUT / "file_manifest.json", {"timestamp_utc": stamp, "files": files, "self_reference_note": "file_manifest.json and drive_manifest.json are excluded from this hash list; Drive readback also checks the final manifest bytes."})
    if drive["drive_copy_success"]:
        destination = Path(drive["drive_run_dir"])
        copied = []
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path.name != "drive_manifest.json":
                relative = path.relative_to(OUT).as_posix()
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                assert digest(path) == digest(target)
                copied.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
        drive["files_copied"] = copied
        drive["files_considered"] = [f["path"] for f in copied] + ["drive_manifest.json"]
        drive["files_skipped"] = []
        drive["files_copied"].append({"path": "drive_manifest.json", "bytes": None, "sha256": None, "reason": "Final manifest is copied and byte-verified separately; a manifest cannot recursively record its own checksum."})
        js(OUT / "drive_manifest.json", drive)
        shutil.copy2(OUT / "drive_manifest.json", destination / "drive_manifest.json")
        assert digest(OUT / "drive_manifest.json") == digest(destination / "drive_manifest.json")
    print(json.dumps({"status": "passed", "local_artifacts": str(OUT), "proof_files": len(files), "drive_copy_success": drive["drive_copy_success"], "drive_run_dir": drive["drive_run_dir"], "global_brain_proof_score": None}))


if __name__ == "__main__":
    main()
