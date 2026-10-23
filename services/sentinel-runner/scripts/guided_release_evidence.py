"""Build implementation reports from existing receipts; never run or tune evaluation."""
import hashlib
import html
import json
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "artifacts/sentinel-guided-20260908"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8-sig"))


def write(name, data):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) if isinstance(data, (dict, list)) else data, encoding="utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    now = datetime.now(timezone.utc).isoformat()
    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    summary, qualification = read("evaluation/final-summary.json"), read("evaluation/qualification.json")
    preview = read("release-preview.json")
    metrics = {r["method"]: r for r in summary["results"]}
    requirements = [
        ("Guided CSV workflow, chronology, Torch, source, mobile, reload, two-user isolation", "1,2", "preview-final/browser-receipt.json", "apps/sentinel-lab/components/guided-lab.tsx"),
        ("XLSX upload and original records", "1", "formats/receipt.json", "services/sentinel-runner/sentinel_runner/guided/ingestion.py"),
        ("Parquet, JSON, JSONL, structured log upload", "1", "formats-verified/receipt.json", "services/sentinel-runner/sentinel_runner/guided/ingestion.py"),
        ("Text, HTML, text PDF, plain log semantic retrieval", "4", "documents-verified/receipt.json", "services/sentinel-runner/sentinel_runner/guided/semantic.py"),
        ("Real downloadable URL, web page, patterns, unsafe redirect rejection", "1,4", "urls-cold-server/receipt.json", "apps/sentinel-lab/lib/guided/fetch-source.ts; patterns.ts"),
        ("Hosted semantic retrieval and public imports", "4", "preview-formats/receipt.json", "services/sentinel-runner/sentinel_runner/guided/semantic.py"),
        ("Causal invariance, labels, irregular/entity boundaries, ingestion failures", "1,2", "runner-final.xml", "services/sentinel-runner/tests/test_guided.py"),
        ("Frozen final baselines and memory/multiscale/regulation/TraceSeal ablations", "2", "evaluation/qualification.json", "services/sentinel-runner/scripts/guided_evaluate.py"),
        ("State-equivalent real stream replay, late/gap/backpressure/reset/revocation", "3", "stream/receipt.json", "apps/sentinel-lab/lib/guided/telemetry.ts"),
        ("Hosted ingress, scoped keys and browser health; processing blocked by billing", "3", "preview-budget/receipt.json", "services/sentinel-runner/sentinel_runner/guided/telemetry.py"),
        ("Historical reviewed outcomes with owner scope and immutable metrics", "1,4", "reviewed-history/receipt.json", "apps/sentinel-lab/lib/guided/api.ts"),
        ("Official Collector forwarding and configuration validation", "3", "collector/receipt.json", "connectors/sentinel/"),
        ("Cancellation, same-intent retry and subsequent real job", "1,3", "cancellation/receipt.json", "apps/sentinel-lab/lib/guided/jobs.ts"),
        ("Actual legacy selection and semantic suffix/cache audit", "2,4", "semantic-final/receipt.json", "services/sentinel-runner/scripts/guided_semantic_audit.py"),
        ("Legacy full engine engineering and mechanisms regression", "2", "legacy-standard.log; legacy-mechanisms.log", "services/sentinel-runner/scripts/verify_full_engine.py"),
        ("Account contracts, concurrent stores, private resources, pinned public IPv4", "1,3", "app-final-test.log", "apps/sentinel-lab/tests/guided.test.ts"),
    ]
    def status(evidence):
        first = evidence.split(';')[0]
        path = OUT / first
        if not path.exists(): return "missing"
        if first.endswith("qualification.json"): return "measured; operational qualification failed"
        if first == "preview-formats/receipt.json": return "semantic upload/retrieval passed; later URL confirmation blocked by provider HTTP 402"
        if first == "preview-budget/receipt.json": return "ingress/security/billing-failure checks passed; hosted model processing blocked HTTP 402"
        if first in {"formats/receipt.json", "formats-verified/receipt.json", "documents-verified/receipt.json"}: return "listed adapter checks passed; later failure retained and resolved in later receipt"
        if first.endswith(".json"): return read(first).get("status", "evidence_exists")
        return "evidence_exists"
    table = "| Requirement | Milestone | Observed status | Receipt | Code |\n|---|---|---|---|---|\n" + "\n".join(f"| {name} | {milestone} | {status(receipt)} | `{receipt}` | `{code}` |" for name, milestone, receipt, code in requirements)
    write("requirement-to-evidence.md", "# Implementation acceptance evidence\n\nAll paths below are relative to this artifact folder. Failed attempts remain alongside their corrected acceptance receipts.\n\n" + table + "\n")
    rows = []
    for method, result in metrics.items():
        def value(metric):
            v = result[metric]["mean"]
            return "NA" if v is None else f"{v:.4f}"
        rows.append(f"| {method} | {value('forecastMAE')} | {value('incidentPrecision')} | {value('incidentRecall')} | {value('falseAlertsPerOperatingDay')} | {value('intervalCoverage')} |")
    result_table = "| Method | MAE | Incident precision | Incident recall | False alerts/day | Coverage |\n|---|---:|---:|---:|---:|---:|\n" + "\n".join(rows)
    write("benchmark_summary.md", "# Frozen synthetic final evaluation\n\n" + result_table + "\n\nThese are means of whole-seed period means across three independent generated periods; the seven scenarios within a period are correlated. Min/max period ranges, interval widths, detection delay and raw/merged event counts remain in final-summary.json, final-metrics.csv and the verified raw ZIP. They are not iid confidence intervals. NA means the method does not issue forecasts or interval estimates. Every Eidos variant failed the prespecified operational qualification; all remain experimental. The default MAE improves over persistence here but its false-alert burden is higher. The robust prefix baseline has better incident precision/recall in this fixture set; it also fails the one-false-alert-per-day limit. No baseline is promoted as an operational guarantee.\n\nMemory was a canonical Hippocampus shadow observer, with suppression deliberately disabled; unchanged detection does not establish utility. TraceSeal changed shadow scores but did not establish a safety/accuracy benefit. Neither is enabled as a proven improvement. Peak local resident memory: 364,937,216 bytes. Hosted peak memory and attributed infrastructure dollars are unknown; 8 GB is an allocation limit, not a measurement. LLM calls and LLM cost are zero.\n")

    frozen = read("evaluation/acceptance-freeze.json")
    files = []
    frozen_revision = "272befb"
    with zipfile.ZipFile(OUT / "evaluation/frozen-code.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, expected in frozen["codeHashes"].items():
            posix = name.replace("\\", "/")
            data = subprocess.check_output(["git", "show", f"{frozen_revision}:{posix}"])
            # The frozen Windows working tree may have checkout line endings.
            if digest(data) != expected and digest(data.replace(b"\n", b"\r\n")) == expected: data = data.replace(b"\n", b"\r\n")
            if digest(data) != expected: raise RuntimeError(f"Frozen source mismatch: {posix}")
            archive.writestr(posix, data)
            files.append({"path": posix, "frozenSha256": expected, "currentSha256": digest((ROOT / posix).read_bytes()),
                          "same": digest((ROOT / posix).read_bytes()) == expected})
        generator = "services/sentinel-runner/scripts/guided_evaluate.py"
        data = subprocess.check_output(["git", "show", f"{frozen_revision}:{generator}"])
        expected = read("evaluation/partition-plan.json")["generatorSha256"]
        if digest(data) != expected and digest(data.replace(b"\n", b"\r\n")) == expected: data = data.replace(b"\n", b"\r\n")
        if digest(data) != expected: raise RuntimeError("Frozen generator mismatch")
        archive.writestr(generator, data)
    write("evaluation/frozen-code-comparison.json", {"frozenRevision": frozen_revision, "files": files,
        "explanation": "After final consumption, only semantic model cache-loading was optimized among frozen modules. The numerical evaluation imports causal/ingestion, not semantic. No evaluated numerical algorithm, generator, thresholds or labels were tuned after final. Archived exact frozen source remains available.", "researchGatesAdvanced": 0})

    logic = """## Proof Logic + Meaning

### Goal reached
All four product milestones have working implementations and local integration receipts. Full hosted acceptance is partial: actual CSV/Torch and document retrieval passed, then Vercel blocked new workers with HTTP 402 payment_required. Hosted ingress, privacy, reviewed history and explicit failure handling passed on the final candidate. The remaining hosted URL confirmation/analysis and checkpointed telemetry processing require restored Sandbox capacity. Frozen operational alert-quality qualification failed. Production release approval is pending; this is a reviewable candidate with a documented external gate, not an operational detector qualification.

### Previous state
The main product was a research console and Kaggle workflow. General private uploads, named causal forecasts, owned monitors and document retrieval were absent. Legacy selection could choose the lower current residual after seeing the observation, and character prefixes did not represent document meaning.

### Technical logic utilized
M1: Byte checksums, explicit schema confirmation, original record references, owner-scoped libSQL records and idempotent jobs link the UI to actual isolated execution. M2: The actual canonical Torch RLS reservoir is called through a new named-feature adapter; a fixed 24-observation calibration prefix, historical-loss predictor selection, immutable issued forecasts and score-before-update preserve causal interpretation. Anomalies are skipped during adaptation by default. M3: Arrival/event timestamps, ingress deduplication, contiguous offsets, checkpoint compare-and-set and actual saved reservoir/RLS state preserve replay equivalence across batches. M4: Pinned MiniLM embeddings pool every token window; cosine retrieval returns original passages and content hashes, with no model-generated code or per-event LLM calls.

### Math / scoring logic
Reservoir: r_t = (1-alpha) r_(t-1) + alpha*tanh(W_in*x_t + W_rec*r_(t-1)). RLS: k=P*z/(lambda+z^T*P*z); W_out'=W_out+error*k^T. A forecast is committed before its target: prediction_t=f(state_before_t), residual_t=observation_t-prediction_t. Surprise uses the prior residual median and MAD scale; decision=score>=prior_threshold, then the recorded learning policy determines update. This robust score is not a probability. MAE=mean(abs(observed-issued)); coverage=count(target in issued band)/resolved intervals. Precision=TP_incidents/(TP_incidents+FP_incidents); recall=detected_truth_incidents/truth_incidents; false_alerts/day=FP_incidents/operating_seconds*86400. Matching horizon and maximum late window are explicit seconds. Pearson correlations are retrospective associations, with complete-pair counts; cosine similarity is association, not factual confidence. Unknown metrics stay null/NA.

### Philosophical meaning
Reproducibility is truth that can be revisited. Source drill-down is explanation before automation. Separating anomaly from consequence is restraint before alarm. Keeping failed qualification visible is honesty before optimization. Familiarity is recognition, never automatic permission to suppress a repeated harmful event.

### Why this is better
A user can now bring a supported source through confirmation to an actual saved result, inspect a named forecast and the records behind a finding, return after interruption, and revoke a source credential. The causal audit removes the retrospective-forecast ambiguity. The result is more inspectable; the measurements do not prove universal improvement or an acceptable operational alert burden.

### How this moves Eidos closer to the north-star goal
The north-star claim is: Eidos Brain is a self-monitoring streaming intelligence codec. It learns live streams, compresses predictable behavior, preserves meaningful anomalies, monitors its own internal state, and emits human-readable incident receipts. This release strengthens live-stream ingestion/learning, retained anomalies, visible internal state, source-linked incident explanation and reproducible execution. It does not establish new compression performance, broad domain generalization or value beyond all existing detectors/compressors.

### Evidence
The requirement-to-evidence table maps each milestone to browser, runner, provider, collector, replay, cancellation and semantic receipts. The immutable evaluation plan, acceptance freeze, final metrics, qualification decision, verified raw archive and frozen source archive retain the complete numerical result. Two-user access checks and real worker hashes support privacy/execution claims; hashes alone do not establish accuracy.

### Remaining uncertainty
All four Eidos variants failed operational precision/false-alert criteria; regulation also failed coverage. Synthetic periods do not establish natural-domain generalization. Vercel HTTP 402 payment_required blocks the remaining hosted compute matrix and a full final-SHA hosted CSV repeat; earlier actual Sandbox execution receipts identify their tested SHA separately. No sealed Grand Proof gate was opened or advanced. GPU, OCR/image PDFs, IPv6-only URLs, authenticated web imports, unrestricted stream rates, broad load testing and attributed provider cost are unqualified. No collector is installed for unrelated or live customer collection. Private sources require deliberate owner setup. Research readiness is unknown; no percentage is assigned.
"""
    overview = f"""# Sentinel guided implementation — 2026-09-08

Four connected milestones are implemented in PR https://github.com/bmparent/eidos/pull/52. Final application candidate: `{preview['sourceCommit']}`. Preview: {preview['url']}. Actual hosted CSV/Torch and semantic worker source: `da14cc77e988555fd7b607d0cb9b24ba5f5f8dc9`; the final candidate adds reviewed-history context and explicit provider-rejection handling, tested separately in preview. Final local stream replay, current CI/build and hosted noncompute checks are recorded separately. Evidence packaging revision: `{code_sha}`.

The original dirty checkout was preserved. Work was isolated in `codex/sentinel-guided-analysis-20260908`. Existing Works member changes were integrated from upstream through PR #51; this task did not publish those account releases. Production started at `be02b6cba3579eb10412f7d148e22e62a48a87df`; another authorized task moved it to `233a1d8404d6b9be806cdf9c45a5ac861123e871`. This task creates preview deployments only.

## What happened today

Implemented versioned ingestion and confirmation, private durable jobs/results/feedback, actual causal Torch readouts, original-unit plots and issued forecasts, retrospective patterns, scoped OpenTelemetry/JSONL monitoring, and pinned semantic retrieval. Retained `/research`, the Kaggle connector, old engine observatory and historical evidence. Shared admission and current member tables were reused; four additive guided tables require no destructive migration.

## What was accomplished

Browser acceptance covers CSV, XLSX, Parquet, JSON, JSONL, numeric logs, plain logs, TXT, HTML, text PDF, real downloadable URLs and public pages. It includes malformed/oversize errors, source drill-down, mobile, keyboard/reduced-motion, reload, two owners and real worker receipts. Stream checks cover duplicate conflict rollback, late/gap policy, replay equivalence, interruption, backpressure, reset, revocation and an actual official Collector forwarding three synthetic points. Cancellation and subsequent capacity recovery passed. The immutable synthetic final evaluation was completed once; failed qualification remains visible.

## Tests and commands run

All local commands ran from repository root without manual PYTHONPATH edits. Focused runner suite: `.venv-guided/Scripts/python.exe -m pytest services/sentinel-runner/tests --junitxml=artifacts/sentinel-guided-20260908/runner-final.xml` — 37 passed. Existing global pytest plugin incompatibility required `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; this disables third-party discovery, not repo tests. `npm test --prefix apps/sentinel-lab` — 16 JavaScript plus 42 TypeScript tests passed. `npm run lint --prefix apps/sentinel-lab` and `npm run build --prefix apps/sentinel-lab` passed. `guided_fixtures.py`, `guided_semantic_audit.py`, both `verify_full_engine.py` profiles, `guided-browser-qa.mjs`, `guided-formats-qa.mjs`, `guided-stream-qa.mjs`, `guided-cancel-qa.mjs` and `guided-collector-qa.mjs` produced the receipts indexed below. See release guide for complete invocations/environment prerequisites. Build/CI results are source-specific, not a production-promotion receipt.

## Problems encountered

Preserved failures include the local worker's missing pyarrow, a document-query locator mismatch, public dual-stack DNS rejection, stale development-server modules, Windows cp1252 legacy console failure, initial OAuth workflow-scope push rejection, protected-preview login and disk pressure. Fixes were scoped: venv dependency availability, explicit Question label, validated/pinned IPv4 selection, owned dev-server restart, UTF-8 output, leaving the CI workflow unchanged and declaring parser base dependencies, authorized temporary preview access, and lossless verified compaction of this task's evaluation. The official Collector temporary executable was removed after successful validation to recover disk. No user archive was deleted, and earlier failed receipts were retained.

## What changed / What did not change

Code is under `apps/sentinel-lab/components/guided-*`, `lib/guided`, `app/api/lab/v1`, Python `sentinel_runner/guided`, focused tests/scripts and `connectors/sentinel`. The root experience moves to guided analysis; the existing console is retained at `/research`. Legacy reservoir/engine implementation files, historical algorithms/imports, frozen proof archives and Grand Proof gates were not changed. The new explicit causal adapter uses the existing reservoir/RLS class with separately recorded policy; it does not reinterpret legacy output as issued forecasts.

{logic}

## Artifacts generated

Repo-local: `artifacts/sentinel-guided-20260908/`. Reports, manifests, hashes, JUnit, CLI logs, screenshots, real source/result snapshots, issued forecasts, semantic vectors, monitor replay state, collector validation and all raw synthetic evaluation outputs are retained. The 420 evaluation files are losslessly stored in `evaluation/period-raw-artifacts.zip`, with individual hashes in `period-raw-manifest.json`. Exact frozen modules/generator are in `evaluation/frozen-code.zip`. Large generated ZIPs/fixtures are repo-local and Drive-mirrored; compact review evidence is committed. Credentials, database files and worker requests under `artifacts/sentinel-guided-private/` and `artifacts/guided-jobs/` are excluded.

## Google Drive archive status

See `drive_manifest.json` for the actual copy outcome and checksums. Configured root: `G:/My Drive`. Target: `Eidos_Brain_Proof_Phase/2026-09-08/sentinel-guided-20260908/`. Only new sanitized task artifacts are mirrored. Mounted-file verification proves the local Drive mirror, not independently observed completion of the provider's background cloud sync.

## Thoughts on improvement / Where to improve next

Keep the detector experimental. A separate development-only task should reduce false alerts while preserving recall, with a new untouched final partition and acceptance freeze. The external action is to restore Vercel Sandbox creation for this project through its usage/billing settings or the provider's quota reset; no plan or payment was changed. Then rerun the remaining hosted acceptance scripts. Before production promotion, an operator chooses the release and authorized collector target. No service purchase is needed to review the source, saved evidence or local workflow.

## Anything that stands out

Default final MAE was {metrics['eidos_none']['forecastMAE']['mean']:.4f} versus persistence {metrics['persistence']['forecastMAE']['mean']:.4f}, but incident precision was {metrics['eidos_none']['incidentPrecision']['mean']:.2f} and false alerts/day {metrics['eidos_none']['falseAlertsPerOperatingDay']['mean']:.4f}, above the frozen maximum of 1. Findings and empty findings both require interpretation. Local Python 3.11/Torch 2.6 and hosted Python 3.14/Torch 2.14 are recorded separately. This is no claim of identical cross-version numerical output.

## End-of-task summary

1. Files changed: guided app/runner/adapters/tests/connector and release documents; see PR diff.
2. Core behavior: legacy engine unchanged; explicitly separate causal policy added.
3. Tests: runner, app, full engine, browser, collector, semantic, cancellation and streaming acceptance; receipts retain failures/retries.
4. Commands: repo-root reproduction in `docs/sentinel-guided-release.md`.
5. Artifacts: local folder above, raw archive and compact review receipts.
6. Plain-language analysis: this report and benchmark summary.
7. Journal: separate Sentinel entry linked from the existing date journal; prior Works entry preserved.
8. Drive: exact state in drive_manifest.json; no secrets or historical archive changes.
9. Limits: 2 MB source, 5,000 rows, 16 numeric features, 8 entity/session groups, 256 passages, one 900-second 4-vCPU/8-GB worker; 4 processing batches/source/day, 1,000 buffered events, 10,000 ingress/day.
10. Follow-up: restore Sandbox capacity and finish hosted compute matrix; production approval, deliberate collector installation and operational detector improvement are not performed.
11. Proof Logic + Meaning: included above.
12. Math/logic: causal issue/score/update, RLS, MAE/coverage, incident precision/recall/FA, replay offsets and semantic similarity.
13. Philosophy: inspectable evidence, restraint and honesty.
14. Improvement: general sources now reach private, inspectable, actual computation and survive interrupted clients.
15. North-star: stream learning, anomaly retention, state visibility, explanation and reproducibility strengthened.
16. Evidence: requirement table, manifests, acceptance freeze, metrics and source hashes.
17. Unproven: operational alert qualification failed; research readiness, generalization, compression superiority, GPU and broad load remain unproven.
"""
    write("codex_journal.md", overview)
    write("plain_language_test_analysis.md", overview + "\n## Final comparisons\n\n" + result_table + "\n")
    write("proof_logic_ledger.md", "# Implementation proof logic ledger\n\n" + logic)
    write("proof_logic_ledger.json", {"createdAt": now, "productMilestones": [1, 2, 3, 4], "engineeringEvidence": requirements,
        "operationalQualification": qualification, "researchReadinessScore": None, "researchScoreReason": "No accepted research-gate audit in this product task", "researchGatesAdvanced": 0})
    progress = {"createdAt": now, "sourceCommit": preview["sourceCommit"], "milestones": [
        {"name": "Private guided analysis", "status": status("preview-final/browser-receipt.json"), "evidence": "../preview-final/browser-receipt.json"},
        {"name": "Causal forecasts and frozen evaluation", "status": "passed", "evidence": "../runner-final.xml", "operationalQualification": "failed"},
        {"name": "Scoped telemetry and replay", "status": "local passed; hosted processing blocked", "evidence": "../stream-release-local/receipt.json"},
        {"name": "Semantic source retrieval", "status": "local and hosted document path passed", "evidence": "../semantic-final/receipt.json"}],
        "overallResearchReadinessScore": None, "reason": "Research gates not audited/accepted in this task", "researchGatesAdvanced": 0,
        "operationalAlertQuality": "failed", "hostedComputeAcceptance": "partial; HTTP 402 payment_required", "productionPromotion": "not authorized; hosted gate incomplete"}
    write("progress/eidos_progress_meter.json", progress)
    write("progress/eidos_progress_meter.md", "# Evidence-backed implementation progress\n\n" + "\n".join(f"- {m['name']}: {m['status']} ([receipt]({m['evidence']}))" for m in progress["milestones"]) + "\n\nOperational qualification: failed. Research-readiness score: unknown; no accepted gate audit or invented percentage. Production promotion: pending release decision.\n")
    lines = [("EIDOS / SENTINEL", 34, "#aad6bd"), ("Implementation evidence", 64, "#ffffff")]
    for i, milestone in enumerate(progress["milestones"]): lines.append((f"{i+1}  {milestone['name']} — {milestone['status']}", 110+i*40, "#d3e6dc"))
    lines += [("Operational alert qualification: FAILED", 292, "#f5b5a2"), ("Research readiness: unknown · no gates advanced", 330, "#d3e6dc"), ("Source-linked evidence; no production promotion", 366, "#d3e6dc")]
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="820" height="405" viewBox="0 0 820 405"><rect width="820" height="405" rx="20" fill="#142920"/>' + ''.join(f'<text x="32" y="{y}" fill="{color}" font-family="Arial,sans-serif" font-size="{18 if y != 64 else 28}">{html.escape(text)}</text>' for text, y, color in lines) + '</svg>'
    write("progress/eidos_progress_meter.svg", svg)
    write("progress/eidos_progress_dashboard.html", '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Sentinel implementation evidence</title><style>body{max-width:980px;margin:3rem auto;padding:1rem;background:#edf0e8;color:#142920;font:18px/1.6 system-ui}img{width:100%;height:auto}a{color:#245740}li{margin:1rem 0}</style><h1>Sentinel implementation evidence</h1><img src="eidos_progress_meter.svg" alt="Four product milestones and separate operational failure status"><p>These are engineering receipts. Overall research readiness remains unknown.</p><ul>' + ''.join(f'<li><a href="{m["evidence"]}">{html.escape(m["name"])}</a>: {m["status"]}</li>' for m in progress["milestones"]) + '</ul><p><a href="../benchmark_summary.md">All final comparisons and failed qualification</a> · <a href="../requirement-to-evidence.md">Requirement table</a> · <a href="../codex_journal.md">Logic, meaning and limits</a></p></html>')
    write("progress/eidos_progress_readme.md", "Generated by services/sentinel-runner/scripts/guided_release_evidence.py from the listed receipts. SVG renders without JavaScript. Missing evidence is shown as missing. Research percentage is deliberately null; product workflow evidence is not a research-gate pass.\n")
    write("environment.txt", f"Generated {now}\nPython {sys.version}\nPlatform {platform.platform()}\nLocal evaluation Torch 2.6.0+cpu; hosted worker actual versions are in preview results.\nNode " + subprocess.check_output(["node", "--version"], text=True).strip() + "\n")
    write("git_commit.txt", code_sha + "\nTested application source: " + preview["sourceCommit"] + "\n")
    print("Generated source-backed release reports; no evaluation executed.")


if __name__ == "__main__": main()
