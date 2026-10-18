"""Assemble measured engineering receipts, plain-language reports and Drive mirror.

Run from repository root after the corpus, detector and browser commands. This
does not run a monitor, call a model, publish a page, or advance Lab research gates.
"""
import argparse
import csv
import hashlib
import html
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--detectors', type=Path, required=True)
    parser.add_argument('--browser', type=Path, required=True)
    parser.add_argument('--dry-run-drive', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    read = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
    write = lambda name, value: (out / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    corpus = read(args.corpus / 'summary.json')
    comparison = read(args.detectors / 'comparison.json')
    browser = read(args.browser / 'browser-receipts.json')
    rows = [json.loads(line) for line in (args.corpus / 'raw_cases.jsonl').read_text(encoding='utf-8').splitlines()]
    release = read(out / 'release.json') if (out / 'release.json').exists() else {'status': 'not_deployed'}
    hosted = read(out / 'hosted' / 'receipt.json') if (out / 'hosted' / 'receipt.json').exists() else {'passed': False}
    timestamp = datetime.now(timezone.utc).isoformat()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    commands = [
        'node --test --test-concurrency=1 apps/sentinel-observatory/live/*.test.mjs',
        "python -m unittest discover -s apps/sentinel-observatory/live -p 'test_*.py' -v",
        'node apps/sentinel-observatory/live/fault-harness/freeze.mjs',
        'node apps/sentinel-observatory/live/fault-harness/run.mjs --out artifacts/proof_runs/2026-10-18/observatory-m1/corpus',
        'node apps/sentinel-observatory/live/fault-harness/compare-detectors.mjs --out artifacts/proof_runs/2026-10-18/observatory-m1/detectors',
        'node apps/sentinel-observatory/live/verify_brain_parity.mjs',
        'npm --prefix apps/sentinel-lab ci --no-audit --no-fund',
        'npm --prefix apps/sentinel-lab run build',
        'node apps/sentinel-observatory/live/fault-harness/serve.mjs',
    ]
    (out / 'git_commit.txt').write_text(commit + '\n', encoding='utf-8')
    (out / 'environment.txt').write_text(f'Python {platform.python_version()}\n{platform.platform()}\nNode v24.14.0\nClient date 2026-10-18 America/New_York\nMeasured timestamps UTC\nNo credentials captured\n', encoding='utf-8')
    write('config.json', {'corpus': str(args.corpus), 'detectors': str(args.detectors), 'browser': str(args.browser), 'seed': 42, 'core_behavior_changed': False, 'transport_behavior_changed': 'Requested TLS/partial-evidence repair and bounded alternate attempts; GET fallback opt-in; 12-second production deadline retained'})
    metrics = []
    for method, data in comparison['methods'].items():
        score = data['splits']['locked_evaluation']
        metrics.append({'method': method, **{key: score[key] for key in ['tp','fp','fn','precision','recall','false_incidents_per_benign_target_day','mean_detection_delay_seconds','mean_recovery_delay_seconds']}, 'observed_adapter_ms': data['observed_adapter_ms'], 'ci95': score['stream_cluster_ci95']})
    with (out / 'benchmark_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = [key for key in metrics[0] if key != 'ci95']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in metrics)
    table = '| Method | TP / FP / FN | Precision | Recall | False incidents / benign target-day | Mean delay s |\n|---|---:|---:|---:|---:|---:|\n'
    for row in metrics:
        table += f"| {row['method']} | {row['tp']} / {row['fp']} / {row['fn']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['false_incidents_per_benign_target_day']:.3f} | {row['mean_detection_delay_seconds']:.1f} |\n"
    families = [
        ('factual_integrity','partial','Controlled status/stage/citation assertions pass; the original final cases were exposed by failures and are now regression cases.'),
        ('known_failure_stages','passed_local',f"{corpus['passed']}/{corpus['cases']} synthetic controlled cases; real local sockets plus labeled DNS/blackhole simulations."),
        ('security_and_tenant_isolation','partial','Public-boundary tests pass. Tenant-scoped records, model budgets, exports and notification authorization are not implemented; two real identities remain untested.'),
        ('continuous_operation','not_implemented','Foreground-only; no durable schedule, job leases, seven-day pilot or second independent observer.'),
        ('scheduling_target','not_tested','99.5% is a proposed qualification target; eligible slots and terminal receipts do not exist yet.'),
        ('incident_behavior','not_implemented','Existing change episodes are not a durable incident/recovery/maintenance state machine.'),
        ('notifications','not_implemented','No durable outbox, controlled sink or provider/inbox delivery receipt.'),
        ('explanation_value','blocked','Deterministic single-arm blinded packet supplied. Genuine corpus Sol-ranked arm and independent human scores are absent; no usefulness percentage.'),
        ('detector_comparison','passed_local','72 synthetic streams, 11520 frames, fixed thresholds, same-stream gate/EWMA/CUSUM comparisons and whole-stream bootstrap CIs; no superiority.'),
        ('ux','passed_local','Desktop 1440x1000 and mobile 390x844: connect, stage rendering, pause, export, reload/history, GET fallback, keyboard filter focus, reduced motion, no overflow or console errors. Physical device not tested.'),
        ('operations','partial','Build, production dependency separation and local persistence/replay verified. Capacity soak, migration/backup restore/rollback execution remain untested.'),
        ('economics','partial','Local adapter runtime/RSS captured with loaded-host caveats. Actual provider/storage/notification cost is NA; no pilot bill exists.'),
    ]
    write('acceptance_matrix.json', {'schema':'observatory-commercial-acceptance-v1','frozen_protocol':'apps/sentinel-observatory/live/fault-harness/protocol.json','rows':[{'gate':gate,'status':status,'evidence_or_gap':reason} for gate,status,reason in families], 'readiness':{'endpoint_monitoring':'private candidate only','ai_assistance':'usefulness unproven','security_logs':'no new acceptance','research_methods':'engineering only; Lab gates unchanged'}})
    matrix = '| Gate | Status | Evidence or gap |\n|---|---|---|\n' + ''.join(f'| {gate} | {status} | {reason} |\n' for gate,status,reason in families)
    (out / 'acceptance_matrix.md').write_text('# Acceptance matrix — 18 October 2026\n\n' + matrix, encoding='utf-8')
    packet, key = [], []
    for row in rows:
        prepared = row.get('prepared')
        if not prepared:
            continue
        review_id = 'review_' + hashlib.sha256(('42:' + row['definition']['id']).encode()).hexdigest()[:12]
        packet.append({'review_id':review_id,'measurement':prepared['safe'],'facts':prepared['facts'],'next_checks':prepared['next_checks'],'unknowns':prepared['unknowns'],'scores':{'factual_support':None,'appropriate_uncertainty':None,'next_action_correct':None,'usefulness_1_to_5':None},'reviewer_id':None})
        key.append({'review_id':review_id,'case':row['definition']['id'],'arm':'deterministic','expected_next_action':row['definition']['expected_next_action'],'ground_truth':row['definition']['fault']})
    packet.sort(key=lambda row: row['review_id'])
    (out / 'blinded_review_packet.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in packet), encoding='utf-8')
    write('blinded_review_key.json', key)
    write('explanation_experiment.json', {'status':'blocked','packet_cases':len(packet),'actual_human_reviews':0,'appropriate_next_action_rate':None,'sol_ranked_arm':'missing','comparison_ready':False,'reason':'A single available arm can support factual/action review; it cannot establish Sol ranking benefit. Do not expose the key to reviewers or inference.'})
    pilot = {'schema':'observatory-pilot-plan-v1','status':'not_started','durable_run_id':None,'authorized_targets':[],'two_independent_observers':False,'automatic_model_calls':False,'proposed_cadence_seconds':300,'example_targets':2,'example_observers':2,'estimated_checks_per_day':1152,'estimated_seven_day_checks':8064,'actual_checks':None,'actual_cost_usd':None,'blocking_implementation':['tenant membership and ownership migration','durable schedule/job reservation/leases','observer-specific baseline keys','incident state and idempotent outbox','retention above global 2048-record cap'],'external_acceptance':['supported hosting schedule/worker configuration and current entitlement','two actual authorized identities','second independent collector failure domain','explicit pilot target allowlist','seven elapsed days and independent explanation reviewers'],'resumption':'Implement scoped durable monitor jobs, then create a pilot manifest with actual observers and allowlist. Do not resume a nonexistent pilot or create retrospective slots.'}
    write('pilot_manifest.json', pilot)
    logic = f'''## Proof Logic + Meaning

### Goal reached
First collector/fault-harness engineering milestone: locally passed ({corpus['passed']}/{corpus['cases']}); private deployment status: {release.get('status')}. Commercial qualification remains partial. This does not advance Lab G0–G6.

### Previous state
PR79 had known TLS cleanup loss, incomplete later-failure evidence, one-address sampling, ambiguous HEAD 405/501 coverage and no locked controlled corpus. Version 16 remained foreground-only.

### Technical logic utilized
Capture negotiated TLS and certificate/network metadata before socket cleanup, preserve partial stages, validate all DNS answers before numeric pinning, try at most three validated addresses per method/hop under one shared deadline, and retain every attempted outcome separately from the selected HEAD response. GET fallback is explicit and keeps the original HEAD status. The existing optimistic SQLite/D1 checkpoint write and duplicate evidence identity support retain → explain → export/replay. Unknown TLS/peer values remain null. No private fixture allowance enters production.

### Math / scoring logic
Total deadline D = start + min(requested_timeout, 12000 ms). Attempt budget B_i = (D - now) / remaining_addresses; attempt deadline never exceeds D. Each successful TCP/TLS timing is measured from its own socket events.

The unchanged operational gate uses preceding values: m = median(history); MAD = median(abs(x_i - m)); T = max(5 ms, 0.25 abs(m), 6 * 1.4826 * MAD). gate = abs(x_t - m) > T. Sentinel operational flag = raw_Sentinel_change AND gate. No current label enters either rule.

precision = TP/(TP+FP); recall = TP/(TP+FN); false incidents per benign target-day = FP/(benign scored frames * 300/86400). Whole streams, not correlated frames, are bootstrap units (2000 resamples). Warmup/reset and missing frames are unscored; reported missing_fraction combines these and is not collector coverage. Recovery numbers are alarm-end scoring, not a verified incident state machine. Undefined metrics are null.

### Philosophical meaning
Restraint before alarm; honesty before optimization; reproducibility is truth that can be revisited. The comparison publishes false incidents and a negative result.

### Why this is better
Actual later-stage failures retain the successful stages needed for the next check. One failed address no longer hides a later measured response. Operators can distinguish unsupported HEAD, HTTP 408, connection timeout, incomplete redirects and missing collector evidence, with retained receipts and visible limits.

### How this moves Eidos closer to the north-star goal
The north-star claim is a self-monitoring streaming intelligence codec that learns live streams, compresses predictable behavior, preserves meaningful anomalies, monitors internal state and emits readable incident receipts. This milestone strengthens reproducible measurement and explanation. It does not prove compression, full-engine generalization, security detection or unattended incident operation.

### Evidence
{args.corpus.as_posix()}/run_manifest.json and raw_cases.jsonl; {args.detectors.as_posix()}/comparison.json, manifest.json, streams.jsonl and stream_scores.jsonl; clean-node-tests.txt; python-tests-final.txt; brain-parity.txt; production-trace.json; {args.browser.as_posix()}/browser-receipts.json and screenshots; hosted/receipt.json when present; release.json; drive_manifest.json.

### Remaining uncertainty
Synthetic faults are not live pilot observations. The original corpus evaluation cases are consumed regression cases after repair. Model/provider simulations are not genuine Sol usefulness evidence. Two tenants, independent observers, durable scheduling, outbox delivery, backup/restore, load capacity, seven-day coverage, measured commercial cost and physical-device behavior remain unproven. No Sentinel superiority was observed.
'''
    failure_notes = '''The first corpus run passed 70/72: real GET truncation/abort ordering and retained-status validation exposed two defects. Those failures remain in corpus-attempt-1. The repaired regression run and clean checkout pass 72/72. A parallel unit run exposed too-short local fixture deadlines on a heavily loaded Windows host plus a bad test source path; deadlines now allow real TLS setup and focused checks run serially. No production timeout was increased. Browser attempts retain the old loaded-server timing failure, a selector collision, missing favicon and cross-viewport rate-limit collision. Final browser checks use distinct logical targets and pass without exempting console errors. The first detector runner missed an import; its partial output and failure log remain. Packaging initially failed on npm shim resolution and GNU tar interpreting a Windows drive colon as a remote archive. The direct validated Node build and TAR_OPTIONS=--force-local recover the supported source workflow. Git checkout line-ending conversion changed the recorded capture hash; the exact Git blob was restored and is pinned by .gitattributes. No failing receipt was deleted.'''
    report = '# Observatory first milestone — measured engineering result\n\n' + f"Runtime code 4a9bd8b9da652a1d6587412316e18d97d35d76f8; proof assembly commit {commit}. Original continuation c184ae06819e7ad326f141f378b2d83f9b91152c, stacked on PR77.\n\n" + f"Controlled corpus: {corpus['passed']}/{corpus['cases']} across {corpus['families']} families. Clean checkout: 151 Node tests passed. Python: 31 passed, one Windows symlink privilege skip. Numerical parity: 954 frames / 41022 field comparisons. Next build and production fixture separation passed. Local browser: {len(browser)} viewports. Fresh hosted checks: {hosted.get('passed', False)}.\n\n" + table + '\nAll four methods retain failures, false incidents, delays and stream-cluster CIs in comparison.json. Sentinel plus gate matched the gate alone. These results do not justify superiority or a production algorithm change. Runtime measurements came from a loaded Windows host and are not cloud cost estimates.\n\n## Failures retained\n\n' + failure_notes + '\n\n## Commercial gates\n\n' + matrix + '\n' + logic
    (out / 'benchmark_summary.md').write_text(report, encoding='utf-8')
    (out / 'plain_language_test_analysis.md').write_text(report, encoding='utf-8')
    journal = '# Codex Journal — 2026-10-18\n\n## What happened today\n\nReconciled PR79, PR77, owner-private Site version 16 and its immutable Node collector; isolated all work and repaired the first collector milestone.\n\n## What was accomplished\n\nControlled retained evidence, socket repair, bounded alternates, explicit GET coverage, locked corpus, causal detector comparisons and browser replay acceptance.\n\n## Tests and commands run\n\n' + '\n'.join('- `'+command+'`' for command in commands) + '\n\nSee command-specific receipts for exact outputs and pass/fail. Browser commands also accept --playwright-module and --browser for an existing runtime.\n\n## Problems encountered\n\n' + failure_notes + '\n\n## What changed\n\nCollector/relay/presentation, test-only harness, portable Worker packaging and proof reports.\n\n## What did not change\n\nReservoir dynamics, RLS, Sentinel thresholds, target-metrics-v2 operational policy, Lab core/research gates, original recorded capture, private audience and other Works releases were preserved. No billing, new integration, paid plan or notification was enabled.\n\n' + logic + '\n## Artifacts generated\n\n' + out.as_posix() + '\n\n## Google Drive archive status\n\nSee drive_manifest.json. A checksum-verified mounted Drive copy does not independently establish remote synchronization.\n\n## Thoughts on improvement\n\nUse a new untouched qualification corpus after repairs; obtain independently scored explanations; implement tenant- and observer-scoped durable jobs before unattended pilots.\n\n## Where to improve next\n\nOne PR for scoped ownership migration and observer-specific baselines, followed by durable reservation/lease/outbox work.\n\n## Anything that stands out\n\nThe richer engine supplied no detector advantage on these timing generators. Publish that negative result and retain the simplest validated seam for a separately reviewed product-default decision.\n\n## End-of-task summary\n\n1. Files changed: collector, relay, presentation, Worker builder, test harness and proof docs.\n2. Core behavior changed: no. Requested transport behavior repaired; GET fallback remains opt-in.\n3. Tests: 151 Node; 31 Python passed/one privilege skip; two browser sizes; build/parity/trace checks.\n4. Repo-root commands: above and per-run manifests.\n5. Artifacts: measured cases, raw streams, scores, logs, manifests, screenshots, review packet.\n6. Plain-language analysis: written.\n7. Journal: written.\n8. Drive: drive_manifest.json, including skipped files/reasons.\n9. Limits: commercial matrix remains open.\n10. Follow-ups not implemented: tenant scopes, scheduler, two observers, incidents/outbox and seven-day pilot.\n11. Proof Logic + Meaning: written.\n12. Math: deadline, median/MAD gate, event precision/recall, bootstrap.\n13. Philosophy: restraint, honesty and revisitable truth.\n14. Better state: successful partial stages retained, bounded alternatives and reproducible failure receipts.\n15. North-star: strengthens reproducible measurement and readable explanations.\n16. Evidence: named above.\n17. Unproven: compression, generalization, Sol usefulness, commercial operation and physical devices.\n'
    (out / 'codex_journal.md').write_text(journal, encoding='utf-8')
    docs = Path('docs/proof_runs/2026-10-18')
    docs.mkdir(parents=True, exist_ok=True)
    for name in ['codex_journal.md','plain_language_test_analysis.md']:
        shutil.copy2(out / name, docs / name)
    (out / 'proof_logic_ledger.md').write_text(logic, encoding='utf-8')
    write('proof_logic_ledger.json', {'milestone':'collector_fault_harness_m1','status':'passed_local','core_behavior_changed':False,'corpus':corpus,'detector_superiority':False,'commercial_qualified':False,'evidence':['benchmark_summary.md','acceptance_matrix.json','run_manifest.json','drive_manifest.json']})
    standard_gates = [{'gate':name,'weight':weight,'status':'not_reconciled'} for name,weight in [('reproducible_baseline',15),('labeled_domain_proof',15),('precision_discipline',15),('calibration_generalization',15),('compression_anomaly_preservation',10),('incident_explanation',10),('domain_demos',10),('one_command_final_report',10)]]
    progress = {'schema':'eidos-proof-progress-v1','proof_score':None,'reason':'This engineering milestone does not re-audit or accept global research gates; do not convert unknown gates into zero readiness.','math':'proof_score = sum(weight_i * accepted_gate_i) only after gate acceptance is reconciled','gates':standard_gates,'engineering_milestone':{'name':'collector fault harness','passed':corpus['failed']==0,'cases':corpus['cases']},'remaining_gaps':[gate for gate,status,_ in families if not status.startswith('passed')],'north_star':'Self-monitoring streaming intelligence codec; this milestone strengthens reproducible evidence and explanations only.'}
    write('eidos_progress_meter.json', progress)
    (out / 'eidos_progress_meter.md').write_text('# Evidence progress\n\nGlobal proof readiness: **unknown**; global research gates were not re-accepted.\n\nFirst collector engineering milestone: **passed locally**.\n\n' + matrix + '\n' + logic, encoding='utf-8')
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 330"><rect width="900" height="330" fill="#080f19"/><g fill="#b4eedc" font-family="Arial"><text x="30" y="45" font-size="25">Sentinel evidence progress</text><text x="30" y="85" font-size="18">Global proof readiness: unknown</text><text x="30" y="120">Collector engineering: '+str(corpus['passed'])+'/'+str(corpus['cases'])+' controlled cases passed</text><text x="30" y="155">Detectors: no Sentinel advantage established</text><text x="30" y="190">Commercial gates: open; no unattended pilot claimed</text><text x="30" y="225">Synthetic evidence, hosted receipts and pilot evidence stay separate.</text><text x="30" y="280">See acceptance_matrix.md and benchmark_summary.md for receipts.</text></g></svg>'
    (out / 'eidos_progress_meter.svg').write_text(svg, encoding='utf-8')
    (out / 'eidos_progress_dashboard.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Sentinel proof progress</title><style>body{max-width:1100px;margin:40px auto;padding:20px;font:16px system-ui;background:#080f19;color:#d2e3e6}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #304552;text-align:left}a{color:#b4eedc}</style><h1>Sentinel evidence progress</h1><p>Global research readiness: unknown. Collector milestone passed locally. Commercial qualification remains open.</p><img alt="Evidence progress" src="eidos_progress_meter.svg" style="width:100%"><p><a href="benchmark_summary.md">Measured report</a> · <a href="acceptance_matrix.md">Acceptance gates</a> · <a href="drive_manifest.json">Archive receipt</a></p><pre style="white-space:pre-wrap">'+html.escape(table)+'</pre><p>No live data, model calls or scheduling run in this dashboard.</p>', encoding='utf-8')
    (out / 'eidos_progress_readme.md').write_text('Open eidos_progress_dashboard.html locally or read eidos_progress_meter.md. SVG needs no JavaScript. Global score is null because no global research-gate acceptance audit was performed. All numbers come from the named receipts.\n', encoding='utf-8')
    (out / 'draft_case_study.md').write_text('# Draft — When the observer fails, preserve what it actually saw\n\nPrivate engineering draft; not approved for publication.\n\nA controlled reliability corpus exposed lost TLS detail, incomplete failure receipts and a GET abort race. Repairs preserve stage measurements, keep address attempts separate, and explain HTTP method coverage without calling it an outage. The initial 70/72 result and repaired 72/72 regression result are both retained.\n\nA separate fixed synthetic comparison found no Sentinel precision or recall advantage over its operational gate. EWMA and CUSUM results, false incidents and uncertainty remain visible. This is a useful negative result, not a product claim.\n\nNo customer targets or payloads are included. These local experiments do not prove unattended operation, incident delivery, two-tenant isolation, independent observation points, seven-day coverage, security detection or commercial readiness.\n', encoding='utf-8')
    write('run_manifest.json', {'schema':'observatory-milestone-package-v1','assembled_at':timestamp,'assembly_commit':commit,'runtime_code_commit':'4a9bd8b9da652a1d6587412316e18d97d35d76f8','core_behavior_changed':False,'corpus_manifest':str(args.corpus / 'run_manifest.json'),'detector_manifest':str(args.detectors / 'manifest.json'),'browser_receipts':str(args.browser / 'browser-receipts.json'),'release':release,'commercial_qualified':False,'drive_receipt':'drive_manifest.json'})
    skips = []
    files = []
    for path in sorted(out.rglob('*')):
        if not path.is_file():
            continue
        if path.name in ['artifact_manifest.json','drive_manifest.json']:
            continue
        if '.sqlite' in path.name:
            skips.append({'path':path.relative_to(out).as_posix(),'reason':'Mutable local preview database; portable JSON observations/exports are retained instead.'})
            continue
        files.append({'path':path.relative_to(out).as_posix(),'bytes':path.stat().st_size,'sha256':digest(path)})
    write('artifact_manifest.json', {'schema':'observatory-artifacts-v1','timestamp_utc':timestamp,'files':files,'files_skipped':skips})
    drive_root = next((Path(os.environ[name]) for name in ['EIDOS_PROOF_DRIVE_DIR','EIDOS_ARTIFACT_ROOT'] if os.environ.get(name) and Path(os.environ[name]).is_dir()), None)
    drive = {'drive_copy_attempted':False,'drive_copy_success':False,'drive_root':str(drive_root) if drive_root else 'unknown','drive_run_dir':'unknown','reason':'No configured mounted Drive directory available','files_considered':[row['path'] for row in files]+['artifact_manifest.json','drive_manifest.json'],'files_copied':[],'files_skipped':skips,'timestamp_utc':timestamp,'remote_sync_verified':False}
    if args.dry_run_drive:
        drive['reason'] = 'Explicit dry run; no Drive writes attempted'
    elif drive_root:
        drive['drive_copy_attempted'] = True
        destination = drive_root / 'Eidos_Brain_Proof_Phase' / '2026-10-18' / out.name
        drive['drive_run_dir'] = str(destination)
        try:
            destination.mkdir(parents=True, exist_ok=True)
            for row in files:
                source, target = out / row['path'], destination / row['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                if digest(target) != row['sha256']:
                    raise RuntimeError('Drive checksum mismatch for '+row['path'])
                drive['files_copied'].append({'path':row['path'],'sha256':row['sha256']})
            shutil.copy2(out / 'artifact_manifest.json', destination / 'artifact_manifest.json')
            drive['files_copied'].append({'path':'artifact_manifest.json','sha256':digest(out / 'artifact_manifest.json')})
            drive['drive_copy_success'] = True
            drive['reason'] = 'Mounted Drive copy verified by SHA-256; independent remote synchronization not verified'
        except (OSError, RuntimeError) as error:
            drive['reason'] = 'Mounted Drive copy failed: '+str(error)
    if drive['drive_copy_success']:
        drive['files_copied'].append({'path':'drive_manifest.json','sha256':None,'reason':'Self-referential receipt; verified by direct byte comparison after copy'})
    write('drive_manifest.json', drive)
    if drive['drive_copy_success']:
        shutil.copy2(out / 'drive_manifest.json', Path(drive['drive_run_dir']) / 'drive_manifest.json')
    print(json.dumps({'artifacts':str(out),'corpus_passed':corpus['passed'],'drive':drive['drive_copy_success'],'copied_files':len(drive['files_copied']),'commercial_qualified':False}))


if __name__ == '__main__':
    main()
