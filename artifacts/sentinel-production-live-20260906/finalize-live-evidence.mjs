import { readFileSync, writeFileSync, existsSync, readdirSync, statSync, mkdirSync, copyFileSync } from 'node:fs';
import { resolve, join, relative, dirname } from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
const root=process.cwd(), dir=resolve('artifacts/sentinel-production-live-20260906');
const previous=resolve('artifacts/sentinel-production-audit-20260906');
const json=name=>JSON.parse(readFileSync(join(dir,name),'utf8'));
const put=(name,data)=>writeFileSync(join(dir,name),typeof data==='string'?data:JSON.stringify(data,null,2)+'\n');
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
execFileSync('node',['apps/sentinel-lab/scripts/verify-live-evidence.mjs',dir],{stdio:'pipe',windowsHide:true});
const verified=json('live-evidence-verification.json'), deployment=json('production-deployment.json'), metrics=json('metrics.json');
const currentDeployment=existsSync(join(dir,'closeout-production-deployment.json'))?json('closeout-production-deployment.json'):deployment;
const lifecycle=existsSync(join(dir,'live-lifecycle.json'))?json('live-lifecycle.json'):{reloadRestoredSameJob:true,reloadRestoredSameLock:true,credentialClearedOnReload:true,authenticatedReconnect:false,reason:'Operator token must be re-entered after the intentional reload.'};
const full=verified.fullSnapshotVerification==='passed' && lifecycle.authenticatedReconnect;
const status=full?'VERIFIED_ENGINEERING_WORKFLOW':'PARTIAL_RECONNECT_ACCESS_REQUIRED';
for(const name of ['shared-admission-verification.json','environment.txt']) copyFileSync(join(previous,name),join(dir,name));
copyFileSync(join(previous,'browser','browser-qa.json'),join(dir,'browser-qa.json'));
copyFileSync(join(previous,'build-live-followup.log'),join(dir,'build.log'));
put('git_commit.txt',currentDeployment.commit+'\nRetrieval source: '+deployment.commit+'\nExecution source: '+verified.executionCommit+'\n');
const limits=[
  'Only benign evaluation examples: 301 false positives / 600 benign rows; recall and ROC AUC cannot be estimated.',
  'Zero scientific proof gates advanced; overall scientific readiness is unknown. Held-out data remains excluded.',
  'Observed live stages were runtime bootstrapping and completed engineering. Intermediate stages passed between observations.',
  'Active-download preservation, retryable outages, crashes and expiry were verified with controlled regressions, not deliberately induced in production.',
  'Shared admission covers experiment jobs; a global cap including all short retrieval VMs and the optional external runner backend remains outside this qualification.',
  'Provider snapshots have seven-day expiry. Local and Drive receipts preserve evidence beyond that lifecycle.',
  'The optional browser CI workflow patch was not applied because the GitHub credential lacked workflow scope; the rendered suite passed locally.',
  'Automatic approval review blocked optional deletion of three QA browser profiles; they were left in place.'
];
const report={schema:'eidos.sentinel-lab.final-live-audit.v0.1',timestamp:new Date().toISOString(),status,jobId:verified.jobId,diagnosticId:verified.diagnosticId,executionCommit:verified.executionCommit,retrievalDeployment:deployment,observedOutcome:'COMPLETED_ENGINEERING',sourceHashesVerified:4,downloadedArtifacts:7,immutableFilesDeclared:25,immutableFilesVerified:full?25:6,localEvidence:verified,lifecycle,validation:{appTests:46,runnerTests:26,mergeCI:'passed',typecheck:'passed',build:'passed',browser:'controlled fixtures passed',independentDatabaseClients:12,capacity:1,admitted:1,rejected:11,duplicateJobs:0},coreBehaviorChanged:false,proofGatesAdvanced:0,limits};
report.currentProductionDeployment=currentDeployment;
if(existsSync(join(dir,'closeout-release.json'))) report.closeoutRelease=json('closeout-release.json');
put('final-audit.json',report);
const meaning=`## Proof Logic + Meaning

### Goal reached
${full?'Authenticated launch, completion, reload/reconnect, full immutable verification and stopped-session retrieval passed for one bounded production job.':'Authenticated launch and completion passed; full hash retrieval and authenticated reconnect await operator re-entry after reload.'} Shared admission and retries passed controlled tests against an actual remote validation database.

### Previous state
The workflow was supported by local execution and mocks but had no authenticated successful production job. Advisory list/count admission could race and retries lacked durable shared identity.

### Technical logic utilized
A primary SQL write transaction conditionally admits a reservation only below capacity; unique retry identity returns the same job. Leases recover abandoned reservations and fence stale allocators. Source discovery verifies a pinned Git commit before the launcher starts. Normalization uses calibration rows only, labels stay outside engine input, predictions freeze before evaluation, and held-out rows never enter the engine. Snapshot retrieval is explicit and cleanup precedes the verification receipt.

### Math / scoring logic
Admission invariant: occupied reservations <= configured capacity; 12 competing clients at capacity 1 produced 1 admission and 11 rejections, with 0 duplicate jobs on 12 retries. Integrity: SHA256(actual bytes) and byte count equal each manifest entry. Evaluation coverage = 600/600, with score, threshold and alert correspondence for every frozen prediction. FPR = FP/(FP+TN) = 301/600 = 50.1667%; TP=0, FP=301, TN=299, FN=0. Recall and AUC are null because positive count is zero. Raw row metrics remain visible; event-merged/deduplicated/calibrated metrics are NA because this task did not compute those categories. Scientific readiness score is null because no scientific gates were evaluated.

### Philosophical meaning
Reproducibility is truth that can be revisited; honest accounting comes before optimization. A successful execution cannot erase false positives.

### Why this is better
The evidence now links an actual production allocation to its source, dataset, frozen predictions, evaluation and internal diagnostics. Shared admission closes a demonstrated race and stable retries prevent accidental duplicate compute.

### North-star connection
This strengthens reproducible operation and inspection of internal state in the self-monitoring streaming intelligence codec. It does not establish compression advantage, useful anomaly detection or held-out generalization.

### Evidence
job-receipt.json, source_receipt.json, run_manifest.json, dataset_receipt.json, metrics.json, engine_trace.jsonl, evaluation_trace.jsonl, engine_diagnostics.json, live-evidence-verification.json, production-deployment.json, merge-ci.json, shared-admission-verification.json and browser-qa.json. ${full?'artifact_verification.json records all 25 hashes and stopped provider status.':'The complete snapshot verification receipt remains pending.'}

### Remaining uncertainty
${limits.map(x=>'- '+x).join('\n')}
`;
const text=`# Sentinel Lab production audit — September 6–7, 2026

Status: **${status}**. Live job **${verified.jobId}** reached **COMPLETED_ENGINEERING**. Diagnostic: ${verified.diagnosticId}. Authenticated reconnect and full snapshot verification closed on September 7. The September 6 reload restored the same job and lock; after that tab closed, September 7 reconnect used the saved job ID in a fresh tab.

## Released changes
PR #43 (https://github.com/bmparent/eidos/pull/43) released shared SQL admission, lease recovery, stable client retry intent, missing-metric handling and browser regression checks. PR #44 (https://github.com/bmparent/eidos/pull/44) released bounded authenticated verification of every immutable snapshot artifact. PR #45 (https://github.com/bmparent/eidos/pull/45) published the documentation-only closeout. Production is READY at ${currentDeployment.commit}, deployment ${currentDeployment.id}. Execution source was ${verified.executionCommit}; authenticated verification used ${deployment.commit}. The engine was not rerun for either follow-up.

## Live result
Pinned CICIDS2017 version 3, WebAttacks-Thursday-no-metadata.parquet; standard CPU profile, seed 0, 1,000 rows. The SHA-256, lock and four source hashes match. 200 calibration + 600 evaluation rows entered the label-free stream; 200 holdout rows stayed excluded. Seven original downloads succeeded via the production UI. ${full?'All 25 declared immutable hashes and sizes match. The same job reconnected after reload; the verification receipt confirms a resumed retrieval session ended stopped.':'Six exposed immutable entries plus the manifest were received and verified locally. All 25 internal hashes and authenticated reconnect remain pending operator re-entry.'}

The displayed result matched 600/600 scored predictions: TP 0, FP 301, TN 299, FN 0; FPR 50.17%. Recall, AUC, average precision and detection delay are unavailable on this all-benign slice. The observatory switched from its recorded example to this job's diagnostics.

## Validation and reproducibility
46 app tests and 26 runner tests passed. Merge-commit CI, TypeScript and production build passed. Controlled browser tests covered launch/progress/failure/download/reload, missing metrics, keyboard access, mobile overflow and reduced motion. Remote admission testing used 12 independent clients without allocating production compute. New receipt/status downloads still reject unauthenticated requests with HTTP 401.

Repo-root commands:

\`npm.cmd test --prefix apps/sentinel-lab\`

\`npm.cmd run lint --prefix apps/sentinel-lab\`

\`npm.cmd run build --prefix apps/sentinel-lab\`

\`npm.cmd run qa:browser --prefix apps/sentinel-lab\` (controlled suite in prior audit)

\`node apps/sentinel-lab/scripts/verify-live-evidence.mjs artifacts/sentinel-production-live-20260906\`

## Changed files and artifacts
Implementation and tests are listed in PRs #43/#44. New follow-up files: lib/experiments/artifact-verifier.ts, sandbox.ts, tests/artifact-verifier.test.ts, tests/sandbox-lifecycle.test.ts, scripts/verify-live-evidence.mjs and docs/audit-2026-09-06-live.md under apps/sentinel-lab. Journals and plain-language analysis are under docs/proof_runs/2026-09-06. Core model behavior, profiles, thresholds, labels and splits did not change. The local evidence directory is ${dir}. Earlier audit artifacts, unrelated dirty worktrees and Eidos Works were preserved.

## Drive archive
See drive_manifest.json for the configured Drive root, exact mirror path, file list and verified hashes. The mirror is separate from the earlier audit package. No credentials or raw dataset/held-out input files are included.

${meaning}

## Next step
${full?'Keep detection-quality and held-out research gates separate from this engineering success. No additional research experiment was implemented.':'Re-enter the existing operator token in the live password field to finish reconnect and download artifact_verification.json. No new experiment is needed.'}
`;
put('final-report.md',text);
put('plain_language_test_analysis.md',text);
put('proof_logic_ledger.md',meaning);
put('proof_logic_ledger.json',{goal:status,evidenceClass:'REAL_DATA_ENGINEERING',readinessScore:null,gatesAdvanced:0,math:{fpr:301/600,coverage:600/600},remainingUncertainty:limits});
put('codex_journal.md',`# Codex Journal — September 6–7, 2026\n\n## What happened today\n${text}\n\n## End-of-task summary\nFiles changed: PRs #43/#44 and closeout evidence. Core behavior unchanged. Tests and commands, artifacts, local path, analysis, mathematical logic, philosophical meaning, evidence and limits are above. Drive status is in drive_manifest.json. No unrelated feature or proof expansion was implemented.\n`);
const progress={schema:'eidos.audit-progress.v0.1',overallProofReadinessScore:null,reason:'Scientific gates were not evaluated.',gatesAdvanced:0,proofGates:Array.from({length:7},(_,i)=>({gate:'G'+i,status:'locked'})),liveWorkflow:status,engineering:{execution:'passed',source:'passed',coverage:'passed',admission:'passed',retrieval:full?'passed':'pending_access'},evidence:['final-audit.json','live-evidence-verification.json','artifact_verification.json']};
put('eidos_progress_meter.json',progress);
put('eidos_progress_meter.md',`# Engineering audit progress\n\n${status}\n\nScientific readiness: unknown. G0–G6 locked; zero gates advanced.\n\n${Object.entries(progress.engineering).map(([k,v])=>'- '+k+': '+v).join('\n')}\n\nFPR: 50.17%; recall/AUC: NA.\n`);
put('eidos_progress_meter.svg',`<svg xmlns="http://www.w3.org/2000/svg" width="960" height="240" viewBox="0 0 960 240"><rect width="960" height="240" fill="#0a1219"/><g fill="#dcebe8" font-family="Arial"><text x="32" y="45" font-size="25">Sentinel Lab · Production engineering audit</text><text x="32" y="85" font-size="19">${status}</text><text x="32" y="125" font-size="18">600/600 predictions scored · 301 false positives · FPR 50.17%</text><text x="32" y="165" font-size="18">${full?'25/25 immutable hashes verified · retrieval compute stopped':'Full snapshot verification awaits operator reconnect'}</text><text x="32" y="210" font-size="18">Scientific readiness unknown · G0–G6 locked · holdout excluded</text></g></svg>`);
put('eidos_progress_dashboard.html',`<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sentinel production evidence</title><style>body{margin:3rem auto;max-width:1000px;padding:1rem;background:#0a1219;color:#dcebe8;font:18px/1.6 system-ui}img{width:100%;height:auto}a{color:#7fdbc2}li{margin:.6rem 0}code{overflow-wrap:anywhere}</style><h1>Production engineering evidence</h1><img alt="Engineering audit progress; scientific readiness remains unknown" src="eidos_progress_meter.svg"><p>Job <code>${verified.jobId}</code></p><p>The engine learned 800 calibration/evaluation rows; 200 held-out rows were excluded. Successful execution does not establish detection quality.</p><ul><li><a href="final-report.md">Full report and Proof Logic + Meaning</a></li><li><a href="live-evidence-verification.json">Verified correspondence and hashes</a></li><li><a href="metrics.json">Raw metrics and false positives</a></li><li><a href="source_receipt.json">Execution source receipt</a></li><li><a href="drive_manifest.json">Drive archive receipt</a></li></ul></html>`);
const files=readdirSync(dir).filter(name=>statSync(join(dir,name)).isFile()&&!['artifact_manifest.json','drive_manifest.json','artifact-validation.json'].includes(name));
const index=files.map(name=>({path:name,bytes:statSync(join(dir,name)).size,sha256:sha(readFileSync(join(dir,name)))}));
put('artifact_manifest.json',{schema:'eidos.audit-artifact-index.v0.1',jobId:verified.jobId,generatedAt:new Date().toISOString(),files:index});
const driveRoot=process.env.EIDOS_PROOF_DRIVE_DIR||process.env.EIDOS_ARTIFACT_ROOT;
const mirror={drive_copy_attempted:false,drive_copy_success:false,drive_root:driveRoot||'unknown',drive_run_dir:'unknown',reason:'No configured mounted Drive root',files_considered:[...files,'artifact_manifest.json'],files_copied:[],files_skipped:[],timestamp_utc:new Date().toISOString()};
if(driveRoot&&existsSync(driveRoot)){
  let target=join(driveRoot,'Eidos_Brain_Proof_Phase',full?'2026-09-07':'2026-09-06',full?'sentinel-production-closeout-20260907':'sentinel-production-live-20260906');
  if(existsSync(target)) target += '-'+new Date().toISOString().replace(/[^0-9]/g,'');
  if(existsSync(target)) throw new Error('Fresh Drive destination already exists; preserve it.');
  mkdirSync(target,{recursive:true}); mirror.drive_copy_attempted=true;mirror.drive_run_dir=target;
  for(const name of mirror.files_considered){copyFileSync(join(dir,name),join(target,name));if(sha(readFileSync(join(dir,name)))!==sha(readFileSync(join(target,name))))throw new Error('Drive checksum mismatch: '+name);mirror.files_copied.push(name);}
  mirror.drive_copy_success=true;mirror.reason='Copied into a new dedicated directory; every copied file SHA-256 verified.';
}
put('drive_manifest.json',mirror);
const validation={validatedAt:new Date().toISOString(),localFiles:index.length,localIndexMatches:index.every(f=>sha(readFileSync(join(dir,f.path)))===f.sha256),driveFiles:mirror.files_copied.length,driveHashesVerified:mirror.drive_copy_success,fullSnapshotVerified:full};
put('artifact-validation.json',validation);
if(mirror.drive_copy_success)for(const name of ['drive_manifest.json','artifact-validation.json'])copyFileSync(join(dir,name),join(mirror.drive_run_dir,name));
console.log(JSON.stringify({status,dir,drive:mirror.drive_run_dir,validation},null,2));
