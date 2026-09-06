"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { EngineObservatory } from "@/components/engine-observatory";
import { ExperimentRunMonitor } from "@/components/experiment-run-monitor";
import { cloneDefaultExperiment } from "@/lib/experiments/shared";
import { ENGINE_PROFILES, engineProfile, reviseDatasetSource, selectDataset } from "@/lib/experiments/profiles";
import { downloadJson, JOB_ID, LabRequestError, requestJson } from "@/lib/experiments/client";
import type { DatasetSearchResult, ExecutionProfile, ExperimentSpec, ExperimentStatus, LockedExperiment, RunnerDispatch } from "@/lib/experiments/types";

function csv(value: string) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function bytes(value: number | null) {
  if (!value) return "SIZE UNLISTED";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let current = value;
  let unit = 0;
  while (current >= 1024 && unit < units.length - 1) {
    current /= 1024;
    unit += 1;
  }
  return `${current.toFixed(current >= 10 ? 0 : 1)} ${units[unit]}`;
}

export function RealDataLab() {
  const [spec, setSpec] = useState<ExperimentSpec>(() => cloneDefaultExperiment() as ExperimentSpec);
  const [query, setQuery] = useState("CICIDS2017");
  const [results, setResults] = useState<DatasetSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchNote, setSearchNote] = useState<string | null>(null);
  const [lock, setLock] = useState<LockedExperiment | null>(null);
  const [preparing, setPreparing] = useState(false);
  const [dispatching, setDispatching] = useState(false);
  const [operatorToken, setOperatorToken] = useState("");
  const [dispatch, setDispatch] = useState<RunnerDispatch | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [resumeId, setResumeId] = useState("");
  const [finished, setFinished] = useState(false);
  const busy = useRef(false);
  const profile = engineProfile(spec);
  const onTerminal = useCallback(() => setFinished(true), []);
  const [engineStatus, setEngineStatus] = useState<ExperimentStatus | null>(null);
  const onStatus = useCallback((status: ExperimentStatus) => setEngineStatus(status), []);

  useEffect(() => {
    try {
      const saved = JSON.parse(sessionStorage.getItem("eidos.lab.active-job") || "null");
      if (saved && JOB_ID.test(saved.jobId)) setDispatch(saved);
    } catch { /* Storage is optional; credentials are never persisted. */ }
  }, []);

  function rememberJob(receipt: RunnerDispatch) {
    setDispatch(receipt);
    setFinished(false);
    setEngineStatus(null);
    try { sessionStorage.setItem("eidos.lab.active-job", JSON.stringify(receipt)); } catch { /* Optional recovery. */ }
  }

  function resumeJob() {
    if (!JOB_ID.test(resumeId.trim())) { setError("Enter a job ID from your receipt, starting with rd-."); return; }
    rememberJob({ jobId: resumeId.trim(), status: "QUEUED", evidenceClass: "REAL_DATA_ENGINEERING", proofVerdict: "BLOCKED_RESOURCE_BEFORE_HELDOUT" });
    setError(null);
  }

  function stopMonitoring() {
    setDispatch(null);
    setEngineStatus(null);
    setLock(null);
    setFinished(false);
    setError(null);
    try { sessionStorage.removeItem("eidos.lab.active-job"); } catch { /* Optional local receipt only. */ }
  }

  function revise(mutator: (current: ExperimentSpec) => ExperimentSpec) {
    setSpec((current) => mutator(current));
    setLock(null);
    setError(null);
  }

  async function searchDatasets() {
    setSearching(true);
    setError(null);
    setSearchNote(null);
    try {
      const body = await requestJson<{ results: DatasetSearchResult[]; warning?: string; mode: string }>(`/api/datasets?q=${encodeURIComponent(query)}`);
      setResults(body.results || []);
      setSearchNote(body.warning || (body.mode === "kaggle" ? "Live public Kaggle catalog results." : "Reviewed protocol fixtures."));
    } catch (searchError) {
      setError(searchError instanceof Error ? searchError.message : "Dataset lookup failed.");
    } finally {
      setSearching(false);
    }
  }

  function chooseDataset(result: DatasetSearchResult) {
    revise((current) => selectDataset(current, result) as ExperimentSpec);
    setResults([]);
    setSearchNote(result.version ? `Selected ${result.ref} at catalog version ${result.version}. Confirm the exact file below.` : `Selected ${result.ref}. Enter an explicit version before locking.`);
  }

  async function prepareLock() {
    if (busy.current) return;
    busy.current = true;
    setPreparing(true);
    setError(null);
    setLock(null);
    try {
      const body = await requestJson<LockedExperiment>("/api/experiments/preflight", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(spec),
      });
      setLock(body);
    } catch (lockError) {
      setLock(null);
      setError(lockError instanceof Error ? lockError.message : "Experiment preflight failed.");
    } finally {
      setPreparing(false);
      busy.current = false;
    }
  }

  async function dispatchRun() {
    if (!lock || busy.current || !operatorToken.trim() || dispatch) return;
    busy.current = true;
    setDispatching(true);
    setError(null);
    try {
      const body = await requestJson<RunnerDispatch>("/api/experiments", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${operatorToken}` },
        body: JSON.stringify({ spec: lock.spec, lockDigest: lock.digest }),
      }, 310_000);
      rememberJob(body);
    } catch (dispatchError) {
      if (dispatchError instanceof LabRequestError && dispatchError.jobId && JOB_ID.test(dispatchError.jobId)) {
        rememberJob({ jobId: dispatchError.jobId, status: "QUEUED", evidenceClass: "REAL_DATA_ENGINEERING", proofVerdict: "BLOCKED_RESOURCE_BEFORE_HELDOUT" });
      }
      setError(dispatchError instanceof Error ? dispatchError.message : "Experiment dispatch failed.");
    } finally {
      setDispatching(false);
      busy.current = false;
    }
  }

  return (
    <section className="real-data-page" aria-labelledby="real-data-title">
      <div className="real-data-intro">
        <div>
          <p className="eyebrow">FULL EIDOS ENGINE · REAL DATA</p>
          <h2 id="real-data-title">Run an experiment</h2>
        </div>
        <div className="real-data-thesis">
          <p>Choose data, review the settings, then launch the engine. You’ll get detection results, memory and regulation diagnostics, and downloadable evidence.</p>
          <div className="class-badges"><span>Engineering evaluation</span><span>Held-out data stays sealed</span></div>
        </div>
      </div>

      {error && <div className="real-error" role="alert"><span>Action needed</span>{error}<button onClick={() => setError(null)} aria-label="Dismiss experiment message">Dismiss</button></div>}

      <div className="real-pipeline" aria-label="Real-data experiment stages">
        <div><b>01</b><span>Choose data & engine</span><small>Start with the pinned example</small></div>
        <div><b>02</b><span>Check readiness</span><small>Review and lock your settings</small></div>
        <div><b>03</b><span>Launch experiment</span><small>Operator access required</small></div>
        <div><b>04</b><span>Inspect results</span><small>Metrics, diagnostics & receipts</small></div>
      </div>

      <div className="real-workspace">
        <div className="experiment-main"><EngineObservatory diagnostics={engineStatus?.engineDiagnostics} jobId={engineStatus?.jobId} /><fieldset className="real-config" disabled={preparing || dispatching}>
          <legend className="sr-only">Experiment settings</legend>
          <div className="starter-dataset"><strong>CIC-IDS2017 · network-flow example</strong><p>A pinned file is ready below. Start with these settings to inspect engine behavior, or expand the source options to choose another dataset.</p><button className="outline-button" onClick={() => revise(() => cloneDefaultExperiment() as ExperimentSpec)}>Use example settings</button></div>
          <label className="profile-picker"><span>Engine profile</span><select value={spec.engine.executionProfile || "cpu_engineering"} onChange={(event) => revise((current) => ({ ...current, engine: { ...current.engine, executionProfile: event.target.value as ExecutionProfile } }))}>{Object.entries(ENGINE_PROFILES).map(([key, value]) => <option key={key} value={key}>{value.label}</option>)}</select></label>
          <p className="profile-description">{profile.description}</p>
          <dl className="profile-facts"><div><dt>Reservoir units</dt><dd>{profile.reservoir.toLocaleString()}</dd></div><div><dt>Memory dimensions</dt><dd>{profile.hippocampus_dim.toLocaleString()}</dd></div><div><dt>Time scales</dt><dd>{profile.fractal_bands}</dd></div><div><dt>TraceSeal</dt><dd>{profile.trace_seal_enabled ? "On · study" : "Off"}</dd></div></dl>
          <details className="source-options"><summary>Dataset & advanced settings</summary>
          <div className="real-section-head"><div><p className="eyebrow">DATASET LOOKUP</p><h3>Pin the source</h3></div><span>KAGGLE</span></div>
          <div className="dataset-search"><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search public Kaggle datasets" aria-label="Search Kaggle datasets" /><button onClick={searchDatasets} disabled={searching}>{searching ? "Looking…" : "Lookup"}</button></div>
          {searchNote && <p className="search-note">{searchNote}</p>}
          {results.length > 0 && <div className="dataset-results">{results.map((result) => <button key={`${result.source}-${result.ref}`} onClick={() => chooseDataset(result)}><span>{result.source.toUpperCase()} · {result.version ? `V${result.version}` : "VERSION REQUIRED"}</span><strong>{result.title}</strong><small>{result.ref} · {bytes(result.totalBytes)}</small></button>)}</div>}

          <div className="real-form-grid">
            <label className="wide"><span>DATASET HANDLE</span><input value={spec.dataset.ref} onChange={(event) => revise((current) => reviseDatasetSource(current, { ref: event.target.value }) as ExperimentSpec)} /></label>
            <label><span>PINNED VERSION</span><input type="number" min={1} value={spec.dataset.version || ""} onChange={(event) => revise((current) => reviseDatasetSource(current, { version: Number(event.target.value) }) as ExperimentSpec)} /></label>
            <label><span>ENGINEERING SEED</span><select value={spec.engine.seed} onChange={(event) => revise((current) => ({ ...current, engine: { ...current.engine, seed: Number(event.target.value) as 0 | 1 } }))}><option value={0}>0</option><option value={1}>1</option></select></label>
            <label className="wide"><span>EXACT FILE PATH</span><input value={spec.dataset.file} onChange={(event) => revise((current) => reviseDatasetSource(current, { file: event.target.value }) as ExperimentSpec)} /></label>
            <label><span>LABEL COLUMN</span><input value={spec.dataContract.labelColumn} onChange={(event) => revise((current) => ({ ...current, dataContract: { ...current.dataContract, labelColumn: event.target.value } }))} /></label>
            <label><span>BENIGN LABELS</span><input value={spec.dataContract.negativeLabels.join(", ")} onChange={(event) => revise((current) => ({ ...current, dataContract: { ...current.dataContract, negativeLabels: csv(event.target.value) } }))} /></label>
            <label><span>ROW ORDER</span><select value={spec.dataContract.orderMode} onChange={(event) => revise((current) => ({ ...current, dataContract: { ...current.dataContract, orderMode: event.target.value as "source" | "column", ...(event.target.value === "source" ? { orderColumn: undefined } : {}) } }))}><option value="source">Source order</option><option value="column">Sort by column</option></select></label>
            <label><span>ORDER COLUMN</span><input disabled={spec.dataContract.orderMode === "source"} value={spec.dataContract.orderColumn || ""} placeholder={spec.dataContract.orderMode === "source" ? "Not used" : "Timestamp"} onChange={(event) => revise((current) => ({ ...current, dataContract: { ...current.dataContract, orderColumn: event.target.value } }))} /></label>
            <label><span>MAX ROWS</span><input type="number" min={1000} max={2000000} step={1000} value={spec.dataContract.maxRows} onChange={(event) => revise((current) => ({ ...current, dataContract: { ...current.dataContract, maxRows: Number(event.target.value) } }))} /></label>
            <label className="wide"><span>EXCLUDED COLUMNS</span><input value={spec.dataContract.excludedColumns.join(", ")} onChange={(event) => revise((current) => ({ ...current, dataContract: { ...current.dataContract, excludedColumns: csv(event.target.value) } }))} /></label>
            <label className="wide"><span>EXPLICIT FEATURE COLUMNS <i>optional</i></span><input value={spec.dataContract.featureColumns.join(", ")} placeholder="Blank = numeric columns with label-like fields rejected" onChange={(event) => revise((current) => ({ ...current, dataContract: { ...current.dataContract, featureColumns: csv(event.target.value) } }))} /></label>
            <label className="wide"><span>EXPECTED FILE SHA-256 <i>optional first-run pin</i></span><input value={spec.dataset.expectedSha256 || ""} placeholder="Runner records the digest when blank" onChange={(event) => revise((current) => ({ ...current, dataset: { ...current.dataset, expectedSha256: event.target.value || undefined } }))} /></label>
          </div>
          </details>
          <p className="source-order-note">{spec.dataContract.orderMode === "source" ? "File order is preserved, but chronology is unverified." : "Rows will be sorted by your locked ordering column. Verify its meaning and units before interpreting detection delays."} This experiment can test engine behavior; it cannot establish operational detection performance.</p>
        </fieldset></div>

        <aside className="run-lock-panel">
          <div className="real-section-head"><div><p className="eyebrow">REVIEW & LAUNCH</p><h3>Your experiment</h3></div><span>{lock ? lock.readyToDispatch ? "SETTINGS READY" : "SETUP NEEDED" : "NOT CHECKED"}</span></div>
          <p className="selected-source"><strong>{spec.dataset.ref} · v{spec.dataset.version || "?"}</strong><span>{spec.dataset.file || "Choose an exact file"}</span><span>Up to {spec.dataContract.maxRows.toLocaleString()} rows · seed {spec.engine.seed}</span></p>
          <div className="split-ledger">
            <div><span>CALIBRATION</span><strong>20%</strong><small>fit transforms only</small></div>
            <div><span>EVALUATION</span><strong>60%</strong><small>labels sealed until freeze</small></div>
            <div className="sealed"><span>HELD-OUT</span><strong>20%</strong><small>not sent to engine</small></div>
          </div>
          <dl className="lock-rules">
            <div><dt>ENGINE</dt><dd>0.4.7.02 · Torch reservoir/HDC</dd></div>
            <div><dt>FEATURE SPACE</dt><dd>64D seeded projection or zero-pad</dd></div>
            <div><dt>NORMALIZATION</dt><dd>Calibration rows only</dd></div>
            <div><dt>PROOF EFFECT</dt><dd>Zero accepting gates</dd></div>
          </dl>
          <label className="operator-token"><span>Operator access token</span><input type="password" autoComplete="off" value={operatorToken} onChange={(event) => setOperatorToken(event.target.value)} placeholder="Enter to launch or retrieve a run" /></label>
          <p className="token-note">Kept in memory for this page only. Readiness checks your settings. Compute startup and dataset access are verified after launch.</p>
          <button className="prepare-button" onClick={prepareLock} disabled={preparing || dispatching}>{preparing ? "Checking settings…" : "Check readiness & lock settings"}</button>

          {lock && <div className="lock-result" aria-live="polite"><span>SETTINGS LOCKED · {(lock.executionBackend || "unattached").toUpperCase()}</span><code>{lock.digest}</code><button className="outline-button" onClick={() => downloadJson(lock, `eidos-run-lock-${lock.digest.slice(0, 12)}.json`)}>Download run settings</button><div className="preflight-issues">{lock.issues.filter((issue) => issue.severity === "blocker").map((issue) => <p className="blocker" key={issue.code}><b>Setup needed</b>{issue.message}</p>)}</div><details className="readiness-details"><summary>Data checks & technical details</summary><div className="preflight-issues">{lock.issues.filter((issue) => issue.severity !== "blocker").map((issue) => <p className={issue.severity} key={issue.code}>{issue.message}</p>)}</div></details><button className="dispatch-button" disabled={!lock.readyToDispatch || !operatorToken.trim() || dispatching || Boolean(dispatch)} onClick={dispatchRun}>{dispatching ? "Starting engine… this may take a few minutes" : dispatch ? "Run receipt available below" : !lock.readyToDispatch ? "Resolve setup before launching" : !operatorToken.trim() ? "Enter operator token to launch" : "Launch full-engine experiment"}</button></div>}
          {dispatch ? <><ExperimentRunMonitor key={dispatch.jobId} dispatch={dispatch} operatorToken={operatorToken} onError={setError} onTerminal={onTerminal} onStatus={onStatus} /><button className="outline-button" onClick={stopMonitoring}>{finished ? "Prepare another experiment" : "Stop monitoring this run"}</button>{!finished && <p className="token-note">Stopping monitoring does not cancel the engine. Save the receipt above to reconnect later.</p>}</> : null}
          <details className="resume-run"><summary>Resume an existing run</summary><label><span>Job ID</span><input value={resumeId} onChange={(event) => setResumeId(event.target.value)} placeholder="rd-…" /></label><button className="outline-button" onClick={resumeJob}>Check this run</button><p>Your last receipt is remembered in this tab. After a reload, enter your operator token again.</p></details>
        </aside>
      </div>

      <div className="honesty-bar"><span>WHAT THIS CAN ESTABLISH</span><p>Engineering behavior on pinned real observations, reproducibility receipts, causal preprocessing, and post-freeze evaluation metrics.</p><span>WHAT IT CANNOT ESTABLISH</span><p>Grand Proof acceptance, production safety, or held-out generalization. Those claims remain blocked.</p></div>
    </section>
  );
}
