"use client";

import { useState } from "react";
import { cloneDefaultExperiment } from "@/lib/experiments/shared";
import type { DatasetSearchResult, ExperimentSpec, LockedExperiment, RunnerDispatch } from "@/lib/experiments/types";

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

  function revise(mutator: (current: ExperimentSpec) => ExperimentSpec) {
    setSpec((current) => mutator(current));
    setLock(null);
    setDispatch(null);
    setError(null);
  }

  async function searchDatasets() {
    setSearching(true);
    setError(null);
    setSearchNote(null);
    try {
      const response = await fetch(`/api/datasets?q=${encodeURIComponent(query)}`, { cache: "no-store" });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || "Dataset lookup failed.");
      setResults(body.results || []);
      setSearchNote(body.warning || (body.mode === "kaggle" ? "Live public Kaggle catalog results." : "Reviewed protocol fixtures."));
    } catch (searchError) {
      setError(searchError instanceof Error ? searchError.message : "Dataset lookup failed.");
    } finally {
      setSearching(false);
    }
  }

  function chooseDataset(result: DatasetSearchResult) {
    revise((current) => ({
      ...current,
      dataset: { ...current.dataset, ref: result.ref, ...(result.version ? { version: result.version } : {}) },
    }));
    setResults([]);
    setSearchNote(result.version ? `Selected ${result.ref} at catalog version ${result.version}. Confirm the exact file below.` : `Selected ${result.ref}. Enter an explicit version before locking.`);
  }

  async function prepareLock() {
    setPreparing(true);
    setError(null);
    setDispatch(null);
    try {
      const response = await fetch("/api/experiments/preflight", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(spec),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || "Experiment preflight failed.");
      setLock(body as LockedExperiment);
    } catch (lockError) {
      setLock(null);
      setError(lockError instanceof Error ? lockError.message : "Experiment preflight failed.");
    } finally {
      setPreparing(false);
    }
  }

  async function dispatchRun() {
    if (!lock) return;
    setDispatching(true);
    setError(null);
    try {
      const response = await fetch("/api/experiments", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${operatorToken}` },
        body: JSON.stringify({ spec: lock.spec, lockDigest: lock.digest }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || body.error || "Experiment dispatch failed.");
      setDispatch(body as RunnerDispatch);
    } catch (dispatchError) {
      setError(dispatchError instanceof Error ? dispatchError.message : "Experiment dispatch failed.");
    } finally {
      setDispatching(false);
    }
  }

  return (
    <section className="real-data-page" aria-labelledby="real-data-title">
      <div className="real-data-intro">
        <div>
          <p className="eyebrow">SENTINEL LAB v0.2 · FULL ENGINE CONTROL PLANE</p>
          <h2 id="real-data-title">Real data,<br />sealed claims.</h2>
        </div>
        <div className="real-data-thesis">
          <p>This path prepares a pinned Kaggle file for the full Torch reservoir/HDC engine. Labels never enter the engine stream, evaluation waits until predictions are frozen, and the held-out partition stays untouched.</p>
          <div className="class-badges"><span>REAL_DATA_ENGINEERING</span><span>GATES REMAIN LOCKED</span></div>
        </div>
      </div>

      {error && <div className="real-error" role="alert"><span>RUN HALTED</span>{error}</div>}

      <div className="real-pipeline" aria-label="Real-data experiment stages">
        <div><b>01</b><span>Kaggle version + exact file</span><small>immutable source</small></div>
        <div><b>02</b><span>Label vault + causal order</span><small>no engine access</small></div>
        <div><b>03</b><span>Calibration-only transform</span><small>holdout excluded</small></div>
        <div><b>04</b><span>Full Eidos 0.4.7.02</span><small>external compute</small></div>
      </div>

      <div className="real-workspace">
        <div className="real-config">
          <div className="real-section-head"><div><p className="eyebrow">DATASET LOOKUP</p><h3>Pin the source</h3></div><span>KAGGLE</span></div>
          <div className="dataset-search"><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search public Kaggle datasets" aria-label="Search Kaggle datasets" /><button onClick={searchDatasets} disabled={searching}>{searching ? "Looking…" : "Lookup"}</button></div>
          {searchNote && <p className="search-note">{searchNote}</p>}
          {results.length > 0 && <div className="dataset-results">{results.map((result) => <button key={`${result.source}-${result.ref}`} onClick={() => chooseDataset(result)}><span>{result.source.toUpperCase()} · {result.version ? `V${result.version}` : "VERSION REQUIRED"}</span><strong>{result.title}</strong><small>{result.ref} · {bytes(result.totalBytes)}</small></button>)}</div>}

          <div className="real-form-grid">
            <label className="wide"><span>DATASET HANDLE</span><input value={spec.dataset.ref} onChange={(event) => revise((current) => ({ ...current, dataset: { ...current.dataset, ref: event.target.value } }))} /></label>
            <label><span>PINNED VERSION</span><input type="number" min={1} value={spec.dataset.version} onChange={(event) => revise((current) => ({ ...current, dataset: { ...current.dataset, version: Number(event.target.value) } }))} /></label>
            <label><span>ENGINEERING SEED</span><select value={spec.engine.seed} onChange={(event) => revise((current) => ({ ...current, engine: { ...current.engine, seed: Number(event.target.value) as 0 | 1 } }))}><option value={0}>0</option><option value={1}>1</option></select></label>
            <label className="wide"><span>EXACT FILE PATH</span><input value={spec.dataset.file} onChange={(event) => revise((current) => ({ ...current, dataset: { ...current.dataset, file: event.target.value } }))} /></label>
            <label><span>LABEL COLUMN</span><input value={spec.dataContract.labelColumn} onChange={(event) => revise((current) => ({ ...current, dataContract: { ...current.dataContract, labelColumn: event.target.value } }))} /></label>
            <label><span>BENIGN LABELS</span><input value={spec.dataContract.negativeLabels.join(", ")} onChange={(event) => revise((current) => ({ ...current, dataContract: { ...current.dataContract, negativeLabels: csv(event.target.value) } }))} /></label>
            <label><span>ROW ORDER</span><select value={spec.dataContract.orderMode} onChange={(event) => revise((current) => ({ ...current, dataContract: { ...current.dataContract, orderMode: event.target.value as "source" | "column", ...(event.target.value === "source" ? { orderColumn: undefined } : {}) } }))}><option value="source">Source order</option><option value="column">Sort by column</option></select></label>
            <label><span>ORDER COLUMN</span><input disabled={spec.dataContract.orderMode === "source"} value={spec.dataContract.orderColumn || ""} placeholder={spec.dataContract.orderMode === "source" ? "Not used" : "Timestamp"} onChange={(event) => revise((current) => ({ ...current, dataContract: { ...current.dataContract, orderColumn: event.target.value } }))} /></label>
            <label><span>MAX ROWS</span><input type="number" min={1000} max={2000000} step={1000} value={spec.dataContract.maxRows} onChange={(event) => revise((current) => ({ ...current, dataContract: { ...current.dataContract, maxRows: Number(event.target.value) } }))} /></label>
            <label className="wide"><span>EXCLUDED COLUMNS</span><input value={spec.dataContract.excludedColumns.join(", ")} onChange={(event) => revise((current) => ({ ...current, dataContract: { ...current.dataContract, excludedColumns: csv(event.target.value) } }))} /></label>
            <label className="wide"><span>EXPLICIT FEATURE COLUMNS <i>optional</i></span><input value={spec.dataContract.featureColumns.join(", ")} placeholder="Blank = numeric columns with label-like fields rejected" onChange={(event) => revise((current) => ({ ...current, dataContract: { ...current.dataContract, featureColumns: csv(event.target.value) } }))} /></label>
            <label className="wide"><span>EXPECTED FILE SHA-256 <i>optional first-run pin</i></span><input value={spec.dataset.expectedSha256 || ""} placeholder="Runner records the digest when blank" onChange={(event) => revise((current) => ({ ...current, dataset: { ...current.dataset, expectedSha256: event.target.value || undefined } }))} /></label>
          </div>
        </div>

        <aside className="run-lock-panel">
          <div className="real-section-head"><div><p className="eyebrow">PREREGISTRATION</p><h3>Run lock</h3></div><span>{lock ? "PREPARED" : "OPEN"}</span></div>
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
          <label className="operator-token"><span>OPERATOR CREDENTIAL</span><input type="password" autoComplete="off" value={operatorToken} onChange={(event) => setOperatorToken(event.target.value)} placeholder="Required only when dispatch is enabled" /></label>
          <button className="prepare-button" onClick={prepareLock} disabled={preparing}>{preparing ? "Computing SHA-256 lock…" : "Prepare immutable run lock"}</button>

          {lock && <div className="lock-result"><span>SPEC SHA-256</span><code>{lock.digest}</code><div className="preflight-issues">{lock.issues.map((issue) => <p className={issue.severity} key={issue.code}><b>{issue.code} · {issue.severity}</b>{issue.message}</p>)}</div><button className="dispatch-button" disabled={!lock.readyToDispatch || dispatching} onClick={dispatchRun}>{dispatching ? "Dispatching…" : lock.runnerConfigured ? "Dispatch to full engine" : "Runner attachment required"}</button></div>}
          {dispatch && <div className="dispatch-receipt"><span>JOB ACCEPTED</span><strong>{dispatch.jobId}</strong><small>{dispatch.status} · {dispatch.evidenceClass}</small></div>}
        </aside>
      </div>

      <div className="honesty-bar"><span>WHAT THIS CAN ESTABLISH</span><p>Engineering behavior on pinned real observations, reproducibility receipts, causal preprocessing, and post-freeze evaluation metrics.</p><span>WHAT IT CANNOT ESTABLISH</span><p>Grand Proof acceptance, production safety, or held-out generalization. Those claims remain blocked.</p></div>
    </section>
  );
}
