"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { describeFailure, evidenceForFinding, formatMeasured, kaggleDatasetUrl, prepareRunOptions, requestFingerprint, suggestedMapping, summarizeRun, type Experience, type LabDocument } from "@/lib/guided/experience";

const stages = ["Add data", "Check its meaning", "Run experiment", "Understand results"];
const terminal = (job: LabDocument | null) => !job || ["completed", "failed", "cancelled", "expired"].includes(job.status);
const initialOptions = { mode: "unusual", target: "", horizonSeconds: 60, windowSeconds: 60 };

/** An additive entry point over the same owner-scoped API and actual runner as the detailed workspace. */
export function EasyExperiment() {
  const [experience, setExperience] = useState<Experience>("easy");
  const [step, setStep] = useState(0);
  const [user, setUser] = useState<LabDocument | null>(null);
  const [key, setKey] = useState("");
  const [dataset, setDataset] = useState<LabDocument | null>(null);
  const [mapping, setMapping] = useState<LabDocument>({});
  const [options, setOptions] = useState<LabDocument>(initialOptions);
  const [run, setRun] = useState<LabDocument | null>(null);
  const [job, setJob] = useState<LabDocument | null>(null);
  const [paused, setPaused] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [url, setUrl] = useState("");
  const [query, setQuery] = useState("");
  const [catalog, setCatalog] = useState<LabDocument | null>(null);
  const [history, setHistory] = useState<LabDocument>({ datasets: [], runs: [], jobs: [] });
  const [filter, setFilter] = useState("");
  const [showAll, setShowAll] = useState(false);
  const credential = useRef("");
  const epoch = useRef(0);
  const busyRef = useRef(false);
  const requests = useRef(new Set<AbortController>());
  const intent = useRef<{ fingerprint: string; key: string } | null>(null);
  const title = useRef<HTMLHeadingElement>(null);
  const active = !terminal(job);

  const api = useCallback(async (path: string, method = "GET", input?: unknown, idempotency?: string): Promise<any> => {
    const controller = new AbortController(); requests.current.add(controller);
    const timeout = setTimeout(() => controller.abort(), 90000);
    const generation = epoch.current;
    try {
      const response = await fetch(`/api/lab/v1/${path}`, { method, signal: controller.signal, cache: "no-store",
        headers: { ...(input !== undefined ? { "Content-Type": "application/json" } : {}), ...(credential.current ? { Authorization: `Bearer ${credential.current}` } : {}), ...(idempotency ? { "Idempotency-Key": idempotency } : {}) },
        ...(input !== undefined ? { body: JSON.stringify(input) } : {}) });
      if (generation !== epoch.current) throw new Error("Workspace changed; the previous response was discarded.");
      const body = await response.json().catch(() => { throw new Error("The server response was unreadable. Check saved jobs before retrying."); });
      if (generation !== epoch.current) throw new Error("Workspace changed; the previous response was discarded.");
      if (!response.ok) throw new Error(`${response.status}: ${body.error || "Request failed"}${body.diagnosticId ? ` · diagnostic ${body.diagnosticId}` : ""}`);
      return body;
    } catch (failure) {
      if (controller.signal.aborted) throw new Error("Request interrupted. Its outcome may be unknown; check saved jobs before trying a new run.");
      throw failure;
    } finally { clearTimeout(timeout); requests.current.delete(controller); }
  }, []);

  const refresh = useCallback(async () => {
    const generation = epoch.current;
    const [datasets, runs, jobs] = await Promise.all([api("datasets"), api("runs"), api("jobs")]);
    if (generation === epoch.current) setHistory({ datasets, runs, jobs });
  }, [api]);
  const loadDataset = useCallback((item: LabDocument) => {
    setDataset(item); setMapping(suggestedMapping(item)); setRun(null); setFilter(""); setShowAll(false);
    setOptions({ ...initialOptions, target: item.confirmed?.features?.[0] || suggestedMapping(item).features?.[0] || "" });
    setStep(item.confirmed ? 2 : 1);
  }, []);
  async function perform(operation: () => Promise<void>) {
    if (busyRef.current) return;
    const generation = epoch.current;
    busyRef.current = true; setBusy(true); setError(""); setNotice("");
    try { await operation(); }
    catch (failure) { if (generation === epoch.current) setError(describeFailure(failure instanceof Error ? failure.message : "The operation failed.")); }
    finally { if (generation === epoch.current) { busyRef.current = false; setBusy(false); } }
  }
  async function launch(path: string, input: unknown) {
    const generation = epoch.current;
    const fingerprint = requestFingerprint(path, input);
    if (intent.current?.fingerprint !== fingerprint) intent.current = { fingerprint, key: crypto.randomUUID().replaceAll("-", "_") };
    const next = await api(path, "POST", input, intent.current.key);
    if (generation === epoch.current) { setJob(next); setPaused(false); }
  }
  useEffect(() => {
    let mounted = true;
    void api("session").then(session => { if (mounted) setUser(session.user); }).catch(() => undefined);
    return () => { mounted = false; requests.current.forEach(controller => controller.abort()); };
  }, [api]);
  useEffect(() => { if (user) void refresh().catch(failure => setError(String(failure))); }, [user, refresh]);
  useEffect(() => { title.current?.focus(); }, [step]);
  useEffect(() => {
    if (!job || terminal(job) || paused || !user) return;
    let stopped = false, failures = 0;
    const generation = epoch.current;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const next = await api(`jobs/${encodeURIComponent(job.id)}`);
        if (stopped || generation !== epoch.current) return;
        if (next.status === "completed") {
          // Only a server-persisted completed result advances the journey.
          if (next.result?.runId) {
            const result = await api(`runs/${encodeURIComponent(next.result.runId)}`);
            const input = await api(`datasets/${encodeURIComponent(result.datasetId)}`);
            if (stopped || generation !== epoch.current) return;
            summarizeRun(result, input);
            setDataset(input); setMapping(suggestedMapping(input)); setRun(result); setStep(3); setShowAll(false);
          } else if (next.result?.datasetId) {
            const input = await api(`datasets/${encodeURIComponent(next.result.datasetId)}`);
            if (stopped || generation !== epoch.current) return;
            loadDataset(input);
          } else throw new Error("The completed job has no saved dataset or run reference. Inspect it in the detailed workspace.");
          setJob(next); intent.current = null; void refresh().catch(() => undefined); return;
        }
        if (terminal(next)) { setJob(next); intent.current = null; setError(describeFailure(next.error || `Job ${next.status}. No successful analysis is implied.`)); void refresh().catch(() => undefined); return; }
        failures = 0; setJob(next);
      } catch (failure) {
        if (stopped || generation !== epoch.current) return;
        if (++failures >= 3) { setPaused(true); setError(`Status monitoring paused. Your job may still exist; use Resume status or Saved work. ${String(failure)}`); return; }
      }
      if (!stopped) timer = setTimeout(poll, 3000);
    };
    timer = setTimeout(poll, 500);
    return () => { stopped = true; clearTimeout(timer); };
  }, [job?.id, job?.status, paused, user, api, loadDataset, refresh]);

  async function signIn() {
    credential.current = key.trim();
    try {
      const session = await api("session", "POST");
      if (session.user.credential !== "member") credential.current = "";
      setUser(session.user); setKey("");
    } catch (failure) { credential.current = ""; throw failure; }
  }
  async function signOut() {
    // Clear private state even when the network cannot confirm cookie removal.
    epoch.current++; requests.current.forEach(controller => controller.abort());
    credential.current = ""; intent.current = null; busyRef.current = false;
    setUser(null); setKey(""); setDataset(null); setMapping({}); setRun(null); setJob(null);
    setHistory({ datasets: [], runs: [], jobs: [] }); setBusy(false); setPaused(false); setStep(0); setError(""); setNotice("");
    try { await api("session", "DELETE"); }
    catch { setError("Local data was cleared, but server sign-out was not confirmed. Close this browser session before sharing the device."); }
  }
  async function upload(file: File) {
    if (!user || active) throw new Error("Sign in and finish or cancel the current job before adding another file.");
    if (file.size < 1 || file.size > 2000000) throw new Error("Choose a nonempty file no larger than 2,000,000 bytes. Nothing is silently sampled.");
    const generation = epoch.current;
    const bytes = new Uint8Array(await file.arrayBuffer());
    const sha256 = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))).map(value => value.toString(16).padStart(2, "0")).join("");
    if (generation !== epoch.current) return;
    let binary = ""; for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
    await launch("datasets", { filename: file.name, content: btoa(binary), sha256 });
  }
  async function openRun(id: string) {
    const generation = epoch.current;
    const result = await api(`runs/${encodeURIComponent(id)}`);
    const input = await api(`datasets/${encodeURIComponent(result.datasetId)}`);
    if (generation !== epoch.current) return;
    summarizeRun(result, input); setDataset(input); setRun(result); setStep(3); setShowAll(false); setFilter("");
  }
  async function download(path: string, filename: string) {
    const generation = epoch.current;
    const controller = new AbortController(); requests.current.add(controller);
    try {
      const response = await fetch(`/api/lab/v1/${path}`, { signal: controller.signal, cache: "no-store", headers: credential.current ? { Authorization: `Bearer ${credential.current}` } : {} });
      if (!response.ok) throw new Error("The download failed. Check your access before retrying.");
      const blob = await response.blob(); if (generation !== epoch.current) return;
      const href = URL.createObjectURL(blob), link = document.createElement("a"); link.href = href; link.download = filename; link.click();
      setTimeout(() => URL.revokeObjectURL(href), 1000);
    } finally { requests.current.delete(controller); }
  }
  async function searchKaggle() {
    const response = await fetch(`/api/datasets?q=${encodeURIComponent(query.trim())}`, { cache: "no-store" });
    const result = await response.json(); if (!response.ok) throw new Error(result.error || "Catalog lookup failed.");
    setCatalog(result);
  }
  const changeMapping = (field: string, value: unknown) => setMapping(current => ({ ...current, [field]: value }));
  const locked = busy || active;
  const temporal = dataset?.confirmed?.mode === "temporal" && dataset?.kind !== "documents";
  let summary: ReturnType<typeof summarizeRun> | null = null, resultError = "";
  if (run && dataset) { try { summary = summarizeRun(run, dataset); } catch (failure) { resultError = String(failure); } }
  const matches = summary?.checked.filter(({ finding }) => `${finding.what || ""} ${finding.entity || ""}`.toLowerCase().includes(filter.toLowerCase())) || [];
  const findings = showAll ? matches : matches.slice(0, 20);

  return <div className="easy-lab">
    <a className="el-skip" href="#experiment-main">Skip to experiment</a>
    <header className="el-header"><Link href="/" className="el-brand">Eidos <strong>Sentinel Lab</strong></Link><nav aria-label="Lab workspaces"><a href="#saved-work">Saved work</a><Link href="/advanced">Detailed workspace</Link><Link href="/research">Research console</Link></nav></header>
    <main id="experiment-main" className="el-main">
      <section className="el-intro"><div><p className="el-eyebrow">Real inputs. Traceable findings.</p><h1>From your data<br />to a clearer answer.</h1><p>Find unusual records, inspect the evidence, and evaluate forecasts without configuring a research console.</p></div><div className="el-experience" role="group" aria-label="Experiment setup"><button aria-pressed={experience === "easy"} onClick={() => setExperience("easy")}>Easy setup</button><button aria-pressed={experience === "advanced"} onClick={() => setExperience("advanced")}>Advanced settings</button><small>Same data and engine. Your selections stay intact.</small></div></section>
      <p className="el-disclosure"><strong>Experimental analysis, not an operational guarantee.</strong> Findings are signals to investigate. Forecast performance and missing evidence are reported, not assumed.</p>
      {!user ? <form className="el-panel el-signin" onSubmit={event => { event.preventDefault(); void perform(signIn); }}><div><h2>Open your private workspace</h2><p>Use an approved Lab key or an authorized Eidos member agent key. No credential is saved in browser storage.</p></div><label>Access key<input type="password" autoComplete="off" value={key} onChange={event => setKey(event.target.value)} required /></label><button className="el-primary" disabled={busy || !key.trim()}>Sign in</button></form>
        : <div className="el-session"><span>Private workspace · saved input/result access expires after 30 days; archives are not deleted</span><button disabled={busy} onClick={() => void signOut()}>Sign out</button></div>}
      {error && <div role="alert" className="el-error"><strong>Action needed</strong><p>{error}</p><button onClick={() => setError("")}>Dismiss</button></div>}
      {notice && <p role="status" className="el-notice">{notice}</p>}
      {job && <section className="el-panel el-job" aria-live="polite"><div><p className="el-eyebrow">{job.operation || "Saved job"}</p><h2>{String(job.status).replaceAll("_", " ")}</h2><p>{job.error ? describeFailure(job.error) : "This is the recorded worker state, not simulated progress."}</p><code>{job.id}</code></div><div className="el-actions">{!terminal(job) && <><button disabled={busy} onClick={() => void perform(async () => { setJob(await api(`jobs/${encodeURIComponent(job.id)}/resume`, "POST")); setPaused(false); })}>{paused ? "Resume status" : "Resume queued job"}</button><button disabled={busy} onClick={() => void perform(async () => { const cancelled = await api(`jobs/${encodeURIComponent(job.id)}/cancel`, "POST"); setJob(cancelled); intent.current = null; })}>Cancel job</button></>}{terminal(job) && <button onClick={() => setJob(null)}>Hide job status</button>}</div></section>}
      <ol className="el-steps" aria-label="Experiment progress">{stages.map((stage, i) => <li key={stage}><button aria-current={step === i ? "step" : undefined} disabled={locked || (i > 0 && !dataset) || (i === 2 && !dataset?.confirmed) || (i === 3 && !run)} onClick={() => setStep(i)}><span>{i + 1}</span>{stage}</button></li>)}</ol>
      <section className="el-panel"><div className="el-section-title"><div><p className="el-eyebrow">Step {step + 1} of 4</p><h2 ref={title} tabIndex={-1}>{stages[step]}</h2></div>{dataset && <p>{dataset.filename} · {formatMeasured(dataset.rowCount)} records</p>}</div>
        {step === 0 && <div className="el-source-grid"><div className="el-upload" onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); if (locked) return; const file = event.dataTransfer.files[0]; if (file) void perform(() => upload(file)); }}><span aria-hidden="true" className="el-upload-mark">↑</span><h3>Upload your data</h3><p>Drop a file here or choose one below.</p><label>Data file<input type="file" disabled={!user || locked} accept=".csv,.xlsx,.parquet,.json,.jsonl,.ndjson,.log,.pdf,.html,.htm,.txt,.md" onChange={event => { const file = event.target.files?.[0]; event.target.value = ""; if (file) void perform(() => upload(file)); }} /></label><p className="el-small">CSV, XLSX, Parquet, JSON/JSONL, logs, text PDF, HTML or text. Up to 2 MB, 5,000 rows, 64 columns and 256 passages. Unsupported or oversized data fails explicitly.</p></div><div className="el-stack"><form onSubmit={event => { event.preventDefault(); void perform(() => launch("imports", { url: url.trim() })); }}><h3>Use a public data link</h3><label>Public HTTPS URL<input type="url" required value={url} onChange={event => setUrl(event.target.value)} placeholder="https://example.org/measurements.csv" /></label><button disabled={!user || locked}>Load public source</button><p className="el-small">A downloadable data file is different from a web page. Pages are analyzed as text, not guessed into measurements.</p></form><form onSubmit={event => { event.preventDefault(); void perform(searchKaggle); }}><h3>Find a dataset on Kaggle</h3><label>Search datasets<input value={query} onChange={event => setQuery(event.target.value)} required maxLength={120} placeholder="For example: machine sensor readings" /></label><button disabled={busy || !query.trim()}>Search catalog</button><p className="el-small">For this release: open a dataset, download a supported file within the size limit, then upload it here. Direct authenticated Kaggle import into Easy setup is not enabled.</p></form></div>{catalog && <div className="el-catalog"><h3>Kaggle catalog results</h3><p>{catalog.warning || (catalog.mode === "kaggle" ? "Live public catalog results." : "Reviewed catalog fixtures; not a live search result.")}</p>{!(catalog.results || []).length && <p>No datasets were returned. Try a different search.</p>}{(catalog.results || []).map((item: LabDocument) => { const href = kaggleDatasetUrl(item.ref); return <article key={item.ref}><h4>{item.title}</h4><p>{item.subtitle}</p><p className="el-small">{item.ref} · version {item.version ?? "not listed"} · {item.source === "kaggle" ? "Kaggle catalog" : "Curated fixture"}</p>{href && <a href={href} target="_blank" rel="noreferrer">Open dataset on Kaggle ↗</a>}</article>; })}<Link href="/research">Use the existing versioned Kaggle research importer ↗</Link></div>}</div>}
        {step === 1 && dataset && <><p>Check the interpretation before the engine sees your data. Nothing below treats file order as time.</p>{(dataset.warnings || []).map((warning: string, i: number) => <p className="el-warning" key={i}>{warning}</p>)}
          {dataset.kind === "documents" ? <><h3>Text and document analysis</h3><p>This source supports passage-backed patterns and retrieval, not a numerical forecast.</p><pre>{JSON.stringify((dataset.passages || []).slice(0, 2), null, 2)}</pre></> : <>
            <div className="el-form-grid"><label>What does one row represent?<input value={mapping.meaning || ""} onChange={event => changeMapping("meaning", event.target.value)} placeholder="One machine reading at a particular time" /></label><FieldSelect label="Time column" columns={dataset.columns || []} value={mapping.timestamp} empty="No time column — analyze as an unordered table" onChange={value => changeMapping("timestamp", value)} /><label>Time zone when offsets are absent<input value={mapping.timezone || ""} onChange={event => changeMapping("timezone", event.target.value)} placeholder="For example: America/New_York" /></label><FieldSelect label="Separate machines, services or people by" columns={dataset.columns || []} value={mapping.entity} empty="One entity in this dataset" onChange={value => changeMapping("entity", value)} /></div>
            <fieldset><legend>Measurements to analyze</legend><p className="el-small">Labels, identifiers, constants and nonnumeric fields are not automatically used as measurements. Up to 16 features.</p><div className="el-measurements">{(dataset.columns || []).map((column: LabDocument) => { const excluded = column.labelCandidate || column.idCandidate || column.constant || column.type !== "number"; const selected = (mapping.features || []).includes(column.name); return <label key={column.name} className="el-check"><input type="checkbox" disabled={excluded || (!selected && (mapping.features || []).length >= 16)} checked={selected} onChange={event => setMapping(current => ({ ...current, features: event.target.checked ? [...current.features, column.name] : current.features.filter((name: string) => name !== column.name), units: { ...current.units, [column.name]: current.units?.[column.name] || "unknown" } }))} /><span>{column.name}<small>{column.labelCandidate ? "Evaluation label — excluded" : excluded ? "Not a varying numeric measurement" : `${column.missing ?? "Unknown"} missing values`}</small></span></label>; })}</div></fieldset>
            {experience === "advanced" && <div className="el-advanced"><h3>Data preparation rules</h3><div className="el-form-grid"><label>Reference description<input value={mapping.reference || ""} onChange={event => changeMapping("reference", event.target.value)} /></label><FieldSelect label="Session column" columns={dataset.columns || []} value={mapping.session} empty="One session per entity" onChange={value => changeMapping("session", value)} /><label>Missing measurements<select value={mapping.missing || "exclude"} onChange={event => changeMapping("missing", event.target.value)}><option value="exclude">Exclude incomplete measurement rows</option><option value="prefix_median">Use calibration-prefix medians (temporal only)</option></select></label><label>Out-of-order timestamps<select value={mapping.ordering || "sort"} onChange={event => changeMapping("ordering", event.target.value)}><option value="sort">Sort within the confirmed entity/session</option><option value="reject">Reject and repair at the source</option></select></label>{(mapping.features || []).map((feature: string) => <label key={feature}>{feature} units<input value={mapping.units?.[feature] || "unknown"} onChange={event => setMapping(current => ({ ...current, units: { ...current.units, [feature]: event.target.value } }))} /></label>)}</div><p>These settings are passed to the existing worker. Arbitrary code and custom threshold expressions are not accepted by this interface.</p></div>}
            <details><summary>Preview original records and schema</summary><pre>{JSON.stringify({ columns: dataset.columns, records: (dataset.records || []).slice(0, 5) }, null, 2)}</pre></details>
          </>}
          <p className="el-small">Current rules: {mapping.missing === "prefix_median" ? "calibration-prefix imputation" : "exclude incomplete measurement rows"}; {mapping.ordering === "reject" ? "reject out-of-order timestamps" : "sort within entity/session after time is confirmed"}. Change these in Advanced settings.</p>
          <button className="el-primary" disabled={!user || locked} onClick={() => void perform(() => launch(`datasets/${encodeURIComponent(dataset.id)}/confirm`, mapping))}>Confirm this interpretation</button>
        </>}
        {step === 2 && dataset && <><p>Choose the question. Both setup modes use the same confirmed input and actual analysis service.</p><fieldset className="el-goals"><legend>What should this experiment do?</legend>{[["unusual", "Find unusual behavior", "Locate records that differ from the confirmed reference."], ["forecast", "Forecast a measurement", "Evaluate a named measurement over a declared time horizon."], ["patterns", "Explore patterns", "Inspect relationships or source-linked text patterns."]].map(([mode, heading, description]) => <label key={mode}><input type="radio" name="analysis-goal" checked={options.mode === mode} disabled={mode === "forecast" && !temporal} onChange={() => setOptions(current => ({ ...current, mode }))} /><span><strong>{heading}</strong><small>{description}{mode === "forecast" && !temporal ? " Requires confirmed numeric chronology." : ""}</small></span></label>)}</fieldset>
          {temporal && <div className="el-form-grid"><label>Measurement to forecast<select value={options.target} onChange={event => setOptions(current => ({ ...current, target: event.target.value }))}>{(dataset.confirmed.features || []).map((feature: string) => <option key={feature}>{feature}</option>)}</select></label><label>How far ahead? (seconds)<input type="number" min={1} max={86400} step={1} value={options.horizonSeconds} onChange={event => setOptions(current => ({ ...current, horizonSeconds: event.target.value === "" ? "" : Number(event.target.value) }))} /></label>{experience === "advanced" && <label>Maximum late-match window (seconds)<input type="number" min={1} max={86400} step={1} value={options.windowSeconds} onChange={event => setOptions(current => ({ ...current, windowSeconds: event.target.value === "" ? "" : Number(event.target.value) }))} /></label>}</div>}
          <div className="el-run-contract"><h3>Before you run</h3><p>{dataset.kind === "documents" ? "Text uses the pinned semantic encoder; it is not a temporal Eidos forecast." : temporal ? "Uses the causal Eidos/Torch profile with named measurements and disclosed baseline selections. At least 48 records per entity/session are required by the worker." : "Uses unordered-table baselines, not the temporal Eidos engine."}</p>{temporal && <p>Target: <strong>{options.target}</strong> · horizon {options.horizonSeconds || "not set"} seconds · late-match window {options.windowSeconds || "not set"} seconds.</p>}<p>One shared worker. The configured worker deadline is 900 seconds, not a completion estimate. No silent engine fallback or hidden sampling.</p></div>
          {experience === "advanced" && <details><summary>Exact confirmed data contract</summary><pre>{JSON.stringify(dataset.confirmed, null, 2)}</pre></details>}
          <button className="el-primary" disabled={!user || locked} onClick={() => void perform(() => launch(`datasets/${encodeURIComponent(dataset.id)}/analyze`, prepareRunOptions(dataset, options)))}>Run experiment</button>
        </>}
        {step === 3 && resultError && <p role="alert" className="el-error">{resultError}</p>}
        {step === 3 && summary && run && dataset && <><div className="el-result-lead"><p className="el-eyebrow">What the experiment found</p><h3>{summary.headline}</h3><p>{summary.limitation}</p><p><strong>Actual method:</strong> {run.method}</p><p>{run.sourceBadge}</p></div>{summary.unverified > 0 && <p role="alert" className="el-error">{summary.unverified} findings have incomplete or ambiguous source references. Treat them as unverified and inspect the exported result.</p>}<div className="el-summary-grid"><article><h3>How well did forecasting work?</h3><p>{summary.comparison}</p></article><article><h3>How certain are these findings?</h3><p>{summary.accuracy}</p><p>Score, prediction error and interval coverage measure different things.</p></article></div>
          <label>Find a measurement or entity<input type="search" value={filter} onChange={event => { setFilter(event.target.value); setShowAll(false); }} placeholder="Filter findings" /></label>
          {matches.length === 0 && <p>No findings match this view. That is not a statement that all activity is normal.</p>}
          {findings.map(({ finding }) => <FindingCard key={finding.id} finding={finding} dataset={dataset} advanced={experience === "advanced"} disabled={busy} onReview={review => void perform(async () => { await api(`runs/${encodeURIComponent(run.id)}/feedback`, "POST", { findingId: finding.id, review }); setNotice("Review saved. Original scores, labels and model training were not changed."); })} />)}
          {!showAll && matches.length > 20 && <button onClick={() => setShowAll(true)}>Show all {matches.length} findings</button>}
          <section className="el-forecasts"><h3>What could happen next?</h3>{(run.forecasts || []).length ? <><p>These are forecasts issued during this analysis. Historical replay is not proof of a real-time future prediction. Unresolved targets remain unverified, and error bands can be too narrow during change.</p><div className="el-table-wrap"><table><caption>Latest 10 issued forecasts — all issuances remain in the downloaded result</caption><thead><tr><th>Issued at / horizon</th><th>Target</th><th>Forecast / empirical band</th><th>Method / outcome</th></tr></thead><tbody>{run.forecasts.slice(-10).map((forecast: LabDocument) => <tr key={forecast.id}><td>{forecast.issuedAt || "Not recorded"}<br />{formatMeasured(forecast.horizonSeconds, "seconds")}</td><td>{forecast.target} ({forecast.unit || "unit unknown"})</td><td>{formatMeasured(forecast.prediction)}<br />[{formatMeasured(forecast.lower)}, {formatMeasured(forecast.upper)}]</td><td>{forecast.predictor || "Not identified"}<br />{forecast.resolution || "Unverified"}</td></tr>)}</tbody></table></div></> : <p>No forecast was produced for this run. A narrative prediction would be unsupported by these results.</p>}</section>
          <details open={experience === "advanced"}><summary>Technical metrics, limitations and reproducibility</summary>{(run.limitations || []).map((limitation: string, i: number) => <p key={i}>{limitation}</p>)}<p>Original input SHA-256: <code>{run.inputSha256}</code></p><pre>{JSON.stringify({ engine: run.engine, receipt: run.receipt, metrics: run.metrics }, null, 2)}</pre></details><div className="el-actions"><button onClick={() => void perform(() => download(`runs/${encodeURIComponent(run.id)}/download`, `${run.id}.json`))}>Download full result</button><button onClick={() => void perform(() => download(`datasets/${encodeURIComponent(dataset.id)}/source`, dataset.filename))}>Download original data</button><Link href="/advanced">Open detailed questions, charts and comparisons</Link></div>
        </>}
      </section>
      {user && <details id="saved-work" className="el-panel"><summary>Saved work and job recovery</summary><button disabled={busy} onClick={() => void perform(refresh)}>Refresh saved work</button><div className="el-history-grid"><div><h3>Datasets</h3>{!history.datasets.length && <p>No saved datasets.</p>}{history.datasets.map((item: LabDocument) => <button key={item.id} disabled={locked || item.expired} onClick={() => void perform(async () => loadDataset(await api(`datasets/${encodeURIComponent(item.id)}`)))}>{item.filename} · {item.expired ? "access expired" : item.confirmed ? "confirmed" : "needs confirmation"}</button>)}</div><div><h3>Results</h3>{!history.runs.length && <p>No completed results.</p>}{history.runs.map((item: LabDocument) => <button key={item.id} disabled={locked || item.expired} onClick={() => void perform(() => openRun(item.id))}>{item.method} · {item.id}</button>)}</div><div><h3>Jobs</h3>{!history.jobs.length && <p>No saved jobs.</p>}{history.jobs.map((item: LabDocument) => <button key={item.id} disabled={busy || active} onClick={() => void perform(async () => { const saved = await api(`jobs/${encodeURIComponent(item.id)}`); if (saved.status === "completed" && saved.result?.runId) await openRun(saved.result.runId); else if (saved.status === "completed" && saved.result?.datasetId) loadDataset(await api(`datasets/${encodeURIComponent(saved.result.datasetId)}`)); setJob(saved); setPaused(false); })}>{item.status} · {item.id}</button>)}</div></div></details>}
      <footer className="el-footer">Inspect the source. Compare the baseline. Preserve uncertainty. <Link href="/advanced">Telemetry and detailed workspace ↗</Link></footer>
    </main>
  </div>;
}

function FieldSelect({ label, columns, value, empty, onChange }: { label: string; columns: LabDocument[]; value?: string; empty: string; onChange: (value: string) => void }) {
  return <label>{label}<select value={value || ""} onChange={event => onChange(event.target.value)}><option value="">{empty}</option>{columns.map(column => <option key={column.name} value={column.name}>{column.name}</option>)}</select></label>;
}
function FindingCard({ finding, dataset, advanced, disabled, onReview }: { finding: LabDocument; dataset: LabDocument; advanced: boolean; disabled: boolean; onReview: (review: string) => void }) {
  const evidence = evidenceForFinding(finding, dataset);
  return <article className="el-finding"><p className="el-eyebrow">Finding · {finding.id}</p><h3>{finding.what || "Finding requires inspection"}</h3><dl><dt>What was observed?</dt><dd>{finding.basis || "No explanatory basis was supplied; inspect the source."}</dd><dt>Why might it matter?</dt><dd>{finding.whyItMatters || "Operational consequence has not been established."}</dd><dt>What could explain it?</dt><dd>This result does not establish a cause. Check source quality, expected operating changes and related events before assigning an explanation.</dd><dt>What should you do next?</dt><dd>{finding.nextAction || "Inspect the cited original records and compare them with the confirmed reference."}</dd><dt>Remaining uncertainty</dt><dd>{Array.isArray(finding.uncertainty) ? finding.uncertainty.join(" ") : "Not quantified in this finding."}</dd></dl><h4>Where in the data, and when?</h4>{evidence.unresolved > 0 && <p className="el-warning">{evidence.unresolved} source references could not be resolved.</p>}{evidence.references.length === 0 && <p className="el-warning">No verified original record is available for this finding.</p>}{evidence.references.slice(0, 12).map(reference => <details key={reference.id}><summary>{dataset.filename} · {reference.id} · {reference.when}</summary><p className="el-small">Source timestamp shown exactly as recorded. Confirmed time zone: {dataset.confirmed?.timezone || "use recorded offsets; no zone inferred"}.</p><pre>{JSON.stringify(reference.value, null, 2)}</pre><p className="el-small">Source SHA-256: <code>{dataset.sha256}</code></p>{advanced && <pre>{JSON.stringify(reference.member, null, 2)}</pre>}</details>)}{evidence.references.length > 12 && <p>Showing 12 of {evidence.references.length} source records. The full result retains all references.</p>}{advanced && <details><summary>Detector, grouping and ranking</summary><pre>{JSON.stringify({ detector: finding.detector, grouping: finding.grouping, ranking: finding.ranking }, null, 2)}</pre></details>}<label>Review this finding<select defaultValue="" disabled={disabled} onChange={event => { if (event.target.value) onReview(event.target.value); }}><option value="">Choose a review outcome</option><option value="expected">Expected behavior</option><option value="useful">Useful finding</option><option value="false_alarm">False alarm</option><option value="investigate">Needs investigation</option></select></label></article>;
}
