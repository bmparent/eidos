"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import type { Document } from "@/lib/guided/store";

const steps = ["Add data", "Confirm understanding", "Analyze", "Investigate"];
const format = (n: unknown) => typeof n === "number" ? new Intl.NumberFormat("en", { maximumFractionDigits: 3 }).format(n) : "Not measured";
const freshKey = () => crypto.randomUUID().replaceAll("-", "_");

export function GuidedLab() {
  const [user, setUser] = useState<Document | null>(null);
  const [key, setKey] = useState("");
  const memberKey = useRef("");
  const [step, setStep] = useState(0);
  const [section, setSection] = useState("analysis");
  const [dataset, setDataset] = useState<Document | null>(null);
  const [mapping, setMapping] = useState<Document>({ features: [], labels: [], timestamp: "", timezone: "", entity: "", session: "", units: {}, meaning: "", reference: "this entity's historical period", missing: "exclude", ordering: "sort" });
  const [run, setRun] = useState<Document | null>(null);
  const [job, setJob] = useState<Document | null>(null);
  const [datasets, setDatasets] = useState<Document[]>([]);
  const [runs, setRuns] = useState<Document[]>([]);
  const [jobs, setJobs] = useState<Document[]>([]);
  const [monitors, setMonitors] = useState<Document[]>([]);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [url, setUrl] = useState("");
  const [mode, setMode] = useState("unusual");
  const [target, setTarget] = useState("");
  const [horizon, setHorizon] = useState(60);
  const [windowSeconds, setWindowSeconds] = useState(60);
  const [entity, setEntity] = useState("all");
  const [filter, setFilter] = useState("");
  const [zoom, setZoom] = useState(100);
  const [record, setRecord] = useState<Document | null>(null);
  const [question, setQuestion] = useState("Why was this flagged?");
  const [answer, setAnswer] = useState<Document | null>(null);
  const [selectedFinding, setSelectedFinding] = useState("");
  const [comparison, setComparison] = useState<Document | null>(null);
  const [connector, setConnector] = useState<Document | null>(null);
  const [sourceName, setSourceName] = useState("");
  const [sourceTarget, setSourceTarget] = useState("service.latency");
  const [sourceUnit, setSourceUnit] = useState("ms");
  const [automatic, setAutomatic] = useState(false);
  const [synthetic, setSynthetic] = useState(false);
  const [paused, setPaused] = useState(false);
  const retry = useRef<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);

  const api = useCallback(async (path: string, method = "GET", value?: unknown, idempotency?: string) => {
    const response = await fetch(`/api/lab/v1/${path}`, { method, headers: {
      ...(value ? { "Content-Type": "application/json" } : {}), ...(idempotency ? { "Idempotency-Key": idempotency } : {}),
      ...(memberKey.current ? { Authorization: `Bearer ${memberKey.current}` } : {}),
    }, ...(value ? { body: JSON.stringify(value) } : {}) });
    const result = await response.json();
    if (!response.ok) throw new Error(`${result.error || "Request failed"}${result.diagnosticId ? ` · ${result.diagnosticId}` : ""}`);
    return result;
  }, []);
  const refresh = useCallback(async () => {
    const results = await Promise.allSettled([api("datasets"), api("runs"), api("jobs"), api("monitors")]);
    if (results[0].status === "fulfilled") setDatasets(results[0].value);
    if (results[1].status === "fulfilled") setRuns(results[1].value);
    if (results[2].status === "fulfilled") setJobs(results[2].value);
    if (results[3].status === "fulfilled") setMonitors(results[3].value);
  }, [api]);
  const act = async (operation: () => Promise<void>) => {
    setBusy(true); setError(""); setNotice("");
    try { await operation(); } catch (error) { setError(error instanceof Error ? error.message : "The operation failed."); }
    finally { setBusy(false); }
  };
  const selectDataset = useCallback(async (id: string) => {
    const item = await api(`datasets/${id}`); setDataset(item); setRun(null); setAnswer(null); setRecord(null);
    const suggested = item.columns.filter((c: Document) => c.type === "number" && !c.labelCandidate && !c.idCandidate && !c.constant).slice(0, 16).map((c: Document) => c.name);
    setMapping(item.confirmed || { features: suggested, labels: item.columns.filter((c: Document) => c.labelCandidate).map((c: Document) => c.name),
      timestamp: "", timezone: "", entity: "", session: "", units: Object.fromEntries(suggested.map((f: string) => [f, "unknown"])),
      meaning: "", reference: "this entity's historical period", missing: "exclude", ordering: "sort" });
    setTarget(item.confirmed?.features?.[0] || suggested[0] || "");
    setStep(item.confirmed ? 2 : 1); setSection("analysis");
  }, [api]);
  useEffect(() => { void api("session").then(result => setUser(result.user)).catch(() => undefined); }, [api]);
  useEffect(() => { if (user) void refresh(); }, [user, refresh]);
  useEffect(() => { heading.current?.focus(); }, [step]);
  useEffect(() => {
    if (!job || ["completed", "failed", "cancelled", "expired"].includes(job.status) || paused) return;
    let cancelled = false, timer: ReturnType<typeof setTimeout>;
    let failures = 0;
    const poll = async () => {
      try {
        const result = await api(`jobs/${job.id}`);
        if (cancelled) return;
        setJob(result); failures = 0;
        if (result.status === "completed") {
          await refresh(); retry.current = null;
          if (result.result?.datasetId) await selectDataset(result.result.datasetId);
          if (result.result?.runId) { setRun(await api(`runs/${result.result.runId}`)); setStep(3); setSection("analysis"); }
          if (result.result?.citations) setAnswer(result.result);
          return;
        }
        if (["failed", "cancelled", "expired"].includes(result.status)) { setError(result.error || result.status); return; }
      } catch (error) {
        if (++failures >= 3) { setPaused(true); setError(`Monitoring paused after repeated errors. The saved job is recoverable. ${String(error)}`); return; }
      }
      if (!cancelled) timer = setTimeout(poll, 3000);
    };
    timer = setTimeout(poll, 1000);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [job?.id, job?.status, paused, api, refresh, selectDataset]);

  async function signIn() {
    const response = await fetch("/api/lab/v1/session", { method: "POST", headers: { Authorization: `Bearer ${key}` } });
    const result = await response.json(); if (!response.ok) throw new Error(result.error);
    memberKey.current = result.user.credential === "member" ? key : "";
    setUser(result.user); setKey("");
  }
  async function launch(path: string, input: unknown) {
    retry.current ||= freshKey();
    const next = await api(path, "POST", input, retry.current);
    setJob(next); setPaused(false);
  }
  async function upload(file: File) {
    if (!user) throw new Error("Sign in before adding private data.");
    if (file.size > 2000000 || file.size === 0) throw new Error("Choose a file from 1 byte to 2 MB. Split larger files before uploading.");
    setUploadProgress(0); retry.current = freshKey();
    const bytes = new Uint8Array(await file.arrayBuffer());
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))).map(b => b.toString(16).padStart(2, "0")).join("");
    let binary = ""; for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
    const result: Document = await new Promise((resolve, reject) => {
      const request = new XMLHttpRequest(); request.open("POST", "/api/lab/v1/datasets");
      request.setRequestHeader("Content-Type", "application/json"); request.setRequestHeader("Idempotency-Key", retry.current!);
      if (memberKey.current) request.setRequestHeader("Authorization", `Bearer ${memberKey.current}`);
      request.upload.onprogress = event => { if (event.lengthComputable) setUploadProgress(Math.round(event.loaded / event.total * 100)); };
      request.onload = () => { try { const result = JSON.parse(request.responseText); request.status < 300 ? resolve(result) : reject(new Error(result.error)); } catch { reject(new Error("Upload response was unreadable. Resume from saved jobs.")); } };
      request.onerror = () => reject(new Error("Transfer interrupted. Check saved jobs before retrying."));
      request.send(JSON.stringify({ filename: file.name, content: btoa(binary), sha256: digest }));
    });
    setJob(result); setPaused(false); setUploadProgress(null);
  }
  async function openRun(id: string) {
    const result = await api(`runs/${id}`); setRun(result); setDataset(await api(`datasets/${result.datasetId}`));
    setStep(3); setSection("analysis"); setRecord(null); setAnswer(null); setEntity("all"); setComparison(null);
  }
  async function download(path: string, filename: string) {
    const response = await fetch(`/api/lab/v1/${path}`, { headers: memberKey.current ? { Authorization: `Bearer ${memberKey.current}` } : {} });
    if (!response.ok) throw new Error("Download failed. Your session may have expired.");
    const href = URL.createObjectURL(await response.blob()); const a = document.createElement("a"); a.href = href; a.download = filename; a.click(); URL.revokeObjectURL(href);
  }
  const entityOptions = Array.from(new Set((run?.observations || []).map((o: Document) => o.entity))) as string[];
  const visibleFindings = (run?.findings || []).filter((f: Document) => (entity === "all" || f.entity === entity) && `${f.what} ${f.entity}`.toLowerCase().includes(filter.toLowerCase()));
  const points = (run?.observations || []).filter((o: Document) => o.actual !== undefined && (entity === "all" || o.entity === entity));
  const shownPoints = points.slice(-Math.max(1, Math.round(points.length * zoom / 100)));

  return <div className="guided-lab">
    <a className="gl-skip" href="#analysis-content">Skip to analysis</a>
    <header className="gl-header"><Link href="/" className="gl-brand"><span className="gl-mark">E</span><span>Eidos <b>Sentinel Lab</b></span></Link>
      <nav aria-label="Workspace"><button aria-current={section === "analysis" ? "page" : undefined} onClick={() => setSection("analysis")}>Analysis</button><button onClick={() => { setSection("history"); void refresh(); }}>Saved work</button><button onClick={() => { setSection("monitors"); void refresh(); }}>Monitors</button><Link href="/research">Research console ↗</Link></nav>
      <span className="gl-account">{user ? "Private workspace" : "Private analysis"}</span></header>
    <main id="analysis-content" className="gl-main">
      <div className="gl-intro"><div><p className="gl-eyebrow">Evidence before explanation</p><h1>Understand what changed.</h1><p>Bring your data. Confirm its meaning. Follow every finding back to the source.</p></div><span className="gl-pill">Engineering preview · proof gates unchanged</span></div>
      {!user && <form className="gl-signin" onSubmit={e => { e.preventDefault(); void act(signIn); }}><div><h2>Your data stays in your workspace.</h2><p>Use an approved Lab key or your Eidos member agent key. Credentials are never included in analysis.</p></div><label>Access key<input type="password" autoComplete="off" value={key} onChange={e => setKey(e.target.value)} required /></label><button className="gl-primary" disabled={busy || !key}>Sign in</button><a href="https://eidos-works.com/lab" target="_blank" rel="noreferrer">Request access ↗</a></form>}
      {user && <div className="gl-session"><span>Signed in · {user.credential} access · results retained for 30 days</span><button onClick={() => void act(async () => { await api("session", "DELETE"); memberKey.current = ""; setUser(null); setDataset(null); setRun(null); setJob(null); setDatasets([]); setRuns([]); setMonitors([]); })}>Sign out</button></div>}
      {error && <div role="alert" className="gl-error"><b>Action needed</b><p>{error}</p><button onClick={() => setError("")}>Dismiss</button></div>}
      {notice && <p role="status" className="gl-notice">{notice}</p>}
      {job && <section className="gl-job" aria-live="polite"><div><span className="gl-eyebrow">{job.operation} · {job.id}</span><h3>{job.status.replaceAll("_", " ")}</h3><p>{job.error || "Stage reflects the worker state. Runtime is bounded to 15 minutes; it is not a completion estimate."}</p></div><div className="gl-actions">{!["completed", "failed", "cancelled", "expired"].includes(job.status) && <><button onClick={() => void act(async () => { setPaused(false); setJob(await api(`jobs/${job.id}/resume`, "POST")); })}>{paused ? "Resume status" : "Retry queued launch"}</button><button onClick={() => void act(async () => setJob(await api(`jobs/${job.id}/cancel`, "POST")))}>Cancel job</button></>}<button onClick={() => setJob(null)}>Hide</button></div></section>}
      {section === "history" && <section className="gl-panel"><h2>Saved work</h2><p>Inputs, interpretations and results outlive the compute session. Expired records remain archived; this release does not delete historical data.</p><div className="gl-three"><div><h3>Datasets</h3>{!datasets.length && <p>No saved datasets yet.</p>}{datasets.map(d => <button className="gl-history-item" key={d.id} disabled={d.expired} onClick={() => void act(() => selectDataset(d.id))}>{d.filename}<small>{d.rowCount} records · {d.expired ? "expired" : d.confirmed ? "confirmed" : "needs confirmation"}</small></button>)}</div><div><h3>Analyses</h3>{!runs.length && <p>No analyses yet.</p>}{runs.map(r => <button className="gl-history-item" key={r.id} disabled={r.expired} onClick={() => void act(() => openRun(r.id))}>{r.method}<small>{r.id.slice(-12)} · {new Date(r.createdAt).toLocaleString()}</small></button>)}</div><div><h3>Jobs & recovery</h3>{jobs.map(j => <button className="gl-history-item" key={j.id} onClick={() => void act(async () => { setJob(await api(`jobs/${j.id}`)); setPaused(false); })}>{j.status}<small>{j.id}</small></button>)}</div></div></section>}
      {section === "monitors" && <section className="gl-panel"><div className="gl-section-heading"><div><p className="gl-eyebrow">Continuous evidence</p><h2>Connect a measurement.</h2></div><a href="/connector">Collector setup & replay guide ↗</a></div><p>Connect an authorized OpenTelemetry collector or replay JSONL events. A website URL does not give access to server telemetry.</p>
        <form className="gl-monitor-form" onSubmit={e => { e.preventDefault(); void act(async () => { setConnector(await api("monitors", "POST", { name: sourceName, target: sourceTarget, unit: sourceUnit, autoProcess: automatic, synthetic })); await refresh(); }); }}><label>Source name<input value={sourceName} onChange={e => setSourceName(e.target.value)} required maxLength={80} placeholder="API response latency" /></label><label>OTLP metric name<input value={sourceTarget} onChange={e => setSourceTarget(e.target.value)} required /></label><label>Unit<input value={sourceUnit} onChange={e => setSourceUnit(e.target.value)} required /></label><label className="gl-check"><input type="checkbox" checked={automatic} onChange={e => setAutomatic(e.target.checked)} />Process automatically, up to four batches/day</label><label className="gl-check"><input type="checkbox" checked={synthetic} onChange={e => setSynthetic(e.target.checked)} />Synthetic validation source</label><button className="gl-primary" disabled={!user || busy}>Create source</button></form>
        {connector && <div className="gl-key"><h3>Save this scoped credential</h3><p>Shown once. Keep it in your collector environment, separate from its configuration file.</p><label>Ingest key<input type="password" readOnly value={connector.key} /></label><button onClick={() => void navigator.clipboard.writeText(connector.key).then(() => setNotice("Credential copied. Store it securely."))}>Copy key</button><p><code>{typeof window !== "undefined" ? window.location.origin : ""}/api/lab/v1/telemetry/{connector.monitor?.id || connector.sourceId}</code></p><button onClick={() => setConnector(null)}>I saved it</button></div>}
        {!monitors.length && <p className="gl-empty">No sources connected. Your first source will show arrival health, warmup, buffered events and recovery state here.</p>}
        {monitors.map(m => <article className="gl-monitor" key={m.id}><div className="gl-section-heading"><div><h3>{m.name}</h3><p>{m.target} · {m.unit} · {m.synthetic ? "synthetic validation" : "user telemetry"}</p></div><span className="gl-pill">{m.health}</span></div><div className="gl-metrics"><Metric name="Processed offset" value={m.processedOffset} /><Metric name="Buffered" value={m.buffered} /><Metric name="Warmup remaining" value={m.warmupRemaining} /><Metric name="Late events" value={m.checkpoint?.late ?? 0} /><Metric name="Telemetry gaps" value={m.checkpoint?.gaps ?? 0} /></div><p>{m.policy}</p><div className="gl-actions"><button onClick={() => void act(async () => { setJob(await api(`monitors/${m.id}/process`, "POST")); setPaused(false); })}>Process buffered events</button><button onClick={() => void act(async () => { await api(`monitors/${m.id}/revoke`, "POST"); setNotice("All ingestion keys for this source were revoked."); })}>Revoke ingestion keys</button><button onClick={() => void act(async () => setConnector({ ...await api(`monitors/${m.id}/key`, "POST"), sourceId: m.id }))}>Issue new key</button><button onClick={() => void act(refresh)}>Refresh health</button></div><details><summary>Recent event evidence</summary><pre>{JSON.stringify(m.lastOutcome?.outcomes || m.events, null, 2)}</pre></details><details><summary>Reset learned state</summary><form onSubmit={e => { e.preventDefault(); const data = new FormData(e.currentTarget); void act(async () => { await api(`monitors/${m.id}/reset`, "POST", { reason: data.get("reason") }); await refresh(); }); }}><label>Reason for relearning<input name="reason" minLength={10} required /></label><button>Archive checkpoint and reset</button></form></details></article>)}
      </section>}
      {section === "analysis" && <><ol className="gl-steps" aria-label="Analysis steps">{steps.map((label, i) => <li key={label}><button aria-current={step === i ? "step" : undefined} disabled={i > 0 && !dataset || i === 2 && !dataset?.confirmed || i === 3 && !run} onClick={() => setStep(i)}><span>{String(i + 1).padStart(2, "0")}</span>{label}</button></li>)}</ol>
      <section className="gl-panel"><div className="gl-section-heading"><div><p className="gl-eyebrow">Step {step + 1} of 4</p><h2 ref={heading} tabIndex={-1}>{steps[step]}</h2></div>{dataset && <span className="gl-pill">{dataset.filename} · {dataset.rowCount} records</span>}</div>
      {step === 0 && <div className="gl-add"><div className="gl-drop" onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); const file = e.dataTransfer.files[0]; if (file) void act(() => upload(file)); }}><span className="gl-upload-icon" aria-hidden="true">↑</span><h3>A clearer picture starts here.</h3><p>Drop your file, or choose one to upload.</p><input ref={fileInput} aria-label="Upload data file" type="file" accept=".csv,.xlsx,.parquet,.json,.jsonl,.ndjson,.log,.pdf,.html,.htm,.txt,.md" disabled={!user || busy} onChange={e => { const file = e.target.files?.[0]; if (file) void act(() => upload(file)); e.target.value = ""; }} /><p className="gl-small">CSV · XLSX · Parquet · JSON · JSONL · logs · text PDF · HTML · text</p><p className="gl-small">2 MB per file · 5,000 rows · 64 columns · 256 passages</p>{uploadProgress !== null && <label>Transferred {uploadProgress}%<progress max={100} value={uploadProgress} /></label>}</div><div className="gl-source-options"><form onSubmit={e => { e.preventDefault(); retry.current = freshKey(); void act(() => launch("imports", { url })); }}><h3>Import a public source</h3><p>Downloadable data is parsed as a dataset. Web pages become source-linked passages.</p><label>Public HTTPS URL<input type="url" placeholder="https://example.org/data.csv" value={url} onChange={e => setUrl(e.target.value)} required /></label><button disabled={!user || busy}>Import source →</button><p className="gl-small">Private networks and authenticated pages require an explicitly configured connector.</p></form><div><h3>Choose a dataset</h3><p>The existing Kaggle workflow keeps versioned downloads and research receipts.</p><Link href="/research">Open Kaggle dataset lookup ↗</Link></div><div><h3>Connect telemetry</h3><p>Stream a named measurement with a scoped credential and visible connection health.</p><button onClick={() => setSection("monitors")}>Set up a source →</button></div></div></div>}
      {step === 1 && dataset && <><p>Confirm what each record represents. File order alone is never treated as chronology.</p><div className="gl-metrics"><Metric name={dataset.kind === "documents" ? "Passages" : "Rows"} value={dataset.rowCount} /><Metric name="Columns" value={dataset.columns.length} /><Metric name="Duplicate rows" value={dataset.duplicates ?? null} /><Metric name="Source bytes" value={dataset.bytes} /></div>{dataset.warnings.map((w: string, i: number) => <p className="gl-warning" key={i}>{w}</p>)}
        {dataset.kind === "table" ? <><div className="gl-table-wrap"><table><caption>Inferred schema — original values preserved</caption><thead><tr><th>Measurement</th><th>Type</th><th>Missing</th><th>Distinct</th><th>Role / units</th></tr></thead><tbody>{dataset.columns.map((c: Document) => <tr key={c.name}><td><label className="gl-check"><input type="checkbox" checked={mapping.features.includes(c.name)} disabled={c.labelCandidate || c.idCandidate || c.constant || c.type !== "number"} onChange={e => setMapping({ ...mapping, features: e.target.checked ? [...mapping.features, c.name] : mapping.features.filter((f: string) => f !== c.name), units: { ...mapping.units, [c.name]: mapping.units[c.name] || "unknown" } })} />{c.name}</label></td><td>{c.type}</td><td>{c.missing}</td><td>{c.distinct}</td><td>{c.labelCandidate ? "Evaluator label — excluded" : c.idCandidate ? "Identifier — excluded" : c.constant ? "Constant — excluded" : mapping.features.includes(c.name) ? <input aria-label={`${c.name} units`} value={mapping.units[c.name] || "unknown"} onChange={e => setMapping({ ...mapping, units: { ...mapping.units, [c.name]: e.target.value } })} /> : "Not selected"}</td></tr>)}</tbody></table></div><div className="gl-form-grid"><label>One row represents<input value={mapping.meaning} onChange={e => setMapping({ ...mapping, meaning: e.target.value })} placeholder="One latency measurement for one service" /></label><label>Reference group<input value={mapping.reference} onChange={e => setMapping({ ...mapping, reference: e.target.value })} /></label><ColumnSelect label="Timestamp" columns={dataset.columns} value={mapping.timestamp} empty="Unordered table — no chronology" onChange={value => setMapping({ ...mapping, timestamp: value })} /><label>Time zone, if timestamps have no offset<input placeholder="e.g. America/New_York" value={mapping.timezone} onChange={e => setMapping({ ...mapping, timezone: e.target.value })} /></label><ColumnSelect label="Entity key" columns={dataset.columns} value={mapping.entity} empty="One entity" onChange={value => setMapping({ ...mapping, entity: value })} /><ColumnSelect label="Session key" columns={dataset.columns} value={mapping.session} empty="One session per entity" onChange={value => setMapping({ ...mapping, session: value })} /><label>Missing values<select value={mapping.missing} onChange={e => setMapping({ ...mapping, missing: e.target.value })}><option value="exclude">Exclude incomplete measurement rows</option><option value="prefix_median">Use calibration-prefix medians (temporal only)</option></select></label><label>Out-of-order timestamps<select value={mapping.ordering} onChange={e => setMapping({ ...mapping, ordering: e.target.value })}><option value="sort">Sort explicitly within each entity/session</option><option value="reject">Reject; repair at the source</option></select></label></div><details><summary>Preview original records</summary><pre>{JSON.stringify(dataset.records.slice(0, 5), null, 2)}</pre></details></> : <><p>Document collections use unordered semantic retrieval. No time sequence or numeric forecast is invented.</p>{dataset.passages.slice(0, 3).map((p: Document) => <blockquote key={p.id}><p>{p.text}</p><cite>Page {p.page} · {p.id}</cite></blockquote>)}</>}
        <button className="gl-primary" disabled={busy || !user} onClick={() => { retry.current = freshKey(); void act(() => launch(`datasets/${dataset.id}/confirm`, mapping)); }}>Confirm interpretation →</button></>}
      {step === 2 && dataset && <><p>Analysis follows your confirmed data contract. Research controls stay separate from the main workflow.</p><div className="gl-mode-grid">{[["unusual", "Find unusual behavior", "Compare measurements with the reference you confirmed."], ["forecast", "Forecast a measurement", "Issue a named target and horizon, with measured error and uncertainty."], ["patterns", "Explore patterns", "Inspect structure and source-linked associations."]].map(([id, title, text]) => <label className={`gl-mode ${mode === id ? "selected" : ""}`} key={id}><input type="radio" name="mode" value={id} checked={mode === id} disabled={id === "forecast" && dataset.confirmed.mode !== "temporal"} onChange={() => setMode(id)} /><strong>{title}</strong><span>{text}</span>{id === "forecast" && dataset.confirmed.mode !== "temporal" && <small>Requires confirmed numeric chronology.</small>}</label>)}</div>{dataset.confirmed.mode === "temporal" && <div className="gl-form-grid"><label>Named target<select value={target} onChange={e => setTarget(e.target.value)}>{dataset.confirmed.features.map((f: string) => <option key={f}>{f}</option>)}</select></label><label>Forecast horizon (seconds)<input type="number" min={1} max={86400} value={horizon} onChange={e => setHorizon(Number(e.target.value))} /></label><label>Maximum late-match window (seconds)<input type="number" min={1} max={86400} value={windowSeconds} onChange={e => setWindowSeconds(Number(e.target.value))} /></label></div>}<details><summary>Advanced · method and workload</summary><p>{dataset.kind === "documents" ? "Pinned all-MiniLM-L6-v2 semantic encoder; 384 dimensions, up to 256 source passages." : dataset.confirmed.mode === "temporal" ? "Causal Torch Eidos RLS profile: 64 reservoir units, no lossy feature projection. Predictor choice uses historical errors; persistence selections are identified. At least 48 records per entity/session." : "Isolation Forest and median/MAD baseline on the whole reference dataset; no temporal engine claim."}</p><p>One shared job at a time. A worker is limited to 15 minutes and the existing 4 vCPU / 8 GB Sandbox envelope. No per-row language-model calls. Coverage under drift is not guaranteed.</p><pre>{JSON.stringify(dataset.confirmed, null, 2)}</pre></details><button className="gl-primary" disabled={busy || !user} onClick={() => { retry.current = freshKey(); void act(() => launch(`datasets/${dataset.id}/analyze`, { mode, target, horizonSeconds: horizon, windowSeconds })); }}>Run analysis →</button></>}
      {step === 3 && run && dataset && <><div className="gl-result-heading"><div><p className="gl-badge">{run.sourceBadge}</p><h3>{run.findings.length ? `${run.findings.length} finding${run.findings.length === 1 ? "" : "s"} to inspect` : "No findings crossed this run's threshold"}</h3><p>{run.method}</p></div><button onClick={() => void act(() => download(`runs/${run.id}/download`, `${run.id}.json`))}>Download results ↓</button></div><div className="gl-metrics">{run.encoder ? <><Metric name="Indexed passages" value={run.metrics.passages} /><Metric name="Semantic dimensions" value={run.metrics.dimensions} /><Metric name="Language-model calls" value={run.metrics.llmCalls} /></> : <><Metric name="Forecast MAE" value={run.metrics.forecastMAE} /><Metric name="Persistence MAE" value={run.metrics.persistenceMAE} /><Metric name="Interval coverage" value={run.metrics.intervalCoverage} /><Metric name="Precision" value={run.metrics.precision} /></>}</div><p className="gl-small">Coverage is the observed fraction of matched outcomes inside issued bands. Missing metrics are not zero. Anomaly is not consequence; an empty finding list does not establish safety.</p><div className="gl-filters"><label>Entity<select disabled={!entityOptions.length} value={entity} onChange={e => setEntity(e.target.value)}><option value="all">All entities</option>{entityOptions.map(e => <option key={e}>{e}</option>)}</select></label><label>Filter findings<input type="search" value={filter} onChange={e => setFilter(e.target.value)} placeholder="Measurement or entity" /></label><label>Recent window: {zoom}%<input type="range" disabled={!points.length} min={5} max={100} step={5} value={zoom} onChange={e => setZoom(Number(e.target.value))} /></label></div>
        {shownPoints.length > 0 && <EvidencePlot points={shownPoints} onSelect={id => void act(async () => setRecord(await api(`datasets/${dataset.id}?record=${encodeURIComponent(id)}`)))} />}
        {run.forecasts?.length > 0 && <details><summary>Issued forecasts and uncertainty ({run.forecasts.length})</summary><p>Predictions below were committed before their target observations. Bands summarize accepted past errors and may under-cover during drift. Latest 100 issuances shown; download includes every issuance and commitment.</p><div className="gl-table-wrap"><table><thead><tr><th>Issued at</th><th>Target / horizon</th><th>Prediction / band</th><th>Predictor</th><th>Outcome</th></tr></thead><tbody>{run.forecasts.slice(-100).map((f: Document) => <tr key={f.id}><td>{f.issuedAt}</td><td>{f.target} · {f.horizonSeconds}s · {f.unit}</td><td>{format(f.prediction)} · [{format(f.lower)}, {format(f.upper)}]</td><td>{f.predictor}</td><td>{f.resolution}</td></tr>)}</tbody></table></div></details>}
        <div className="gl-investigate-grid"><div>{visibleFindings.length === 0 && <p className="gl-empty">No findings match this filter. Original records and run diagnostics remain available.</p>}{visibleFindings.map((finding: Document) => <article className="gl-finding" key={finding.id}><p className="gl-eyebrow">{finding.detector?.split("@")[0]}</p><h3>{finding.what}</h3><p>{finding.whyItMatters}</p><dl><dt>Evidence basis</dt><dd>{finding.basis}</dd><dt>Uncertainty</dt><dd>{finding.uncertainty.join(" ")}</dd><dt>Next action</dt><dd>{finding.nextAction}</dd></dl><details><summary>Grouping and ranking</summary><p>{finding.grouping}</p><p>{finding.ranking}</p></details><div className="gl-record-links">{finding.members.slice(0, 12).map((member: Document) => <button key={member.recordId || member.passageId} onClick={() => void act(async () => setRecord(await api(`datasets/${dataset.id}?record=${encodeURIComponent(member.recordId || member.passageId)}`)))}>View {member.recordId || member.passageId} ↗</button>)}</div><div className="gl-actions"><button onClick={() => { setSelectedFinding(finding.id); setQuestion("Why was this flagged?"); }}>Ask about this</button><select aria-label={`Review ${finding.id}`} defaultValue="" onChange={e => { if (e.target.value) void act(async () => { await api(`runs/${run.id}/feedback`, "POST", { findingId: finding.id, review: e.target.value }); setNotice("Review saved with provenance. Frozen labels, scores and training are unchanged."); }); }}><option value="">Review finding…</option><option value="expected">Expected behavior</option><option value="useful">Useful finding</option><option value="false_alarm">False alarm</option><option value="investigate">Needs investigation</option></select></div></article>)}</div><aside className="gl-evidence"><h3>Original evidence</h3>{record ? <><p>{record.source} · {record.id}</p><pre>{JSON.stringify(record.record, null, 2)}</pre><p className="gl-hash">Source SHA-256 {record.sourceSha256}</p></> : <p>Select a finding member or chart point to inspect the original record or passage.</p>}<button onClick={() => void act(() => download(`datasets/${dataset.id}/source`, dataset.filename))}>Download original input ↓</button><h3>Ask the result</h3><div className="gl-question-chips">{["Why was this flagged?", "What changed?", "Has this happened before?", "Which records support this?"].map(q => <button key={q} onClick={() => setQuestion(q)}>{q}</button>)}</div><form onSubmit={e => { e.preventDefault(); retry.current = freshKey(); void act(async () => { const result = await api(`runs/${run.id}/questions`, "POST", { question, findingId: selectedFinding }, retry.current!); if (result.status) { setJob(result); setPaused(false); } else setAnswer(result); }); }}><label>Question<textarea aria-label="Question" maxLength={500} value={question} onChange={e => setQuestion(e.target.value)} /></label><button className="gl-primary" disabled={busy}>Find supporting evidence</button></form>{answer && <div aria-live="polite"><p>{answer.answer}</p>{answer.citations?.map((citation: Document) => <blockquote key={citation.id}><p>{citation.text || JSON.stringify(citation.record)}</p><cite>{citation.source} · {citation.id}</cite><button onClick={() => void act(async () => setRecord(await api(`datasets/${dataset.id}?record=${encodeURIComponent(citation.id)}`)))}>Inspect source</button></blockquote>)}<p className="gl-small">{answer.method}</p>{answer.reviews?.length > 0 && <details><summary>Reviewed outcomes</summary><pre>{JSON.stringify(answer.reviews, null, 2)}</pre></details>}{answer.similarHistoricalEvents?.length > 0 && <details><summary>Similar historical findings</summary><pre>{JSON.stringify(answer.similarHistoricalEvents, null, 2)}</pre></details>}</div>}</aside></div>
        <details><summary>Compare a saved run</summary><label>Run<select defaultValue="" onChange={e => { if (e.target.value) void act(async () => setComparison(await api(`runs/${e.target.value}`))); }}><option value="">Select a run</option>{runs.filter(r => r.id !== run.id).map(r => <option key={r.id} value={r.id}>{r.id.slice(-12)} · {r.method}</option>)}</select></label>{comparison && <><p>Compare only matching targets, units and references. These runs may use different datasets or methods; differences alone do not establish improvement.</p><table><thead><tr><th>Metric</th><th>This run</th><th>Selected run</th></tr></thead><tbody>{["forecastMAE", "persistenceMAE", "intervalCoverage", "intervalMeanWidth", "precision", "recall"].map(metric => <tr key={metric}><th>{metric}</th><td>{format(run.metrics[metric])}</td><td>{format(comparison.metrics[metric])}</td></tr>)}</tbody></table></>}</details><details><summary>Reproducibility and remaining uncertainty</summary><p>Input SHA-256: <code className="gl-hash">{run.inputSha256}</code></p>{run.limitations.map((text: string) => <p key={text}>{text}</p>)}<pre>{JSON.stringify({ engine: run.engine, receipt: run.receipt, metrics: run.metrics }, null, 2)}</pre></details><Link href="/research">Open research console & recorded engine observatory ↗</Link></>}
      </section></>}
      <footer className="gl-footer"><span>Eidos · inspect, reproduce, challenge.</span><span>Private results · bounded compute · source-linked evidence</span></footer>
    </main>
  </div>;
}

function Metric({ name, value }: { name: string; value: unknown }) { return <div className="gl-metric"><span>{name}</span><strong>{format(value)}</strong></div>; }
function ColumnSelect({ label, columns, value, empty, onChange }: { label: string; columns: Document[]; value: string; empty: string; onChange: (value: string) => void }) {
  return <label>{label}<select aria-label={label} value={value} onChange={e => onChange(e.target.value)}><option value="">{empty}</option>{columns.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}</select></label>;
}
function EvidencePlot({ points, onSelect }: { points: Document[]; onSelect: (id: string) => void }) {
  const sampled = points.filter((_, i) => i % Math.max(1, Math.ceil(points.length / 240)) === 0);
  const values = sampled.flatMap(p => [p.actual, p.expected, p.lower, p.upper]).filter(v => typeof v === "number");
  const min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
  const x = (i: number) => 45 + i / Math.max(1, sampled.length - 1) * 910;
  const y = (v: number) => 225 - (v - min) / span * 185;
  const path = (key: string) => sampled.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p[key]).toFixed(1)}`).join(" ");
  return <figure className="gl-plot"><figcaption><strong>Observed and committed forecast</strong><span>Actual <i className="gl-actual" /> Expected <i className="gl-expected" /> · {sampled[0]?.unit}</span></figcaption><svg viewBox="0 0 1000 270" role="img" aria-label="Actual measurements and previously committed forecasts. The table below provides keyboard access.">{[0, .25, .5, .75, 1].map(f => <g key={f}><line x1={45} x2={955} y1={y(min + span * f)} y2={y(min + span * f)} stroke="#303d3b" /><text x={4} y={y(min + span * f) + 4} fill="#bdc4bf" fontSize={10}>{format(min + span * f)}</text></g>)}<path d={sampled.filter(p => p.lower != null && p.upper != null).map(p => `${sampled.indexOf(p) === sampled.findIndex(p => p.lower != null) ? "M" : "L"}${x(sampled.indexOf(p))},${y(p.upper)}`).join(" ") + " " + sampled.filter(p => p.lower != null && p.upper != null).reverse().map(p => `L${x(sampled.indexOf(p))},${y(p.lower)}`).join(" ") + " Z"} fill="#c4a271" opacity={0.12} /><path d={path("expected")} fill="none" stroke="#dca767" strokeWidth={1.6} strokeDasharray="5 3" /><path d={path("actual")} fill="none" stroke="#70ccb2" strokeWidth={2} />{sampled.filter(p => p.anomaly).map(p => { const i = sampled.indexOf(p); return <circle key={p.recordId} cx={x(i)} cy={y(p.actual)} r={4} fill="#eea176" />; })}<text x={45} y={255} fill="#bdc4bf" fontSize={10}>{sampled[0]?.time}</text><text x={955} y={255} textAnchor="end" fill="#bdc4bf" fontSize={10}>{sampled.at(-1)?.time}</text></svg><details><summary>Explore plotted records ({points.length})</summary><div className="gl-table-wrap"><table><thead><tr><th>Record</th><th>Observed at</th><th>Actual</th><th>Expected</th><th>Band</th><th>Score</th></tr></thead><tbody>{points.slice(-100).map(p => <tr key={`${p.recordId}-${p.forecastId}`}><td><button onClick={() => onSelect(p.recordId)}>{p.recordId}</button></td><td>{p.time}</td><td>{format(p.actual)}</td><td>{format(p.expected)}</td><td>{format(p.lower)} – {format(p.upper)}</td><td>{format(p.score)}</td></tr>)}</tbody></table></div><p>Latest 100 rows shown. Download results for all forecasts and issuance times.</p></details></figure>;
}
