"use client";

import { useEffect, useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
import { ArrowIcon, CloseIcon, DownloadIcon, EidosMark, MenuIcon, PlayIcon, UploadIcon } from "@/components/icons";
import { RealDataLab } from "@/components/real-data-lab";
import { TraceChart } from "@/components/trace-chart";
import type { ImportedArtifact, ScenarioId, SmokeRequest, SmokeResult } from "@/lib/sentinel/types";

const SCENARIOS: Array<{ value: ScenarioId; label: string }> = [
  { value: "S0_nominal", label: "S0 · Nominal carrier" },
  { value: "S1_hidden_backdoor", label: "S1 · Hidden backdoor" },
  { value: "S2_slow_drift", label: "S2 · Slow drift" },
  { value: "S3_regime_shift", label: "S3 · Regime shift" },
  { value: "S6_noise_thrash", label: "S6 · Noise thrash" },
  { value: "S7_harmless_repeat", label: "S7 · Harmless repeat" },
  { value: "S8_dangerous_repeat", label: "S8 · Consequential repeat" },
  { value: "C1_nuisance_subspace", label: "C1 · Nuisance subspace" },
];

const GATES = [
  { id: "G0", title: "Reproducibility", criterion: "Same locks, manifests, and seeds reproduce the claimed artifact set." },
  { id: "G1", title: "Causality + label isolation", criterion: "Past-only execution and blinded labels are proven from immutable traces." },
  { id: "G2", title: "Safety", criterion: "Escape paths and adverse-condition checks satisfy preregistered bounds." },
  { id: "G3", title: "Mechanisms", criterion: "Targeted ablations support the claimed contribution of each mechanism." },
  { id: "G4", title: "Joint value", criterion: "The locked observer beats preregistered baselines across required metrics." },
  { id: "G5", title: "Cross-domain", criterion: "At least one resource-qualified natural-domain test clears its lock." },
  { id: "G6", title: "Independent operator", criterion: "An independent operator reproduces the run from the handoff package." },
];

type Tab = "lab" | "real-data" | "evidence" | "gates" | "compare";
const HISTORY_KEY = "eidos.sentinel.v1.history";
const ARTIFACTS_KEY = "eidos.sentinel.v1.artifacts";

function formatMetric(value: number) {
  return Number.isFinite(value) ? value.toFixed(3) : "—";
}

function evidenceRows(run: SmokeResult) {
  return [
    ["01 / OBSERVATION", run.incident.observation],
    ["02 / WHY THIS MATTERS", run.incident.why],
    ["03 / WHAT WOULD DISCONFIRM", run.incident.disconfirm],
    ["04 / NEXT ACTION", run.incident.action],
    ["05 / UNCERTAINTY", run.incident.uncertainty],
  ];
}

async function digestText(text: string) {
  const bytes = new TextEncoder().encode(text);
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(hash), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function artifactKind(name: string, value: unknown): ImportedArtifact["kind"] {
  const haystack = `${name} ${JSON.stringify(value ?? "")}`.toLowerCase();
  if (haystack.includes("verdict")) return "verdict";
  if (haystack.includes("run_lock") || haystack.includes("run-lock") || haystack.includes("lock_id")) return "run-lock";
  if (haystack.includes("metric")) return "metrics";
  if (haystack.includes("event")) return "events";
  return "unknown";
}

function deepFindString(value: unknown, keys: string[]): string | undefined {
  if (!value || typeof value !== "object") return undefined;
  for (const [key, item] of Object.entries(value)) {
    if (keys.includes(key.toLowerCase()) && typeof item === "string") return item;
    const nested = deepFindString(item, keys);
    if (nested) return nested;
  }
  return undefined;
}

async function normalizeArtifact(file: File): Promise<ImportedArtifact> {
  const content = await file.text();
  const digest = await digestText(content);
  let parsed: unknown = content;
  let records: number | undefined;
  try {
    if (file.name.endsWith(".jsonl")) {
      const rows = content.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
      parsed = rows;
      records = rows.length;
    } else {
      parsed = JSON.parse(content);
      records = Array.isArray(parsed) ? parsed.length : 1;
    }
  } catch {
    records = content.split(/\r?\n/).filter(Boolean).length;
  }
  const kind = artifactKind(file.name, parsed);
  return {
    id: `${digest.slice(0, 12)}-${file.name}`,
    filename: file.name,
    kind,
    digest,
    importedAt: new Date().toISOString(),
    verdict: deepFindString(parsed, ["verdict", "status"]),
    protocolId: deepFindString(parsed, ["protocol_id", "protocolid", "protocol"]),
    records,
    summary: `${kind === "unknown" ? "Unclassified" : kind.replace("-", " ")} artifact · ${records ?? 0} record${records === 1 ? "" : "s"}`,
  };
}

function EvidenceLedger({ run, compact = false }: { run: SmokeResult; compact?: boolean }) {
  return (
    <section className={compact ? "evidence-ledger compact" : "evidence-ledger"} aria-labelledby={compact ? "drawer-title" : "evidence-title"}>
      <div className="section-heading evidence-heading">
        <div><p className="eyebrow">FIVE-FIELD CARD</p><h2 id={compact ? "drawer-title" : "evidence-title"}>Incident evidence</h2></div>
        <span className="evidence-class">{run.evidenceClass}</span>
      </div>
      <dl className="evidence-fields">
        {evidenceRows(run).map(([label, body]) => <div className="evidence-field" key={label}><dt>{label}</dt><dd>{body}</dd></div>)}
      </dl>
      <div className="reference-block">
        <span>ARTIFACT REFERENCES</span>
        {run.incident.references.map((reference) => <code key={reference}>{reference}</code>)}
      </div>
    </section>
  );
}

function GateRail() {
  return (
    <section className="gate-rail" aria-labelledby="gate-rail-title">
      <div className="gate-rail-copy"><p className="eyebrow">PROOF LEDGER</p><h2 id="gate-rail-title">No gate advances on smoke evidence.</h2></div>
      <div className="gate-cells">
        {GATES.map((gate) => <div className="gate-cell" key={gate.id}><span>{gate.id}</span><strong>LOCKED</strong></div>)}
      </div>
    </section>
  );
}

export function SentinelLab({ initialRun }: { initialRun: SmokeResult }) {
  const [tab, setTab] = useState<Tab>("lab");
  const [menuOpen, setMenuOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [run, setRun] = useState(initialRun);
  const [scenario, setScenario] = useState<ScenarioId>(initialRun.scenario);
  const [seed, setSeed] = useState<0 | 1>(initialRun.seed);
  const [frames, setFrames] = useState<240 | 480 | 720>(initialRun.frames as 240 | 480 | 720);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<Array<Pick<SmokeResult, "runId" | "scenarioLabel" | "seed" | "frames">>>([]);
  const [artifacts, setArtifacts] = useState<ImportedArtifact[]>([]);
  const [importing, setImporting] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    try {
      const storedHistory = localStorage.getItem(HISTORY_KEY);
      const storedArtifacts = localStorage.getItem(ARTIFACTS_KEY);
      if (storedHistory) setHistory(JSON.parse(storedHistory));
      if (storedArtifacts) setArtifacts(JSON.parse(storedArtifacts));
    } catch {
      // Local history is navigation convenience, never proof state.
    }
  }, []);

  useEffect(() => {
    if (!drawerOpen) return;
    const close = (event: globalThis.KeyboardEvent) => event.key === "Escape" && setDrawerOpen(false);
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [drawerOpen]);

  function navigate(next: Tab) {
    setTab(next);
    setMenuOpen(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function executeRun() {
    setRunning(true);
    setError(null);
    const request: SmokeRequest = { scenario, seed, frames, system: "eidos_ms_v1_observer" };
    try {
      const response = await fetch("/api/smoke", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request) });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || "Smoke run failed.");
      const nextRun = body as SmokeResult;
      setRun(nextRun);
      setHistory((currentHistory) => {
        const nextHistory = [{ runId: nextRun.runId, scenarioLabel: nextRun.scenarioLabel, seed: nextRun.seed, frames: nextRun.frames }, ...currentHistory.filter((entry) => entry.runId !== nextRun.runId)].slice(0, 5);
        localStorage.setItem(HISTORY_KEY, JSON.stringify(nextHistory));
        return nextHistory;
      });
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : "Smoke run failed.");
    } finally {
      setRunning(false);
    }
  }

  function downloadRun() {
    const blob = new Blob([JSON.stringify(run, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${run.runId}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  async function importFiles(files: FileList | File[]) {
    const accepted = Array.from(files).filter((file) => /\.(json|jsonl|txt|md)$/i.test(file.name));
    if (!accepted.length) {
      setError("Choose JSON, JSONL, text, or Markdown proof artifacts.");
      return;
    }
    setImporting(true);
    setError(null);
    try {
      const incoming = await Promise.all(accepted.map(normalizeArtifact));
      setArtifacts((currentArtifacts) => {
        const merged = [...incoming, ...currentArtifacts.filter((existing) => !incoming.some((item) => item.digest === existing.digest))].slice(0, 20);
        localStorage.setItem(ARTIFACTS_KEY, JSON.stringify(merged));
        return merged;
      });
    } catch (importError) {
      setError(importError instanceof Error ? importError.message : "Artifact import failed.");
    } finally {
      setImporting(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  function handleFileInput(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files) void importFiles(event.target.files);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    void importFiles(event.dataTransfer.files);
  }

  return (
    <div className="site-shell">
      <header className="site-header">
        <button className="brand" onClick={() => navigate("lab")} aria-label="Open Sentinel Lab home"><EidosMark /><span>EIDOS <b>/</b> SENTINEL LAB</span></button>
        <nav className={menuOpen ? "primary-nav open" : "primary-nav"} aria-label="Primary navigation">
          {(["lab", "real-data", "evidence", "gates", "compare"] as Tab[]).map((item) => <button key={item} className={tab === item ? "active" : ""} onClick={() => navigate(item)}>{item}</button>)}
        </nav>
        <div className="header-status"><i /> PROTOCOL OPEN</div>
        <button className="menu-button" onClick={() => setMenuOpen((open) => !open)} aria-expanded={menuOpen} aria-label="Toggle navigation"><MenuIcon open={menuOpen} /></button>
      </header>

      <main>
        <section className="hero">
          <div className="hero-index">GP / 02</div>
          <div className="hero-copy"><p className="eyebrow">CURRENT EVIDENCE POSTURE · {run.protocol.id}</p><h1>Proof is blocked<br />before held-out.</h1><p className="hero-summary">The observer and proof harness exist. Resource-qualified held-out runs have not been completed.</p></div>
          <div className="verdict-card"><span>CURRENT VERDICT</span><strong>BLOCKED_RESOURCE_<br />BEFORE_HELDOUT</strong><small>0 / 7 gates advanced</small></div>
        </section>

        {error && <div className="error-banner" role="alert"><span>REQUEST HALTED</span>{error}<button onClick={() => setError(null)} aria-label="Dismiss error"><CloseIcon /></button></div>}

        {tab === "lab" && (
          <>
            <section className="workspace" aria-labelledby="lab-title">
              <aside className="config-rail">
                <div className="section-heading"><div><p className="eyebrow">PREREGISTERED SURFACE</p><h2 id="lab-title">Run engineering smoke</h2></div><span className="run-sequence">ENG / 00{history.length + 1}</span></div>
                <label><span>SCENARIO</span><select value={scenario} onChange={(event) => setScenario(event.target.value as ScenarioId)}>{SCENARIOS.map((option) => <option value={option.value} key={option.value}>{option.label}</option>)}</select></label>
                <div className="form-pair">
                  <label><span>SEED</span><select value={seed} onChange={(event) => setSeed(Number(event.target.value) as 0 | 1)}><option value={0}>0 · engineering</option><option value={1}>1 · engineering</option></select></label>
                  <label><span>FRAMES</span><select value={frames} onChange={(event) => setFrames(Number(event.target.value) as 240 | 480 | 720)}><option value={240}>240</option><option value={480}>480</option><option value={720}>720</option></select></label>
                </div>
                <label><span>SYSTEM</span><div className="locked-field">eidos_ms_v1_observer <b>LOCKED</b></div></label>
                <button className="run-button" onClick={executeRun} disabled={running}>{running ? <span className="running-dot" /> : <PlayIcon />}{running ? "Evaluating causal trace…" : "Run engineering smoke"}</button>
                <p className="config-note">Held-out seeds 100–119 are intentionally absent. This browser projection cannot advance a proof gate.</p>
                {history.length > 0 && <div className="run-history"><span>RECENT LOCAL RUNS</span>{history.slice(0, 3).map((entry) => <code key={entry.runId}>{entry.runId}</code>)}</div>}
              </aside>

              <div className="trace-panel">
                <div className="section-heading trace-heading"><div><p className="eyebrow">{run.runId}</p><h2>Causal observer trace</h2></div><div className="trace-stats"><span>PEAK RAW <b>{formatMetric(run.summary.peakRaw)}</b></span><span>QUOTIENT <b>{formatMetric(run.summary.peakQuotient)}</b></span><span>WINDOWS <b>{run.summary.candidateWindows}</b></span></div></div>
                <TraceChart key={run.runId} run={run} />
                <div className="trace-actions"><p><i /> {run.disclaimer}</p><div><button className="text-button mobile-evidence" onClick={() => setDrawerOpen(true)}>View evidence <ArrowIcon /></button><button className="text-button" onClick={downloadRun}><DownloadIcon /> Export run</button></div></div>
              </div>
              <div className="desktop-evidence"><EvidenceLedger run={run} compact /></div>
            </section>
            <GateRail />
          </>
        )}

        {tab === "real-data" && <RealDataLab />}

        {tab === "evidence" && (
          <section className="tab-page evidence-page">
            <div className="page-intro"><p className="eyebrow">ARTIFACT REVIEW</p><h2>Evidence before interpretation.</h2><p>Inspect the current five-field card, then register full-engine artifacts locally by content digest. Imports remain in this browser; they are not proof acceptance.</p></div>
            <div className="evidence-page-grid">
              <EvidenceLedger run={run} />
              <div className="artifact-column">
                <div className="drop-zone" onDragOver={(event) => event.preventDefault()} onDrop={handleDrop}><UploadIcon /><p className="eyebrow">LOCAL ARTIFACT INTAKE</p><h3>{importing ? "Hashing artifacts…" : "Drop proof artifacts here."}</h3><p>JSON, JSONL, text, or Markdown. Files are classified and SHA-256 hashed in your browser.</p><button className="outline-button" onClick={() => fileInput.current?.click()} disabled={importing}>Choose files</button><input ref={fileInput} type="file" multiple accept=".json,.jsonl,.txt,.md" onChange={handleFileInput} hidden /></div>
                <div className="artifact-list">
                  <div className="artifact-list-head"><span>REGISTERED LOCALLY</span><b>{artifacts.length.toString().padStart(2, "0")}</b></div>
                  {artifacts.length === 0 ? <p className="empty-state">No full-engine artifacts imported. The engineering-smoke run above is still inspectable and exportable.</p> : artifacts.map((artifact) => <article className="artifact-row" key={artifact.id}><div><span>{artifact.kind.toUpperCase()}</span><h4>{artifact.filename}</h4><p>{artifact.summary}</p></div><code title={artifact.digest}>{artifact.digest.slice(0, 12)}</code></article>)}
                </div>
              </div>
            </div>
          </section>
        )}

        {tab === "gates" && (
          <section className="tab-page gates-page">
            <div className="page-intro"><p className="eyebrow">GRAND PROOF · G0–G6</p><h2>Every claim waits on its gate.</h2><p>Smoke runs exercise the operator surface. Only resource-qualified, locked artifacts can change this ledger.</p></div>
            <div className="gate-list">{GATES.map((gate) => <article key={gate.id}><div className="gate-id">{gate.id}</div><div><h3>{gate.title}</h3><p>{gate.criterion}</p></div><span className="locked-badge">LOCKED</span></article>)}</div>
            <div className="gate-verdict"><span>LEDGER VERDICT</span><strong>BLOCKED_RESOURCE_BEFORE_HELDOUT</strong><small>Engineering smoke contributes zero accepting artifacts.</small></div>
          </section>
        )}

        {tab === "compare" && (
          <section className="tab-page compare-page">
            <div className="page-intro"><p className="eyebrow">ENGINEERING PROJECTION</p><h2>Observer against simple baselines.</h2><p>These numbers are derived from the current synthetic trace. They are diagnostic—not Grand Proof evidence.</p></div>
            <div className="comparison-context"><span>{run.scenarioLabel}</span><span>SEED {run.seed}</span><span>{run.frames} FRAMES</span></div>
            <div className="comparison-table-wrap"><table className="comparison-table"><thead><tr><th>DETECTOR</th><th>PEAK</th><th>EVENT FOUND</th><th>FIRST FRAME</th><th>FALSE ALERTS</th><th>METHOD NOTE</th></tr></thead><tbody>{run.comparisons.map((row) => <tr key={row.id} className={row.id === "eidos_ms_v1_observer" ? "primary-row" : ""}><td><strong>{row.label}</strong><code>{row.id}</code></td><td>{formatMetric(row.peak)}</td><td><span className={row.eventDetected ? "found" : "not-found"}>{row.eventDetected ? "YES" : "NO"}</span></td><td>{row.firstDetection ?? "—"}</td><td>{row.falseAlerts}</td><td>{row.note}</td></tr>)}</tbody></table></div>
            <p className="table-note">No ranking or production claim is implied. Run the locked full-engine protocol before interpreting comparative value.</p>
          </section>
        )}
      </main>

      <footer className="site-footer"><span>EIDOS-GP-v1 · LAB v0.2</span><p>ENGINEERING EVIDENCE DOES NOT ADVANCE PROOF GATES.</p><span>PAST-ONLY / LABEL-ISOLATED</span></footer>

      {drawerOpen && <div className="drawer-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && setDrawerOpen(false)}><div className="evidence-drawer" role="dialog" aria-modal="true" aria-labelledby="drawer-title"><button className="drawer-close" onClick={() => setDrawerOpen(false)} aria-label="Close evidence drawer"><CloseIcon /></button><EvidenceLedger run={run} compact /></div></div>}
    </div>
  );
}
