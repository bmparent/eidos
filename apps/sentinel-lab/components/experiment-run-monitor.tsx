"use client";

import { useEffect, useState } from "react";
import type { ExperimentStatus, RunnerDispatch } from "@/lib/experiments/types";
import { downloadJson, LabRequestError, requestJson } from "@/lib/experiments/client";

const STAGES = [
  "QUEUED",
  "BOOTSTRAPPING_RUNTIME",
  "PREPARING_DATASET",
  "RUNNING_FULL_ENGINE",
  "EVALUATING_FROZEN_PREDICTIONS",
  "COMPLETED_ENGINEERING",
];
const TERMINAL = new Set(["COMPLETED_ENGINEERING", "FAILED", "EXPIRED"]);

function readableStatus(value: string) {
  return value.toLowerCase().replaceAll("_", " ");
}

function ratio(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "N/A";
}

function count(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString() : "N/A";
}

type Props = {
  dispatch: RunnerDispatch;
  operatorToken: string;
  onError: (message: string) => void;
  onTerminal: () => void;
  onStatus: (status: ExperimentStatus) => void;
};

export function ExperimentRunMonitor({ dispatch, operatorToken, onError, onTerminal, onStatus }: Props) {
  const [status, setStatus] = useState<ExperimentStatus | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    let consecutiveFailures = 0;
    const controller = new AbortController();
    if (!operatorToken.trim()) return;
    setPollError(null);

    async function poll() {
      try {
        const next = await requestJson<ExperimentStatus>(`/api/experiments/${dispatch.jobId}`, {
          headers: { Authorization: `Bearer ${operatorToken}` },
          signal: controller.signal,
        }, 125_000);
        if (cancelled) return;
        if (next.jobId !== dispatch.jobId || ![...STAGES, "FAILED", "EXPIRED"].includes(next.status)) throw new Error("The server returned an unrecognized job receipt.");
        consecutiveFailures = 0;
        setPollError(null);
        setStatus(next);
        onStatus(next);
        if (TERMINAL.has(next.status)) onTerminal();
        else timeout = setTimeout(poll, 8_000);
      } catch (error) {
        if (cancelled) return;
        consecutiveFailures += 1;
        const unauthorized = error instanceof LabRequestError && [401, 403].includes(error.status);
        if (consecutiveFailures < 3 && !unauthorized) timeout = setTimeout(poll, 4_000 * consecutiveFailures);
        else setPollError(unauthorized ? "Operator token rejected. Re-enter it above, then check status again." : error instanceof Error ? error.message : "Experiment status lookup failed.");
      }
    }

    void poll();
    return () => {
      cancelled = true;
      controller.abort();
      if (timeout) clearTimeout(timeout);
    };
  }, [dispatch.jobId, onTerminal, onStatus, operatorToken, retry]);

  async function downloadArtifact(artifactName: string) {
    setDownloading(artifactName);
    try {
      const response = await fetch(`/api/experiments/${dispatch.jobId}/artifacts/${encodeURIComponent(artifactName)}`, {
        cache: "no-store",
          headers: { Authorization: `Bearer ${operatorToken}` },
          signal: AbortSignal.timeout(125_000),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || body.error || "Artifact download failed. Retry retrieval with your operator token.");
      }
      const href = URL.createObjectURL(await response.blob());
      const anchor = document.createElement("a");
      anchor.href = href;
      anchor.download = artifactName;
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(href), 0);
    } catch (error) {
      onError(error instanceof Error ? error.message : "Artifact download failed.");
    } finally {
      setDownloading(null);
    }
  }

  const current = status?.status || dispatch.status;
  const stageIndex = Math.max(0, STAGES.indexOf(current));
  const metrics = status?.metrics;

  return (
    <section className="job-monitor" aria-labelledby="job-monitor-title">
      <div className="job-monitor-head">
        <div><span>RUN RECEIPT</span><strong id="job-monitor-title">{dispatch.jobId}</strong></div>
        <small>{status?.executionBackend || dispatch.executionBackend ? `${(status?.executionBackend || dispatch.executionBackend)?.toUpperCase()} COMPUTE` : "AWAITING BACKEND"}</small>
      </div>
      <button className="outline-button" onClick={() => downloadJson(dispatch, `eidos-job-${dispatch.jobId}.json`)}>Save job receipt</button>
      {!operatorToken.trim() && <p className="job-detail" role="status">Enter your operator token above to reconnect. Your job can continue while this page is closed.</p>}
      {pollError && <div className="poll-recovery" role="alert"><strong>Status updates paused</strong><p>{pollError}</p><p>The last receipt below may be out of date. This does not mean the engine stopped.</p><button className="outline-button" onClick={() => setRetry((value) => value + 1)}>Retry status check</button></div>}

      <div className="job-stage" aria-live="polite">
        <span className={TERMINAL.has(current) ? current === "COMPLETED_ENGINEERING" ? "complete" : "failed" : "active"} />
        <div><b>{readableStatus(current)}</b><small>{status?.updatedAt ? `updated ${new Date(status.updatedAt).toLocaleTimeString()}` : "waiting for first runner receipt"}</small></div>
      </div>

      <div className="job-progress" aria-label={`Experiment stages, not time remaining: ${readableStatus(current)}`}>
        {STAGES.slice(0, -1).map((stage, index) => <i key={stage} className={index <= stageIndex ? "reached" : ""} />)}
      </div>

      {status?.rowsSentToEngine ? <p className="job-detail">{status.rowsSentToEngine.toLocaleString()} calibration + evaluation rows entered the label-free engine stream.</p> : null}
      {status?.detail ? <p className="job-failure"><b>{status.error || current}</b>{status.detail}</p> : null}

      {metrics ? (
        <div className="metric-receipt">
          <div><span>ROC AUC</span><strong>{ratio(metrics.roc_auc)}</strong></div>
          <div><span>AVG PRECISION</span><strong>{ratio(metrics.average_precision)}</strong></div>
          <div><span>PRECISION</span><strong>{ratio(metrics.precision)}</strong></div>
          <div><span>RECALL</span><strong>{ratio(metrics.recall)}</strong></div>
          <div><span>FALSE POSITIVE RATE</span><strong>{ratio(metrics.false_positive_rate)}</strong></div>
          <div><span>DETECTION DELAY</span><strong>{typeof metrics.mean_detection_delay_frames === "number" && Number.isFinite(metrics.mean_detection_delay_frames) ? `${metrics.mean_detection_delay_frames.toFixed(1)} fr` : "N/A"}</strong></div>
          <p>{count(metrics.evaluation_rows_scored)} frozen predictions scored · TP {count(metrics.confusion?.tp)} · FP {count(metrics.confusion?.fp)} · FN {count(metrics.confusion?.fn)} · TN {count(metrics.confusion?.tn)}</p>
          <p>{metrics.prediction_coverage_complete ? "Complete evaluation coverage verified." : "Legacy receipt: complete coverage was not enforced by this runner."} N/A means the metric cannot be estimated.</p>
          <div className="metric-explainer"><p><b>Recall</b> — the share of labeled attacks detected.</p><p><b>Precision</b> — the share of alerts that match labeled attacks.</p><p><b>False positive rate</b> — benign observations that triggered alerts.</p></div>
          {metrics.limitations?.map((item) => <p key={item}>{item}</p>)}
        </div>
      ) : null}

      {status?.engineDiagnostics && <div className="engine-receipt"><h4>What the engine actually used</h4><dl><div><dt>Reservoir / memory</dt><dd>{status.engineDiagnostics.reservoir_units.toLocaleString()} / {status.engineDiagnostics.memory_dimensions.toLocaleString()}</dd></div><div><dt>Time scales / TraceSeal</dt><dd>{status.engineDiagnostics.leak_bands} / {status.engineDiagnostics.trace_seal_enabled ? "on" : "off"}</dd></div><div><dt>Memory writes</dt><dd>{status.engineDiagnostics.memory_writes.toLocaleString()}</dd></div><div><dt>Regulation</dt><dd>{status.engineDiagnostics.thermodynamics_enabled ? "active" : "off"}</dd></div></dl><p>{status.engineDiagnostics.scope}</p></div>}

      {status?.artifacts?.length ? (
        <div className="artifact-actions">
          <span>AUDIT ARTIFACTS</span>
          {status.artifacts.map((artifact) => (
            <button key={artifact} disabled={downloading === artifact} onClick={() => void downloadArtifact(artifact)}>
              {downloading === artifact ? "Retrieving…" : artifact}
            </button>
          ))}
        </div>
      ) : null}

      <p className="job-proof-note">This receipt is engineering evidence only. Held-out data remains sealed and G0–G6 remain locked.</p>
    </section>
  );
}
