"use client";

import { useEffect, useState } from "react";
import type { ExperimentStatus, RunnerDispatch } from "@/lib/experiments/types";

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
  return value === null || value === undefined ? "N/A" : value.toFixed(3);
}

type Props = {
  dispatch: RunnerDispatch;
  operatorToken: string;
  onError: (message: string) => void;
};

export function ExperimentRunMonitor({ dispatch, operatorToken, onError }: Props) {
  const [status, setStatus] = useState<ExperimentStatus | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    let consecutiveFailures = 0;

    async function poll() {
      try {
        const response = await fetch(`/api/experiments/${dispatch.jobId}`, {
          cache: "no-store",
          headers: { Authorization: `Bearer ${operatorToken}` },
        });
        const body = await response.json();
        if (!response.ok) throw new Error(body.error || "Experiment status lookup failed.");
        if (cancelled) return;
        consecutiveFailures = 0;
        const next = body as ExperimentStatus;
        setStatus(next);
        if (!TERMINAL.has(next.status)) timeout = setTimeout(poll, 4_000);
      } catch (error) {
        if (cancelled) return;
        consecutiveFailures += 1;
        if (consecutiveFailures < 3) timeout = setTimeout(poll, 4_000);
        else onError(error instanceof Error ? error.message : "Experiment status lookup failed.");
      }
    }

    void poll();
    return () => {
      cancelled = true;
      if (timeout) clearTimeout(timeout);
    };
  }, [dispatch.jobId, onError, operatorToken]);

  async function downloadArtifact(artifactName: string) {
    setDownloading(artifactName);
    try {
      const response = await fetch(`/api/experiments/${dispatch.jobId}/artifacts/${encodeURIComponent(artifactName)}`, {
        cache: "no-store",
        headers: { Authorization: `Bearer ${operatorToken}` },
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.error || "Artifact download failed.");
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
        <div><span>JOB ACCEPTED</span><strong id="job-monitor-title">{dispatch.jobId}</strong></div>
        <small>{(status?.executionBackend || dispatch.executionBackend || "external").toUpperCase()} COMPUTE</small>
      </div>

      <div className="job-stage" aria-live="polite">
        <span className={TERMINAL.has(current) ? current === "COMPLETED_ENGINEERING" ? "complete" : "failed" : "active"} />
        <div><b>{readableStatus(current)}</b><small>{status?.updatedAt ? `updated ${new Date(status.updatedAt).toLocaleTimeString()}` : "waiting for first runner receipt"}</small></div>
      </div>

      <div className="job-progress" aria-label={`Experiment progress: ${readableStatus(current)}`}>
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
          <div><span>DETECTION DELAY</span><strong>{metrics.mean_detection_delay_frames === null ? "N/A" : `${metrics.mean_detection_delay_frames.toFixed(1)} fr`}</strong></div>
          <p>{metrics.evaluation_rows_scored.toLocaleString()} frozen predictions scored · TP {metrics.confusion.tp} · FP {metrics.confusion.fp} · FN {metrics.confusion.fn} · TN {metrics.confusion.tn}</p>
        </div>
      ) : null}

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
