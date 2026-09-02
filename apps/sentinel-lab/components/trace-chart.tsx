"use client";

import { useMemo, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent } from "react";
import type { SmokeResult, TracePoint } from "@/lib/sentinel/types";

const WIDTH = 960;
const HEIGHT = 338;
const PAD = { left: 48, right: 18, top: 22, bottom: 35 };
const PLOT_WIDTH = WIDTH - PAD.left - PAD.right;
const PLOT_HEIGHT = HEIGHT - PAD.top - PAD.bottom;

function xFor(frame: number, frames: number) {
  return PAD.left + (frame / Math.max(1, frames - 1)) * PLOT_WIDTH;
}

function yFor(value: number) {
  return PAD.top + (1 - value) * PLOT_HEIGHT;
}

function linePath(trace: TracePoint[], key: "raw" | "quotient" | "persistence") {
  return trace.map((point, index) => `${index === 0 ? "M" : "L"}${xFor(point.frame, trace.length).toFixed(2)},${yFor(point[key]).toFixed(2)}`).join(" ");
}

export function TraceChart({ run }: { run: SmokeResult }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [cursor, setCursor] = useState(() => Math.min(run.trace.length - 1, Math.round(run.trace.length * 0.58)));
  const paths = useMemo(() => ({ raw: linePath(run.trace, "raw"), quotient: linePath(run.trace, "quotient"), persistence: linePath(run.trace, "persistence") }), [run.trace]);
  const point = run.trace[Math.min(cursor, run.trace.length - 1)];

  function updatePointer(event: PointerEvent<SVGSVGElement>) {
    const bounds = svgRef.current?.getBoundingClientRect();
    if (!bounds) return;
    const viewX = ((event.clientX - bounds.left) / bounds.width) * WIDTH;
    const frame = Math.round(((viewX - PAD.left) / PLOT_WIDTH) * (run.trace.length - 1));
    setCursor(Math.max(0, Math.min(run.trace.length - 1, frame)));
  }

  function handleKeyboard(event: KeyboardEvent<HTMLDivElement>) {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    if (event.key === "Home") setCursor(0);
    if (event.key === "End") setCursor(run.trace.length - 1);
    if (event.key === "ArrowLeft") setCursor((value) => Math.max(0, value - 1));
    if (event.key === "ArrowRight") setCursor((value) => Math.min(run.trace.length - 1, value + 1));
  }

  return (
    <div className="trace-shell" tabIndex={0} onKeyDown={handleKeyboard} aria-label={`Causal observer trace. Frame ${point.frame}, raw ${point.raw}, quotient ${point.quotient}, persistence ${point.persistence}. Use arrow keys to inspect.`}>
      <div className="trace-readout" aria-hidden="true">
        <span>F/{String(point.frame).padStart(3, "0")}</span>
        <span><i className="legend-dot raw" />RAW {point.raw.toFixed(3)}</span>
        <span><i className="legend-dot quotient" />QUOT {point.quotient.toFixed(3)}</span>
        <span><i className="legend-dot persistence" />PERSIST {point.persistence.toFixed(3)}</span>
      </div>
      <svg ref={svgRef} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="trace-svg" role="img" aria-label="Interactive line chart of causal observer measurements" onPointerMove={updatePointer} onPointerDown={updatePointer}>
        <title>Causal observer engineering-smoke trace</title>
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line x1={PAD.left} y1={yFor(tick)} x2={WIDTH - PAD.right} y2={yFor(tick)} className="grid-line" />
            <text x={PAD.left - 11} y={yFor(tick) + 4} textAnchor="end" className="axis-label">{tick.toFixed(2)}</text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
          const frame = Math.round((run.trace.length - 1) * tick);
          return <text key={tick} x={xFor(frame, run.trace.length)} y={HEIGHT - 9} textAnchor={tick === 0 ? "start" : tick === 1 ? "end" : "middle"} className="axis-label">{frame}</text>;
        })}
        {run.eventWindows.map((window) => (
          <g key={`${window.start}-${window.end}`}>
            <rect x={xFor(window.start, run.trace.length)} y={PAD.top} width={Math.max(2, xFor(window.end, run.trace.length) - xFor(window.start, run.trace.length))} height={PLOT_HEIGHT} className="event-window" />
            <text x={xFor(window.start, run.trace.length) + 7} y={PAD.top + 15} className="event-label">{window.kind.toUpperCase()}</text>
          </g>
        ))}
        <line x1={PAD.left} y1={yFor(run.summary.threshold)} x2={WIDTH - PAD.right} y2={yFor(run.summary.threshold)} className="threshold-line" />
        <path d={paths.raw} className="trace-line trace-raw" />
        <path d={paths.quotient} className="trace-line trace-quotient" />
        <path d={paths.persistence} className="trace-line trace-persistence" />
        <line x1={xFor(point.frame, run.trace.length)} y1={PAD.top} x2={xFor(point.frame, run.trace.length)} y2={PAD.top + PLOT_HEIGHT} className="cursor-line" />
        <circle cx={xFor(point.frame, run.trace.length)} cy={yFor(point.raw)} r="3.5" className="cursor-point cursor-raw" />
        <circle cx={xFor(point.frame, run.trace.length)} cy={yFor(point.quotient)} r="3.5" className="cursor-point cursor-quotient" />
      </svg>
      <div className="trace-footer"><span>PAST-ONLY · 64D SYNTHETIC CARRIER</span><span className="trace-legend"><i className="dash" /> THRESHOLD {run.summary.threshold.toFixed(2)}</span></div>
    </div>
  );
}
