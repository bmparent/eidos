"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { EngineDiagnostics } from "@/lib/experiments/types";
import { downloadJson } from "@/lib/experiments/client";
import reference from "@/lib/experiments/reference-run.json";

function number(value: unknown) { return typeof value === "number" && Number.isFinite(value) ? value : null; }
function format(value: unknown) { const n = number(value); return n === null ? "—" : n.toFixed(3); }

export function EngineObservatory({ diagnostics, jobId }: { diagnostics?: EngineDiagnostics; jobId?: string }) {
  const data = diagnostics || reference.diagnostics as EngineDiagnostics;
  const sourceKey = diagnostics ? jobId || diagnostics.code_sha256 : "reference";
  return <ObservatoryView key={sourceKey} data={data} isReference={!diagnostics} jobId={jobId} />;
}

function ObservatoryView({ data, isReference, jobId }: { data: EngineDiagnostics; isReference: boolean; jobId?: string }) {
  const id = useId().replaceAll(":", "");
  const points = data.geometry?.points || [];
  const [position, setPosition] = useState(Math.floor(points.length * 0.65));
  const [yaw, setYaw] = useState(-0.55);
  const [playing, setPlaying] = useState(false);
  const [motion, setMotion] = useState(false);
  const [visible, setVisible] = useState(true);
  const shell = useRef<HTMLElement>(null);
  const drag = useRef<{ x: number; yaw: number } | null>(null);

  useEffect(() => {
    const preference = window.matchMedia("(prefers-reduced-motion: reduce)");
    setMotion(!preference.matches);
    const change = () => { setMotion(!preference.matches); if (preference.matches) setPlaying(false); };
    preference.addEventListener("change", change);
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting));
    if (shell.current) observer.observe(shell.current);
    return () => { preference.removeEventListener("change", change); observer.disconnect(); };
  }, []);

  useEffect(() => {
    if (!visible || (!motion && !playing)) return;
    const timer = setInterval(() => {
      if (document.hidden || drag.current) return;
      if (motion) setYaw((value) => value + 0.003);
      if (playing) setPosition((value) => (value + 1) % Math.max(points.length, 1));
    }, 80);
    return () => clearInterval(timer);
  }, [visible, playing, motion, points.length]);

  const projected = useMemo(() => {
    const extent = Math.max(0.001, ...points.flatMap((p) => [Math.abs(p.x), Math.abs(p.y), Math.abs(p.z)]));
    return points.map((point) => {
      const x = point.x / extent * 190;
      const y = point.y / extent * 190;
      const z = point.z / extent * 190;
      const rx = x * Math.cos(yaw) + z * Math.sin(yaw);
      const rz = -x * Math.sin(yaw) + z * Math.cos(yaw);
      const ry = y * 0.88 - rz * 0.35;
      const depth = y * 0.35 + rz * 0.88;
      const perspective = 650 / (650 + depth);
      return { x: 340 + rx * perspective, y: 190 - ry * perspective, depth, step: point.step };
    });
  }, [points, yaw]);
  const selected = projected[Math.min(position, projected.length - 1)];
  const row = [...data.trace].reverse().find((item) => Number(item.step) <= (selected?.step ?? 0)) || data.trace[0] || {};
  const traceIndex = data.trace.indexOf(row);
  const channels = [
    { key: "z", label: "Surprise", symbol: "z", color: "#71f0d1", help: "Distance from the engine's residual baseline, in robust scale units." },
    { key: "hipp_chi", label: "Memory gate", symbol: "χ", color: "#b59aff", help: "Recorded familiarity gate: how strongly recognition inhibits learning." },
    { key: "thermo_temp", label: "Regulation", symbol: "T", color: "#ffbd77", help: "Internal noise-control parameter; this is not a physical temperature." },
  ];

  return <section ref={shell} className="observatory" aria-labelledby={`${id}-title`}>
    <div className="observatory-heading"><div><p className="eyebrow">ENGINE OBSERVATORY</p><h3 id={`${id}-title`}>A trajectory of memory.</h3></div><span className="observation-source">{isReference ? "RECORDED EXAMPLE" : "YOUR COMPLETED RUN"}</span></div>
    <p className="observatory-caption">{isReference ? "Actual Torch engine · synthetic input · standard profile · seed 0. This example is separate from your experiment settings." : `Measured reservoir activity from ${jobId}.`}</p>
    <div className="reservoir-stage">
      <svg viewBox="0 0 680 380" role="img" aria-label="Rotatable three-dimensional projection of sampled reservoir states. Use the frame and viewing-angle sliders to explore." onPointerDown={(event) => { drag.current = { x: event.clientX, yaw }; event.currentTarget.setPointerCapture(event.pointerId); setMotion(false); }} onPointerMove={(event) => { if (drag.current) setYaw(drag.current.yaw + (event.clientX - drag.current.x) / 180); }} onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}>
        <defs><linearGradient id={`${id}-color`}><stop offset="0%" stopColor="#65eecb" /><stop offset="45%" stopColor="#818bff" /><stop offset="80%" stopColor="#d6a0ff" /><stop offset="100%" stopColor="#ffbe78" /></linearGradient><filter id={`${id}-glow`} x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="4" /></filter></defs>
        {[65, 115, 165].map((radius) => <ellipse key={radius} cx="340" cy="240" rx={radius * 1.65} ry={radius * 0.4} fill="none" stroke="#233440" strokeWidth="0.7" />)}
        <line x1="75" y1="240" x2="605" y2="240" stroke="#233440" strokeWidth="0.7" />
        <line x1="340" y1="34" x2="340" y2="317" stroke="#233440" strokeWidth="0.7" />
        <polyline points={projected.map((p) => `${p.x},${p.y}`).join(" ")} fill="none" stroke={`url(#${id}-color)`} opacity="0.5" strokeWidth="1" />
        <polyline points={projected.slice(Math.max(0, position - 45), position + 1).map((p) => `${p.x},${p.y}`).join(" ")} fill="none" stroke={`url(#${id}-color)`} strokeWidth="8" opacity="0.45" filter={`url(#${id}-glow)`} />
        <polyline points={projected.slice(Math.max(0, position - 45), position + 1).map((p) => `${p.x},${p.y}`).join(" ")} fill="none" stroke="#b2fdeb" strokeWidth="1.8" />
        {projected.map((p, i) => <circle key={i} cx={p.x} cy={p.y} r={i === position ? 3.8 : Math.max(0.7, 1.7 - p.depth / 300)} fill={`hsl(${160 + i / Math.max(1, projected.length - 1) * 120} 75% 75%)`} opacity={i <= position ? 0.85 : 0.35} />)}
        {selected && <g><line x1={selected.x} y1={selected.y} x2={selected.x} y2="329" stroke="#91f4da" strokeDasharray="3 5" opacity="0.3" /><circle cx={selected.x} cy={selected.y} r="11" fill="#9bffe7" opacity="0.18" /><circle cx={selected.x} cy={selected.y} r="4" fill="#eafff7" /><text x="20" y="355" fill="#a3b9c6" fontSize="12">STATE / {selected.step}</text></g>}
        <text x="660" y="355" textAnchor="end" fill="#a3b9c6" fontSize="12">{points.length} SAMPLED STATES · PCA / 3D</text>
      </svg>
      <div className="observatory-controls"><button onClick={() => setPlaying((value) => !value)} disabled={!points.length}>{playing ? "Pause replay" : "Replay trajectory"}</button><button aria-pressed={motion} onClick={() => setMotion((value) => !value)}>{motion ? "Stop rotation" : "Rotate view"}</button></div>
    </div>
    <div className="trajectory-slider"><label htmlFor={`${id}-frame`}>Inspect frame <b>{selected?.step ?? "—"}</b></label><input id={`${id}-frame`} type="range" min={0} max={Math.max(0, points.length - 1)} value={position} onChange={(event) => { setPlaying(false); setPosition(Number(event.target.value)); }} /><label className="angle-slider">Viewing angle<input type="range" min={-3.15} max={3.15} step={0.01} value={Math.atan2(Math.sin(yaw), Math.cos(yaw))} onChange={(event) => { setMotion(false); setYaw(Number(event.target.value)); }} /></label></div>
    <div className="signal-channels">{channels.map((channel) => {
      const values = data.trace.map((item) => number(item[channel.key]));
      const maximum = Math.max(0.001, ...values.map((value) => value ?? 0));
      const path = values.reduce((acc, value, i) => value === null ? acc : `${acc} ${i === 0 || values[i - 1] === null ? "M" : "L"}${i / Math.max(1, values.length - 1) * 220},${47 - value / maximum * 36}`, "");
      return <div className="signal-channel" key={channel.key} style={{ "--signal-color": channel.color } as React.CSSProperties}><div><span>{channel.label} <i>{channel.symbol}</i></span><b>{format(row[channel.key])}</b></div><svg viewBox="0 0 220 52" role="img" aria-label={`${channel.label} recorded trace, scaled independently; maximum ${maximum.toFixed(3)}`}><path d={path} fill="none" stroke={channel.color} strokeWidth="1.3" /><line x1={Math.max(0, traceIndex) / Math.max(1, values.length - 1) * 220} x2={Math.max(0, traceIndex) / Math.max(1, values.length - 1) * 220} y1="3" y2="49" stroke="#cedce6" opacity="0.5" /></svg><p>{channel.help}</p></div>;
    })}</div>
    <div className="observatory-footnote"><p>Three principal components preserve {((data.geometry?.variance_explained ?? 0) * 100).toFixed(1)}% of sampled state variance. Point color moves from mint to violet with sample order. Readouts use recorded frame {String(row.step ?? "—")}; channel scales differ. This projection supports inspection, not proof of detection quality.</p><button onClick={() => downloadJson(isReference ? reference : data, isReference ? "eidos-recorded-reference.json" : "eidos-engine-diagnostics.json")}>Inspect the source data ↗</button></div>
  </section>;
}
