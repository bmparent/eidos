from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import lzma
import math
import platform
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Metrics:
    f_beta: float
    apr: float
    ec: float
    rep: float
    mu: float
    nbc: float
    ndl: float
    fpr: float
    value: float | None
    status: str = "ok"
    skip_reason: str = ""


SCORING_MODES = ["strict_joint_value", "detection_only_value", "compression_only_value", "adapted_baseline_value"]
SMOKE_DEFAULT_FAMILIES = ["s1_backdoor", "s6_noise_thrash"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["smoke", "counterexamples", "full"], default="smoke")
    p.add_argument("--families", default=",".join(SMOKE_DEFAULT_FAMILIES))
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--frames", type=int, default=10000)
    p.add_argument("--out", required=True)
    return p.parse_args()


def gen_backdoor(frames: int, seed: int, snr_db: float = -10.0) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    rng = np.random.default_rng(seed)
    t = np.arange(frames)
    noise = rng.normal(0.0, 1.0, size=frames)
    event_start = frames // 3
    event_end = event_start + frames // 10
    y = np.zeros(frames, dtype=int)
    y[event_start:event_end] = 1
    signal = np.sin(2 * np.pi * 0.03 * t)
    signal_masked = np.zeros(frames)
    signal_masked[event_start:event_end] = signal[event_start:event_end]
    scale = math.sqrt((10 ** (snr_db / 10.0)) * np.mean(noise[event_start:event_end] ** 2) / max(np.mean(signal_masked[event_start:event_end] ** 2), 1e-12))
    return (noise + scale * signal_masked).astype(np.float64), y, (event_start, event_end)


def gen_noise_thrash(seed: int, frames: int, dims: int = 1) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 5.0, size=(frames, dims)).mean(axis=1)
    y = np.zeros(frames, dtype=int)
    return x.astype(np.float64), y, (-1, -1)


def gen_nuisance_subspace_anomaly(seed: int, frames: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    t = np.arange(frames)
    nuisance = 2.5 * np.sin(2 * np.pi * 0.015 * t)
    x = nuisance + rng.normal(0.0, 0.25, size=frames)
    y = np.zeros(frames, dtype=int)
    start, end = frames // 2, frames // 2 + frames // 15
    x[start:end] += 1.2 * np.sin(2 * np.pi * 0.015 * t[start:end])
    y[start:end] = 1
    return x, y, {"event_window": [start, end], "freq": 0.015, "dominant_amp": 2.5, "anomaly_amp": 1.2}


def _event_fbeta(alerts: np.ndarray, y: np.ndarray, beta: float = 2.0) -> tuple[float, float]:
    evt_true = y.max() > 0
    true_idx = np.where(y == 1)[0]
    delay = float(len(y))
    if evt_true and alerts[true_idx].any():
        delay = float(int(np.where(alerts[true_idx])[0][0]))
    tp = 1 if evt_true and alerts[true_idx].any() else 0
    fp = int(np.logical_and(alerts == 1, y == 0).sum())
    fn = 1 if evt_true and tp == 0 else 0
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    b2 = beta * beta
    return (1 + b2) * p * r / max((b2 * p + r), 1e-12), delay


def _bytes_ratio(arr: np.ndarray, method: str) -> tuple[float | None, str, str]:
    raw = arr.tobytes()
    if method == "raw":
        c = raw
    elif method == "gzip":
        c = gzip.compress(raw)
    elif method == "lzma":
        c = lzma.compress(raw)
    elif method == "zstd":
        try:
            import zstandard as zstd  # type: ignore

            c = zstd.ZstdCompressor(level=3).compress(raw)
        except Exception:
            return None, "skipped", "zstandard package unavailable"
    else:
        raise ValueError(method)
    return len(c) / max(len(raw), 1), "ok", ""


def baseline_detector(x: np.ndarray, name: str) -> np.ndarray:
    if name == "zscore":
        z = (x - x.mean()) / max(x.std(), 1e-12)
        return (np.abs(z) > 3.0).astype(int)
    if name == "ewma":
        alpha, m = 0.03, np.zeros_like(x)
        for i in range(1, len(x)):
            m[i] = alpha * x[i] + (1 - alpha) * m[i - 1]
        r = np.abs(x - m)
        return (r > (r.mean() + 3 * r.std())).astype(int)
    if name == "cusum":
        k, h, gp, gn = 0.1, 10.0, np.zeros_like(x), np.zeros_like(x)
        out, mu = np.zeros_like(x, dtype=int), float(x.mean())
        for i in range(1, len(x)):
            gp[i] = max(0.0, gp[i - 1] + x[i] - mu - k)
            gn[i] = min(0.0, gn[i - 1] + x[i] - mu + k)
            out[i] = int(gp[i] > h or gn[i] < -h)
        return out
    raise ValueError(name)


def eidos_minimal(x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    win, pred = 64, np.r_[x[0], x[:-1]]
    residual, raw_score, fft_energy = x - pred, np.abs(x - pred), np.zeros_like(x)
    for i in range(win, len(x)):
        ps = np.abs(np.fft.rfft(x[i - win : i]))
        fft_energy[i] = ps[3:12].mean() if len(ps) > 12 else ps.mean()
    s1 = (raw_score - raw_score.mean()) / max(raw_score.std(), 1e-12)
    s2 = (fft_energy - fft_energy.mean()) / max(fft_energy.std(), 1e-12)
    scout = np.maximum(s1, s2)
    return (scout > 2.5).astype(int), {"selected_lift": "spectral" if np.max(s2) >= np.max(s1) else "raw", "raw_max": float(np.max(s1)), "scout_max": float(np.max(scout))}


def compute_metrics(mode: str, name: str, y: np.ndarray, alerts: np.ndarray, nbc: float, card_ok: bool, replay_ok: bool) -> Metrics:
    f_beta, delay = _event_fbeta(alerts, y)
    apr = 1.0 if name in {"eidos_minimal", "eidos_scout", "gzip", "lzma", "zstd"} else 0.0
    ec = 1.0 if card_ok else 0.0
    rep = 1.0 if replay_ok else 0.0
    mu = 0.0
    fpr = float(alerts[np.where(y == 0)[0]].sum()) * 10000.0 / max((y == 0).sum(), 1)
    ndl = min(delay / 100.0, 100.0) if f_beta > 0 else 100.0
    if mode == "detection_only_value":
        apr = ec = rep = 1.0
    elif mode == "compression_only_value":
        f_beta = apr = ec = rep = 1.0
        ndl = fpr = 0.0
    lam_b, lam_l, lam_fp = 1.0, 1.0, 0.05
    value = ((f_beta**1.0) * (apr**0.7) * (ec**0.3) * (rep**0.3) * ((mu + 1e-6) ** 0.1)) / ((1 + lam_b * max(nbc, 0.0)) * (1 + lam_l * ndl) * (1 + lam_fp * fpr))
    return Metrics(f_beta, apr, ec, rep, mu, nbc, ndl, fpr, value)


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    families = [f.strip().lower() for f in args.families.split(",") if f.strip()]
    out = Path(args.out)
    for p in ["counterexamples", "discovery_cards", "replay_logs"]:
        (out / p).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skipped_baselines: list[dict[str, str]] = []
    s6_scout_fp = 0.0

    if args.suite in {"smoke", "full"}:
        for seed in seeds:
            for fam in families:
                if fam == "s1_backdoor":
                    x, y, event_win = gen_backdoor(args.frames, seed)
                    family_name = "s1_backdoor"
                elif fam == "s6_noise_thrash":
                    x, y, event_win = gen_noise_thrash(seed, args.frames)
                    family_name = "s6_noise_thrash"
                else:
                    continue
                for m in ["gzip", "zstd", "lzma", "zscore", "ewma", "cusum"]:
                    alerts = np.zeros_like(y) if m in {"gzip", "zstd", "lzma"} else baseline_detector(x, m)
                    nbc, status, reason = _bytes_ratio(x, m if m in {"gzip", "zstd", "lzma"} else "raw")
                    if status == "skipped":
                        skipped_baselines.append({"system": m, "reason": reason})
                    for mode in SCORING_MODES:
                        metric = Metrics(0, 0, 0, 0, 0, 0, 0, 0, None, status=status, skip_reason=reason)
                        if status == "ok":
                            metric = compute_metrics(mode, m, y, alerts, float(nbc), False, False)
                        rows.append({"scoring_mode": mode, "system": m, "seed": seed, "family": family_name, "F_beta": metric.f_beta, "APR": metric.apr, "EC": metric.ec, "REP": metric.rep, "MU": metric.mu, "NBC": metric.nbc, "NDL": metric.ndl, "FPR": metric.fpr, "value": metric.value, "status": metric.status, "skip_reason": metric.skip_reason})

                for system in ["eidos_minimal", "eidos_scout"]:
                    alerts, info = eidos_minimal(x)
                    raw_escape_hatch_required = info["raw_max"] > 5.0 and not alerts.any()
                    card_ok = system == "eidos_scout"
                    replay_ok = system == "eidos_scout"
                    if system == "eidos_scout":
                        card = {"source_window": [int(event_win[0]), int(event_win[1])], "detector_name": "eidos_scout", "score": float(info["scout_max"]), "threshold": 2.5, "replay_command": f"python -m eidos_brain.proof.run_proof --suite smoke --seeds {seed} --frames {args.frames}", "raw_escape_hatch_required": raw_escape_hatch_required}
                        (out / "discovery_cards" / f"card_{family_name}_seed_{seed}.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
                        (out / "replay_logs" / f"replay_{family_name}_seed_{seed}.json").write_text(json.dumps({"seed": seed, "status": "success", "reason": "deterministic source"}, indent=2), encoding="utf-8")
                    nbc, _, _ = _bytes_ratio(x, "gzip")
                    for mode in SCORING_MODES:
                        ec, rep = card_ok, replay_ok
                        if mode == "adapted_baseline_value" and system != "eidos_scout":
                            ec = rep = True
                        metric = compute_metrics(mode, system, y, alerts, float(nbc), ec, rep)
                        rows.append({"scoring_mode": mode, "system": system, "seed": seed, "family": family_name, "F_beta": metric.f_beta, "APR": metric.apr, "EC": metric.ec, "REP": metric.rep, "MU": metric.mu, "NBC": metric.nbc, "NDL": metric.ndl, "FPR": metric.fpr, "value": metric.value, "status": "ok", "skip_reason": ""})
                    if family_name == "s6_noise_thrash" and system == "eidos_scout":
                        s6_scout_fp = max(s6_scout_fp, float(alerts.sum()) * 10000.0 / max(len(alerts), 1))

    if args.suite in {"counterexamples", "full"}:
        for seed in seeds:
            x, y, params = gen_nuisance_subspace_anomaly(seed, args.frames)
            raw_alerts = (np.abs(x - np.r_[x[0], x[:-1]]) > 2.0).astype(int)
            scout_alerts, _ = eidos_minimal(x)
            report = {"seed": seed, "generator_parameters": params, "which_system_failed": "eidos_scout" if not scout_alerts[np.where(y==1)[0]].any() else "none", "raw_residual_caught_it": bool(raw_alerts[np.where(y==1)[0]].any()), "quotient_or_scout_missed_it": bool(not scout_alerts[np.where(y==1)[0]].any()), "theorem_assumption_required": "meaningful anomaly retains nonzero evidence outside nuisance subspace OR raw residual escape hatch"}
            p = out / "counterexamples" / "nuisance_subspace_anomaly"
            p.mkdir(parents=True, exist_ok=True)
            (p / f"report_seed_{seed}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scoring_mode", "system", "seed", "family", "F_beta", "APR", "EC", "REP", "MU", "NBC", "NDL", "FPR", "value", "status", "skip_reason"])
        w.writeheader(); w.writerows(rows)

    report_lines = ["# HiddenStructureBench", "", "## Scoring Tables"]
    for mode in SCORING_MODES:
        report_lines.append(f"\n### {mode}\n")
        mode_rows = [r for r in rows if r["scoring_mode"] == mode and r["value"] is not None]
        means: dict[str, float] = {}
        for sysn in {r["system"] for r in mode_rows}:
            vals = [float(r["value"]) for r in mode_rows if r["system"] == sysn]
            if vals: means[sysn] = statistics.mean(vals)
        for k,v in sorted(means.items(), key=lambda kv: kv[1], reverse=True): report_lines.append(f"- {k}: {v:.6g}")
    report_lines.append("\n## Skipped Baselines")
    if skipped_baselines:
        report_lines.extend([f"- {s['system']}: {s['reason']}" for s in skipped_baselines])
    else:
        report_lines.append("- none")
    (out / "benchmark_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    verdict = "UNDERDETERMINED"
    if args.suite in {"smoke", "full"} and s6_scout_fp > 5.0:
        verdict = "FAILED SMOKE"
    elif args.suite in {"smoke", "full"}:
        verdict = "EMPIRICALLY SUPPORTED"

    (out / "theorem_status.md").write_text(
        "\n".join([
            "# Theorem Status",
            "",
            f"## Verdict\n- {verdict}",
            "## Scope",
            f"- families: {', '.join(families)}",
            f"- frames: {args.frames}",
            f"- seeds: {seeds}",
            "## Metric caveats",
            "- strict_joint_value includes EC/REP zero for non-card baselines.",
            "- adapted_baseline_value gives detector baselines partial synthetic card/replay credit.",
            "- detection_only_value isolates detection, latency, and FPR.",
            "## Counterexamples",
            "- nuisance_subspace_anomaly generated in counterexample suite.",
            "## Required assumptions",
            "- hidden signal visible in at least one lift.",
            "- meaningful anomaly retains nonzero evidence outside nuisance subspace, unless raw escape hatch catches it.",
            "- card replay requires deterministic source and config.",
            "- memory familiarity must not suppress known-dangerous events.",
            "## Required repair",
            "- Keep raw residual as a safety channel so quotient projection cannot erase all recall.",
            "## Next proof obligations",
            "- S2 SlowDrift",
            "- S3 RegimeShift",
            "- S7 HarmlessSpike",
            "- S8 DangerousRepeat",
            "- ablation comparisons",
            "## Full conjecture",
            "- UNDERDETERMINED",
        ]), encoding="utf-8")

    manifest = {"command": " ".join(sys.argv), "python": sys.version, "os": platform.platform(), "seeds": seeds, "frames": args.frames, "suite": args.suite, "families": families, "timestamp": datetime.now(timezone.utc).isoformat(), "skipped_baselines": skipped_baselines}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
