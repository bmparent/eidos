"""Build Sentinel calibration v1 recall-diagnostics receipts.

This is a reporting-only aggregator. It reads labeled proof-run artifacts and
does not change reservoir dynamics, RLS behavior, raw Sentinel thresholds,
anomaly policy, compression behavior, hippocampus memory, incident-card
generation, or domain adapter math.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tools import run_proof_baseline as proof_helpers


RUN_DATE = "2026-06-07"
RUN_ID = "calibration_recall_diagnostics"
CURRENT_BALANCED_RECALL_BASELINE = 0.166667
NEAR_ATTACK_WINDOW_FRAMES = 25


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value in (None, ""):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def parse_float(value: Any) -> Optional[float]:
    if value in (None, "", "NA"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=2, action="append", metavar=("CASE", "PATH"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-recall", type=float, default=CURRENT_BALANCED_RECALL_BASELINE)
    return parser.parse_args(argv)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_block(metrics: Dict[str, Any], view: str) -> Dict[str, Any]:
    event_view = metrics.get("event_view_metrics", {}).get(view) or {}
    if view == "calibrated":
        event_view = metrics.get("calibrated_event_metrics") or {}
    return {
        "event_count": event_view.get("event_count"),
        "precision": event_view.get("precision"),
        "recall": event_view.get("recall"),
        "f1": event_view.get("f1"),
        "fp_per_10k": event_view.get("false_positives_per_10k_frames"),
    }


def load_run(case: str, path: Path) -> Dict[str, Any]:
    metrics = load_json(path / "labeled_metrics.json", {})
    crash = load_json(path / "crash_scan.json", {})
    drive = load_json(path / "drive_manifest.json", {})
    manifest = load_json(path / "run_manifest.json", {})
    funnel = load_json(path / "candidate_funnel_report.json", {})
    profile_rows = read_csv(path / "confirmation_profile_sweep.csv")
    sample = metrics.get("sample_receipt") or manifest.get("sample_receipt") or {}
    return {
        "case": case,
        "path": path,
        "metrics": metrics,
        "crash": crash,
        "drive": drive,
        "manifest": manifest,
        "funnel": funnel,
        "profiles": profile_rows,
        "sample": sample,
        "completed": bool(metrics),
    }


def aggregate_candidate_funnel(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    cases = []
    for run in runs:
        funnel = dict(run.get("funnel") or {})
        if not funnel:
            continue
        funnel["case"] = run["case"]
        cases.append(funnel)
    stage_totals: Dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        for stage in case.get("stages", []):
            name = str(stage.get("stage"))
            for key in ("event_count", "true_positive_count", "false_positive_count", "dropped_event_count"):
                value = stage.get(key)
                if value is not None:
                    stage_totals[name][key] += int(value)
    return {
        "generated_at_utc": utc_now(),
        "cases": cases,
        "stage_totals": {stage: dict(values) for stage, values in sorted(stage_totals.items())},
        "policy_note": "Candidate funnel is diagnostic-only; labels are used only for reporting and scoring.",
    }


def write_candidate_funnel_md(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# Candidate Funnel Report",
        "",
        "| Case | Stage | Events | TP | FP | Precision | Recall | F1 | FP/10k | Coverage | First Latency | Dropped | Reasons |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in report.get("cases", []):
        case_name = case.get("case")
        for stage in case.get("stages", []):
            lines.append(
                f"| {case_name} | {stage.get('stage')} | {fmt(stage.get('event_count'))} | {fmt(stage.get('true_positive_count'))} | {fmt(stage.get('false_positive_count'))} | {fmt(stage.get('precision'))} | {fmt(stage.get('recall'))} | {fmt(stage.get('f1'))} | {fmt(stage.get('fp_per_10k'))} | {fmt(stage.get('attack_window_coverage'))} | {fmt(stage.get('first_detection_latency'))} | {fmt(stage.get('dropped_event_count'))} | `{stage.get('dropped_event_reason_counts', {})}` |"
            )
    lines.extend(["", "## Drop Examples", ""])
    for case in report.get("cases", []):
        details = case.get("drop_reason_accounting", [])
        if not details:
            continue
        lines.append(f"### {case.get('case')}")
        for item in details[:12]:
            lines.append(
                f"- `{item.get('stage')}` `{item.get('event_id')}` frames `{item.get('start_frame')}`-`{item.get('end_frame')}` reasons `{','.join(item.get('rejected_reasons', []))}` distance `{fmt(item.get('nearest_attack_window_distance'))}` overlap `{item.get('overlaps_attack_window')}`"
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def profile_sweep_rows(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        metrics = run["metrics"]
        raw = metric_block(metrics, "raw")
        for row in run.get("profiles", []):
            out = {
                "case": run["case"],
                "sample_mode": metrics.get("sample_mode"),
                "frames_processed": metrics.get("frames_processed"),
                "profile": row.get("profile"),
                "raw_precision": raw.get("precision"),
                "raw_recall": raw.get("recall"),
                "raw_f1": raw.get("f1"),
                "raw_fp_per_10k": raw.get("fp_per_10k"),
                **row,
            }
            rows.append(out)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_profile_sweep_md(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# Confirmation Profile Sweep",
        "",
        "| Case | Mode | Profile | Raw Recall | Cal Recall | Cal Precision | Cal F1 | Cal FP/10k | Coverage | Latency | Confirmed | Suppressed | Crash |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('case')} | {row.get('sample_mode')} | {row.get('profile')} | {fmt(row.get('raw_recall'))} | {fmt(row.get('calibrated_recall'))} | {fmt(row.get('calibrated_precision'))} | {fmt(row.get('calibrated_f1'))} | {fmt(row.get('calibrated_fp_per_10k'))} | {fmt(row.get('calibrated_coverage'))} | {fmt(row.get('calibrated_first_detection_latency'))} | {fmt(row.get('calibrated_confirmed_count'))} | {fmt(row.get('calibration_suppressed_count'))} | {fmt(row.get('crash_hit_count'))} |"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def sample_manifests(runs: Sequence[Dict[str, Any]], sample_mode: str) -> Dict[str, Any]:
    receipts = []
    for run in runs:
        sample = run.get("sample") or {}
        if sample.get("mode") == sample_mode:
            receipts.append({"case": run["case"], "run_path": str(run["path"]), "sample_receipt": sample})
    return {"generated_at_utc": utc_now(), "sample_mode": sample_mode, "runs": receipts}


def recall_protection_audit(funnel_report: Dict[str, Any]) -> Dict[str, Any]:
    events = []
    for case in funnel_report.get("cases", []):
        for item in case.get("drop_reason_accounting", []):
            if item.get("stage") not in {"deduped_to_confirmed", "confirmed_to_calibrated"}:
                continue
            distance = parse_float(item.get("nearest_attack_window_distance"))
            inside = bool(item.get("overlaps_attack_window"))
            near = distance is not None and abs(distance) <= NEAR_ATTACK_WINDOW_FRAMES
            events.append(
                {
                    **item,
                    "case": case.get("case"),
                    "near_attack_window_frames": NEAR_ATTACK_WINDOW_FRAMES,
                    "attack_window_position": "inside" if inside else ("near" if near else item.get("nearest_attack_window_direction")),
                    "may_hide_true_attack_context_event": bool(inside or near),
                }
            )
    return {
        "generated_at_utc": utc_now(),
        "near_attack_window_frames": NEAR_ATTACK_WINDOW_FRAMES,
        "summary": {
            "suppressed_event_count": len(events),
            "may_hide_true_attack_context_event_count": sum(1 for item in events if item["may_hide_true_attack_context_event"]),
            "position_counts": dict(Counter(str(item.get("attack_window_position")) for item in events)),
        },
        "events": events,
    }


def choose_best_profile(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("profile"))].append(row)
    scored = []
    for profile, items in grouped.items():
        recall_values = [parse_float(item.get("calibrated_recall")) for item in items if parse_float(item.get("calibrated_recall")) is not None]
        fp_values = [parse_float(item.get("calibrated_fp_per_10k")) for item in items if parse_float(item.get("calibrated_fp_per_10k")) is not None]
        coverage_values = [parse_float(item.get("calibrated_coverage")) for item in items if parse_float(item.get("calibrated_coverage")) is not None]
        scored.append(
            {
                "profile": profile,
                "mean_calibrated_recall": sum(recall_values) / len(recall_values) if recall_values else None,
                "max_calibrated_fp_per_10k": max(fp_values) if fp_values else None,
                "min_calibrated_coverage": min(coverage_values) if coverage_values else None,
                "run_count": len(items),
            }
        )
    scored.sort(
        key=lambda item: (
            parse_float(item.get("mean_calibrated_recall")) or -1.0,
            -(parse_float(item.get("max_calibrated_fp_per_10k")) or 0.0),
            parse_float(item.get("min_calibrated_coverage")) or -1.0,
        ),
        reverse=True,
    )
    return scored[0] if scored else {"profile": "unknown"}


def evaluate_decision(
    *,
    runs: Sequence[Dict[str, Any]],
    profile_rows: Sequence[Dict[str, Any]],
    audit: Dict[str, Any],
    baseline_recall: float,
) -> Dict[str, Any]:
    crash_total = sum(int((run.get("crash") or {}).get("crash_hit_count", 0)) for run in runs if run.get("completed"))
    suppressed_risk = audit.get("summary", {}).get("may_hide_true_attack_context_event_count", 0)
    natural_attack_present = any((run.get("sample") or {}).get("mode") == "natural_attack_windows" for run in runs)
    required_modes = {"balanced_blocks", "transition"}
    coverage_failures = []
    recall_improvements = []
    fp_improved = []
    for row in profile_rows:
        mode = row.get("sample_mode")
        profile = row.get("profile")
        cal_recall = parse_float(row.get("calibrated_recall"))
        cal_cov = parse_float(row.get("calibrated_coverage"))
        raw_fp = parse_float(row.get("raw_fp_per_10k"))
        cal_fp = parse_float(row.get("calibrated_fp_per_10k"))
        if mode in required_modes and cal_cov is not None and cal_cov < 90.0:
            coverage_failures.append({"case": row.get("case"), "profile": profile, "coverage": cal_cov})
        if cal_recall is not None and cal_recall > baseline_recall:
            recall_improvements.append({"case": row.get("case"), "profile": profile, "recall": cal_recall})
        if raw_fp is not None and cal_fp is not None and cal_fp < raw_fp:
            fp_improved.append({"case": row.get("case"), "profile": profile, "raw_fp": raw_fp, "calibrated_fp": cal_fp})
    best = choose_best_profile(profile_rows)
    if crash_total or suppressed_risk:
        decision = "REJECT"
        reason = "Crashes occurred or suppression may hide true attack-context events."
    elif not natural_attack_present or coverage_failures or not recall_improvements:
        decision = "HOLD"
        reason = "Diagnostics improved, but recall/generalization remains ambiguous."
    elif not fp_improved:
        decision = "HOLD"
        reason = "Recall improved, but FP/10k improvement was not consistently demonstrated."
    else:
        decision = "APPROVE"
        reason = "Recall improved while FP/10k stayed controlled, coverage passed, crash scans were clean, and suppressions avoided attack context."
    return {
        "decision": decision,
        "reason": reason,
        "best_confirmation_profile": best,
        "crash_hit_count_total": crash_total,
        "coverage_failures": coverage_failures,
        "recall_improvements_over_baseline": recall_improvements,
        "fp_improvements": fp_improved,
        "natural_attack_windows_present": natural_attack_present,
        "suppression_audit_summary": audit.get("summary", {}),
        "core_behavior_changed": False,
        "core_behavior_boundary": {
            "reservoir_dynamics": "unchanged",
            "rls_behavior": "unchanged",
            "raw_sentinel_thresholds": "unchanged",
            "anomaly_policy": "unchanged",
            "compression_behavior": "unchanged",
            "hippocampus_memory": "unchanged",
            "incident_card_generation": "unchanged",
            "domain_adapter_math": "unchanged",
            "scope": "proof runner, sampling, confirmation diagnostics, and reporting only",
        },
    }


def write_generalization_report(path: Path, decision: Dict[str, Any], rows: Sequence[Dict[str, Any]], drive_manifest: Dict[str, Any]) -> None:
    lines = [
        "# Sentinel Calibration v1 Recall Diagnostics - 2026-06-07",
        "",
        "## Decision",
        f"- Decision: `{decision.get('decision')}`",
        f"- Reason: {decision.get('reason')}",
        f"- Best confirmation profile: `{decision.get('best_confirmation_profile', {}).get('profile')}`",
        "- Core behavior changed: `false`",
        "",
        "## Profile Sweep Summary",
        "| Case | Mode | Profile | Raw Recall | Cal Recall | Raw FP/10k | Cal FP/10k | Coverage | Crash |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('case')} | {row.get('sample_mode')} | {row.get('profile')} | {fmt(row.get('raw_recall'))} | {fmt(row.get('calibrated_recall'))} | {fmt(row.get('raw_fp_per_10k'))} | {fmt(row.get('calibrated_fp_per_10k'))} | {fmt(row.get('calibrated_coverage'))} | {fmt(row.get('crash_hit_count'))} |"
        )
    lines.extend(
        [
            "",
            "## Drive",
            f"- Drive copy success: `{drive_manifest.get('drive_copy_success')}`",
            f"- Drive folder: `{drive_manifest.get('drive_run_dir')}`",
            f"- Reason: `{drive_manifest.get('reason')}`",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def append_docs(out_dir: Path, decision: Dict[str, Any], drive_manifest: Dict[str, Any]) -> None:
    analysis = Path("docs/proof_runs/2026-06-07/plain_language_test_analysis.md")
    journal = Path("docs/proof_runs/2026-06-07/codex_journal.md")
    analysis.parent.mkdir(parents=True, exist_ok=True)
    analysis.open("a", encoding="utf-8").write(
        "\n".join(
            [
                "",
                "## Calibration recall diagnostics - 2026-06-07",
                "",
                f"- Decision: `{decision.get('decision')}`.",
                f"- Best confirmation profile: `{decision.get('best_confirmation_profile', {}).get('profile')}`.",
                "- Added diagnostic candidate-funnel, block-balanced sampling, natural attack-window sampling, and confirmation profile sweep receipts.",
                "- Core Eidos behavior remained unchanged.",
                f"- Artifacts: `{out_dir}`.",
                f"- Drive: `{drive_manifest.get('drive_run_dir', 'unknown')}`; success: `{drive_manifest.get('drive_copy_success')}`.",
            ]
        )
        + "\n"
    )
    journal.open("a", encoding="utf-8").write(
        "\n".join(
            [
                "",
                "## Calibration recall diagnostics implementation - 2026-06-07",
                "",
                "What changed: proof-side recall diagnostics, block-preserving sampling, natural attack-window sampling, and confirmation profile sweep reporting.",
                "",
                "What did not change: reservoir dynamics, RLS behavior, raw Sentinel thresholds, anomaly policy, compression behavior, hippocampus memory, incident-card generation, and domain adapter math.",
                "",
                f"Decision: `{decision.get('decision')}` - {decision.get('reason')}",
            ]
        )
        + "\n"
    )


def run(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [load_run(case, Path(path)) for case, path in args.run]
    completed = [run for run in runs if run.get("completed")]
    funnel = aggregate_candidate_funnel(completed)
    write_json(out_dir / "candidate_funnel_report.json", funnel)
    write_candidate_funnel_md(out_dir / "candidate_funnel_report.md", funnel)
    rows = profile_sweep_rows(completed)
    write_csv(out_dir / "confirmation_profile_sweep.csv", rows)
    write_profile_sweep_md(out_dir / "confirmation_profile_sweep.md", rows)
    write_csv(out_dir / "generalization_recall_summary.csv", rows)
    write_json(out_dir / "balanced_blocks_manifest.json", sample_manifests(completed, "balanced_blocks"))
    write_json(out_dir / "natural_attack_windows_manifest.json", sample_manifests(completed, "natural_attack_windows"))
    audit = recall_protection_audit(funnel)
    write_json(out_dir / "recall_protection_audit.json", audit)
    decision = evaluate_decision(runs=completed, profile_rows=rows, audit=audit, baseline_recall=args.baseline_recall)
    decision["generated_at_utc"] = utc_now()
    write_json(out_dir / "decision.json", decision)
    write_json(
        out_dir / "run_manifest.json",
        {
            "generated_at_utc": utc_now(),
            "run_id": RUN_ID,
            "run_date": RUN_DATE,
            "source": "aggregate recall diagnostics from labeled proof receipts",
            "runs": [{"case": run["case"], "path": str(run["path"]), "completed": run.get("completed")} for run in runs],
            "outputs": {
                "generalization_recall_report_md": "generalization_recall_report.md",
                "generalization_recall_summary_csv": "generalization_recall_summary.csv",
                "candidate_funnel_report_json": "candidate_funnel_report.json",
                "candidate_funnel_report_md": "candidate_funnel_report.md",
                "confirmation_profile_sweep_csv": "confirmation_profile_sweep.csv",
                "confirmation_profile_sweep_md": "confirmation_profile_sweep.md",
                "balanced_blocks_manifest_json": "balanced_blocks_manifest.json",
                "natural_attack_windows_manifest_json": "natural_attack_windows_manifest.json",
                "recall_protection_audit_json": "recall_protection_audit.json",
                "decision_json": "decision.json",
            },
            "core_behavior_changed": False,
        },
    )
    write_json(out_dir / "drive_manifest.json", {"drive_copy_attempted": True, "drive_copy_success": False, "reason": "pending copy"})
    drive_manifest = proof_helpers.mirror_to_drive(out_dir, RUN_ID, RUN_DATE)
    write_json(out_dir / "drive_manifest.json", drive_manifest)
    write_generalization_report(out_dir / "generalization_recall_report.md", decision, rows, drive_manifest)
    if drive_manifest.get("drive_copy_success"):
        target = Path(drive_manifest["drive_run_dir"]) / "drive_manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_dir / "drive_manifest.json", target)
        proof_helpers.copy_selected_to_drive(
            out_dir,
            drive_manifest,
            [
                out_dir / "generalization_recall_report.md",
                out_dir / "generalization_recall_summary.csv",
                out_dir / "candidate_funnel_report.json",
                out_dir / "candidate_funnel_report.md",
                out_dir / "confirmation_profile_sweep.csv",
                out_dir / "confirmation_profile_sweep.md",
                out_dir / "balanced_blocks_manifest.json",
                out_dir / "natural_attack_windows_manifest.json",
                out_dir / "recall_protection_audit.json",
                out_dir / "decision.json",
                out_dir / "run_manifest.json",
                out_dir / "drive_manifest.json",
            ],
        )
    append_docs(out_dir, decision, drive_manifest)
    return {"out_dir": str(out_dir), "decision": decision, "drive_manifest": drive_manifest}


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = run(argv)
    print(json.dumps({"out_dir": result["out_dir"], "decision": result["decision"].get("decision")}, indent=2))
    return 0 if result["decision"].get("decision") in {"APPROVE", "HOLD"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
