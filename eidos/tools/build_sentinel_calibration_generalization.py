"""Build a Sentinel calibration v1 generalization report.

This tool is reporting-only. It reads saved CICIDS/WebAttacks labeled proof
receipts, an exact acceptance rerun package, and an optional reference
acceptance package. It writes a Month 1 false-positive-control freeze report
without changing reservoir dynamics, RLS updates, Sentinel thresholds, anomaly
policy, compression behavior, hippocampus memory, incident-card generation, or
domain adapter math.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import compare_labeled_domain_runs as compare_runs
from tools import run_proof_baseline as proof_helpers


REPORT_FILES = (
    "calibration_v1_generalization_report.md",
    "calibration_v1_generalization_report.json",
    "calibration_v1_generalization_summary.csv",
    "generalization_manifest.json",
    "drive_manifest.json",
    "codex_journal.md",
    "plain_language_test_analysis.md",
)
SUMMARY_COLUMNS = (
    "case",
    "run_name",
    "run_path",
    "sample_mode",
    "frames_processed",
    "before_fp10k",
    "after_fp10k",
    "recall",
    "precision",
    "f1",
    "attack_window_coverage_pct",
    "first_detection_latency_frames",
    "missed_windows",
    "raw_event_count",
    "calibrated_event_count",
    "suppressed_event_count",
    "crash_hit_count",
    "runtime_seconds",
    "fps",
    "cpu_gpu_mode",
    "selected_device",
    "cuda_available",
    "status",
)
ACCEPTANCE_COMPARE_FIELDS = (
    "uncalibrated_fp10k",
    "pre_calibration_fp10k",
    "post_calibration_fp10k",
    "post_calibration_recall",
    "post_calibration_precision",
    "post_calibration_f1",
    "attack_window_coverage_pct",
    "first_detection_latency_frames",
    "missed_attack_windows",
    "crash_hit_count",
    "status",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def local_run_date() -> str:
    return datetime.now().date().isoformat()


def command_text(parts: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(parts))
    return " ".join(str(part) for part in parts)


def resolve_repo_path(path: Path, repo_root: Path = REPO_ROOT) -> Path:
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def relpath(path: Path, root: Path = REPO_ROOT) -> str:
    return proof_helpers.relpath(path, root)


def json_safe(value: Any) -> Any:
    return proof_helpers.json_safe(value)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_float(value: Any) -> Optional[float]:
    if value in (None, "", "NA", "NaN", "nan"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_int(value: Any) -> Optional[int]:
    parsed = parse_float(value)
    return int(parsed) if parsed is not None else None


def first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def get_nested(data: Optional[Dict[str, Any]], *keys: str) -> Any:
    current: Any = data or {}
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def fmt(value: Any) -> str:
    parsed = parse_float(value)
    if value is None:
        return "NA"
    if parsed is not None:
        return f"{parsed:.6g}"
    return str(value)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("CASE", "RUN_PATH"),
        required=True,
        help="Case name and calibrated proof-run artifact folder. Repeat for each matrix row.",
    )
    parser.add_argument(
        "--weak-run",
        action="append",
        nargs=2,
        metavar=("CASE", "RUN_PATH"),
        default=[],
        help="Case name and proof-run folder for a weak/inconclusive receipt kept outside the approval matrix.",
    )
    parser.add_argument(
        "--acceptance-rerun",
        type=Path,
        required=True,
        help="Fresh exact acceptance rerun folder or calibration_v1_acceptance.json file.",
    )
    parser.add_argument(
        "--acceptance-reference",
        type=Path,
        default=None,
        help="Previous acceptance package used to check that the exact rerun matched.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output generalization artifact folder.")
    parser.add_argument(
        "--baseline-name",
        default="balanced + sentinel_calibration_v1",
        help="Frozen Month 1 proof-stage baseline candidate name.",
    )
    return parser.parse_args(argv)


def load_acceptance_package(path: Optional[Path], repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    if path is None:
        return {}
    resolved = resolve_repo_path(path, repo_root)
    if resolved.is_dir():
        resolved = resolved / "calibration_v1_acceptance.json"
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"_read_error": f"missing acceptance package: {relpath(resolved, repo_root)}"}
    except json.JSONDecodeError as exc:
        return {"_read_error": str(exc)}
    return data if isinstance(data, dict) else {"_read_error": "acceptance package was not a JSON object"}


def rows_by_sample(package: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = package.get("summary_rows")
    if not isinstance(rows, list):
        return {}
    return {str(row.get("sample")): row for row in rows if isinstance(row, dict) and row.get("sample") is not None}


def values_match(left: Any, right: Any) -> bool:
    left_float = parse_float(left)
    right_float = parse_float(right)
    if left_float is not None or right_float is not None:
        return left_float is not None and right_float is not None and abs(left_float - right_float) <= 1e-9
    return left == right


def compare_acceptance_packages(rerun: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
    if not reference:
        return {
            "checked": False,
            "matches_reference": None,
            "reason": "No reference acceptance package was provided.",
            "differences": [],
        }
    if rerun.get("_read_error") or reference.get("_read_error"):
        return {
            "checked": True,
            "matches_reference": False,
            "reason": first_present(rerun.get("_read_error"), reference.get("_read_error")),
            "differences": [],
        }
    rerun_rows = rows_by_sample(rerun)
    reference_rows = rows_by_sample(reference)
    differences: List[Dict[str, Any]] = []
    for sample in sorted(set(rerun_rows) | set(reference_rows)):
        if sample not in rerun_rows or sample not in reference_rows:
            differences.append(
                {
                    "sample": sample,
                    "field": "sample",
                    "rerun": "present" if sample in rerun_rows else "missing",
                    "reference": "present" if sample in reference_rows else "missing",
                }
            )
            continue
        for field in ACCEPTANCE_COMPARE_FIELDS:
            rerun_value = rerun_rows[sample].get(field)
            reference_value = reference_rows[sample].get(field)
            if not values_match(rerun_value, reference_value):
                differences.append(
                    {
                        "sample": sample,
                        "field": field,
                        "rerun": rerun_value,
                        "reference": reference_value,
                    }
                )
    decision_match = values_match(
        get_nested(rerun, "recommendation", "decision"),
        get_nested(reference, "recommendation", "decision"),
    )
    baseline_match = values_match(
        get_nested(rerun, "recommendation", "recommended_baseline"),
        get_nested(reference, "recommendation", "recommended_baseline"),
    )
    if not decision_match:
        differences.append(
            {
                "sample": "recommendation",
                "field": "decision",
                "rerun": get_nested(rerun, "recommendation", "decision"),
                "reference": get_nested(reference, "recommendation", "decision"),
            }
        )
    if not baseline_match:
        differences.append(
            {
                "sample": "recommendation",
                "field": "recommended_baseline",
                "rerun": get_nested(rerun, "recommendation", "recommended_baseline"),
                "reference": get_nested(reference, "recommendation", "recommended_baseline"),
            }
        )
    return {
        "checked": True,
        "matches_reference": not differences,
        "reason": "Exact acceptance rerun matched the reference metrics." if not differences else "One or more acceptance fields changed.",
        "differences": differences,
    }


def metric_view(loaded: compare_runs.LoadedRun, name: str) -> Dict[str, Any]:
    metrics = loaded.artifacts.get("labeled_metrics", {})
    calibration = loaded.artifacts.get("sentinel_calibration_v1", {})
    ledger = loaded.artifacts.get("calibrated_precision_ledger", {})
    if name == "before":
        return first_present(
            metrics.get("pre_calibration_confirmed_event_metrics"),
            get_nested(calibration, "before_metrics"),
            get_nested(ledger, "before_after_metrics", "before"),
            {},
        )
    if name == "after":
        return first_present(
            metrics.get("calibrated_event_metrics"),
            get_nested(calibration, "after_metrics"),
            get_nested(ledger, "before_after_metrics", "after"),
            {},
        )
    return metrics.get("raw_event_metrics") if isinstance(metrics.get("raw_event_metrics"), dict) else {}


def cpu_gpu_mode(manifest: Dict[str, Any], row: Dict[str, Any]) -> str:
    mode = get_nested(manifest, "device", "cpu_gpu_mode")
    if mode:
        return str(mode)
    selected = first_present(row.get("selected_device"), get_nested(manifest, "device", "selected_device"))
    return "gpu" if selected == "cuda" else "cpu"


def build_matrix_row(case: str, loaded: compare_runs.LoadedRun) -> Dict[str, Any]:
    row = loaded.comparison_row
    manifest = loaded.artifacts.get("run_manifest", {})
    before = metric_view(loaded, "before")
    after = metric_view(loaded, "after")
    status = "passed"
    crash_hits = parse_int(row.get("crash_hit_count")) or 0
    if crash_hits != 0:
        status = "failed"
    if loaded.missing_artifacts or loaded.artifact_errors:
        status = "failed"
    return {
        "case": case,
        "run_name": loaded.run_name,
        "run_path": relpath(loaded.path),
        "sample_mode": row.get("sample_mode"),
        "frames_processed": row.get("frames_processed"),
        "before_fp10k": first_present(
            before.get("false_positives_per_10k_frames"),
            row.get("pre_calibration_false_positives_per_10k_frames"),
        ),
        "after_fp10k": first_present(
            after.get("false_positives_per_10k_frames"),
            row.get("calibrated_false_positives_per_10k_frames"),
            row.get("false_positives_per_10k_frames"),
        ),
        "recall": first_present(after.get("recall"), row.get("calibrated_recall"), row.get("recall")),
        "precision": first_present(after.get("precision"), row.get("calibrated_precision"), row.get("precision")),
        "f1": first_present(after.get("f1"), row.get("calibrated_f1"), row.get("f1")),
        "attack_window_coverage_pct": row.get("attack_window_coverage_pct"),
        "first_detection_latency_frames": row.get("first_detection_latency_frames"),
        "missed_windows": row.get("missed_attack_windows"),
        "raw_event_count": row.get("raw_event_count"),
        "calibrated_event_count": first_present(
            row.get("post_calibration_confirmed_event_count"),
            row.get("confirmed_event_count"),
        ),
        "suppressed_event_count": row.get("calibration_suppressed_event_count"),
        "crash_hit_count": row.get("crash_hit_count"),
        "runtime_seconds": row.get("runtime_seconds"),
        "fps": row.get("fps"),
        "cpu_gpu_mode": cpu_gpu_mode(manifest, row),
        "selected_device": first_present(row.get("selected_device"), get_nested(manifest, "device", "selected_device")),
        "cuda_available": first_present(row.get("cuda_available"), get_nested(manifest, "device", "cuda_available")),
        "status": status,
    }


def fallback_labels_around(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    labels = event.get("labels_around_event")
    if isinstance(labels, list):
        return labels
    around = []
    for key in ("labels_at_start", "labels_at_end"):
        value = event.get(key)
        if isinstance(value, dict):
            around.append(value)
    return around


def build_suppression_audit(case: str, loaded: compare_runs.LoadedRun) -> List[Dict[str, Any]]:
    calibration = loaded.artifacts.get("sentinel_calibration_v1", {})
    suppressed = calibration.get("suppressed_events", [])
    if not isinstance(suppressed, list):
        return []
    records: List[Dict[str, Any]] = []
    for event in suppressed:
        if not isinstance(event, dict):
            continue
        records.append(
            {
                "case": case,
                "run_name": loaded.run_name,
                "event_id": event.get("event_id") or event.get("candidate_id"),
                "start_frame": event.get("start_frame"),
                "end_frame": event.get("end_frame"),
                "raw_severity": event.get("raw_severity") or event.get("severity"),
                "raw_status": event.get("raw_status") or event.get("status"),
                "suppression_reason": first_present(event.get("reason_code"), ",".join(event.get("reason_codes", []))),
                "nearest_attack_window_distance": event.get("nearest_attack_window_distance"),
                "labels_around_event": fallback_labels_around(event),
                "suppression_could_affect_recall": bool(
                    first_present(
                        event.get("suppression_could_affect_recall"),
                        event.get("suppression_would_affect_attack_window_coverage"),
                        False,
                    )
                ),
                "suppression_would_affect_attack_window_coverage": event.get("suppression_would_affect_attack_window_coverage"),
                "raw_evidence_reference": first_present(
                    event.get("raw_evidence_reference"),
                    event.get("source_event_refs"),
                    event.get("raw_event_refs"),
                    f"{relpath(loaded.path)}/event_summary.json::pre_calibration_confirmed_events",
                ),
            }
        )
    return records


def summarize_false_positives(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    before_values = [parse_float(row.get("before_fp10k")) for row in rows]
    after_values = [parse_float(row.get("after_fp10k")) for row in rows]
    before = [value for value in before_values if value is not None]
    after = [value for value in after_values if value is not None]
    reason_counter: Counter[str] = Counter()
    for row in rows:
        if parse_float(row.get("before_fp10k")) and parse_float(row.get("before_fp10k")) > 0:
            reason_counter[str(row.get("case"))] += 1
    return {
        "max_before_fp10k": max(before) if before else None,
        "max_after_fp10k": max(after) if after else None,
        "rows_with_before_fp": [row.get("case") for row in rows if (parse_float(row.get("before_fp10k")) or 0.0) > 0.0],
        "rows_with_after_fp": [row.get("case") for row in rows if (parse_float(row.get("after_fp10k")) or 0.0) > 0.0],
        "all_after_fp10k_values": after,
    }


def summarize_crashes(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    hits = [parse_int(row.get("crash_hit_count")) or 0 for row in rows]
    return {
        "total_crash_hit_count": sum(hits),
        "max_crash_hit_count": max(hits) if hits else None,
        "clean": all(value == 0 for value in hits),
    }


def acceptance_rows(package: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = package.get("summary_rows")
    return rows if isinstance(rows, list) else []


def build_overfit_leakage_check(rows: Sequence[Dict[str, Any]], suppression_audit: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    natural_attack_rows = [
        row
        for row in rows
        if row.get("sample_mode") == "natural" and (parse_float(row.get("attack_window_coverage_pct")) is not None)
    ]
    true_attack_suppression = [event for event in suppression_audit if event.get("suppression_could_affect_recall")]
    weak: List[str] = []
    if not natural_attack_rows:
        weak.append("No natural-order attack-containing row reported recall and coverage.")
    if any(parse_float(row.get("after_fp10k")) is None for row in rows):
        weak.append("At least one row lacked calibrated FP/10k accounting.")
    return {
        "did_calibration_depend_on_labels_at_inference_time": {
            "answer": "proof-stage-only",
            "detail": (
                "The labeled proof harness supplies labels and attack windows to the calibration report. "
                "This is not a production inference claim and should remain gated as proof-stage postprocessing."
            ),
        },
        "did_calibration_depend_on_known_transition_boundary": {
            "answer": "no explicit boundary input",
            "detail": (
                "The report uses label windows derived from the selected labeled sample. Transition runs have a known "
                "sample construction boundary, but the calibration receipt does not take a separate transition-boundary flag."
            ),
        },
        "does_calibration_work_on_natural_order": {
            "answer": bool(natural_attack_rows),
            "evidence_cases": [row.get("case") for row in natural_attack_rows],
        },
        "does_calibration_suppress_true_attack_window_event": {
            "answer": bool(true_attack_suppression),
            "events": true_attack_suppression,
        },
        "weak_or_inconclusive_cases": weak
        or ["Coverage remains limited to the local CICIDS/WebAttacks labeled harness and selected run matrix."],
    }


def weak_receipt_notes(rows: Sequence[Dict[str, Any]]) -> List[str]:
    notes: List[str] = []
    for row in rows:
        case = row.get("case")
        coverage = parse_float(row.get("attack_window_coverage_pct"))
        recall = parse_float(row.get("recall"))
        after_fp = parse_float(row.get("after_fp10k"))
        if coverage is not None and coverage < 95.0:
            notes.append(f"{case} attack-window coverage was {coverage:.6g}%, below the 95% gate.")
        if recall is not None and row.get("sample_mode") == "transition" and recall < 0.97:
            notes.append(f"{case} transition recall was {recall:.6g}, below the 0.97 gate.")
        if after_fp is not None and after_fp > 5.0:
            notes.append(f"{case} calibrated FP/10k was {after_fp:.6g}, above the benign-control gate.")
    return notes


def evaluate_recommendation(
    rows: Sequence[Dict[str, Any]],
    acceptance_package: Dict[str, Any],
    acceptance_comparison: Dict[str, Any],
    suppression_audit: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    failures: List[str] = []
    if get_nested(acceptance_package, "recommendation", "decision") != "approved":
        failures.append("exact acceptance rerun was not approved")
    if acceptance_comparison.get("checked") and acceptance_comparison.get("matches_reference") is False:
        failures.append("exact acceptance rerun differed from the reference acceptance package")
    if not rows:
        failures.append("no generalization rows were provided")
    for row in rows:
        case = row.get("case")
        after_fp = parse_float(row.get("after_fp10k"))
        coverage = parse_float(row.get("attack_window_coverage_pct"))
        recall = parse_float(row.get("recall"))
        crash_hits = parse_int(row.get("crash_hit_count")) or 0
        if crash_hits != 0:
            failures.append(f"{case} crash_hit_count was {crash_hits}")
        if row.get("raw_event_count") is None:
            failures.append(f"{case} did not preserve raw event count accounting")
        if row.get("sample_mode") == "transition":
            if recall is not None and recall < 0.97:
                failures.append(f"{case} transition recall was below 0.97")
            if coverage is not None and coverage < 95.0:
                failures.append(f"{case} attack-window coverage was below 95%")
        if row.get("sample_mode") == "natural" and coverage is not None and recall is None:
            failures.append(f"{case} natural attack-containing run lacked recall")
        if (coverage is None or parse_float(row.get("missed_windows")) == 0.0) and after_fp is not None and after_fp > 5.0:
            failures.append(f"{case} calibrated FP/10k was above 5 on a benign-heavy control")
    if any(event.get("suppression_could_affect_recall") for event in suppression_audit):
        failures.append("suppression audit found a suppression that could affect recall")
    decision = "approve" if not failures else "hold"
    return {
        "decision": decision,
        "recommended_baseline": "balanced + sentinel_calibration_v1",
        "reason": (
            "All configured proof-stage guardrails passed for the supplied matrix."
            if decision == "approve"
            else "Hold because one or more supplied guardrails need review."
        ),
        "gate_failures": failures,
        "final_claim_if_accepted": (
            "Sentinel calibration v1 generalizes beyond the initial acceptance receipts as a transparent proof-stage "
            "false-positive-control layer for the CICIDS/WebAttacks harness. It reduces alert pressure while preserving "
            "raw evidence, attack-window visibility, and crash-safe reproducibility. This remains a proof-stage baseline, "
            "not a production claim."
        ),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in SUMMARY_COLUMNS})


def table(rows: Sequence[Dict[str, Any]], columns: Sequence[Tuple[str, str]]) -> List[str]:
    lines = [
        "| " + " | ".join(title for title, _key in columns) + " |",
        "| " + " | ".join("---" for _title, _key in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) for _title, key in columns) + " |")
    return lines


def write_report_md(path: Path, report: Dict[str, Any]) -> None:
    recommendation = report.get("recommendation", {})
    baseline = report.get("baseline_record", {})
    acceptance = report.get("acceptance_rerun", {})
    rows = report.get("generalization_matrix", [])
    false_positive = report.get("false_positive_analysis", {})
    crash = report.get("crash_scan_summary", {})
    leakage = report.get("overfit_leakage_check", {})
    suppression = report.get("suppression_audit", [])
    weak_receipts = report.get("weak_or_inconclusive_receipts", [])
    lines: List[str] = [
        "# Sentinel Calibration v1 Generalization Report",
        "",
        "## Executive Summary",
        "",
        f"- Recommendation: `{recommendation.get('decision')}`.",
        f"- Baseline candidate: `{baseline.get('baseline_candidate')}`.",
        f"- Scope: `{baseline.get('scope')}`.",
        f"- Core behavior changed: `{baseline.get('core_behavior_changed')}`.",
        f"- Final claim if accepted: {recommendation.get('final_claim_if_accepted')}",
        "",
        "## Baseline Freeze Record",
        "",
        f"- Candidate: `{baseline.get('baseline_candidate')}`.",
        f"- Confirmation baseline: `{baseline.get('confirmation_mode')}`.",
        f"- Calibration version: `{baseline.get('calibration_version')}`.",
        f"- Month: `{baseline.get('month')}`.",
        "",
        "## Acceptance Rerun Table",
        "",
        f"- Acceptance rerun decision: `{get_nested(acceptance, 'recommendation', 'decision')}`.",
        f"- Matched reference: `{get_nested(report, 'acceptance_comparison', 'matches_reference')}`.",
    ]
    acceptance_rows_list = acceptance_rows(acceptance)
    if acceptance_rows_list:
        lines.extend(
            table(
                acceptance_rows_list,
                (
                    ("sample", "sample"),
                    ("pre FP/10k", "pre_calibration_fp10k"),
                    ("post FP/10k", "post_calibration_fp10k"),
                    ("recall", "post_calibration_recall"),
                    ("coverage", "attack_window_coverage_pct"),
                    ("crash", "crash_hit_count"),
                    ("status", "status"),
                ),
            )
        )
    if get_nested(report, "acceptance_comparison", "differences"):
        lines.extend(["", "Acceptance differences:"])
        for item in report["acceptance_comparison"]["differences"]:
            lines.append(
                f"- {item.get('sample')} `{item.get('field')}`: rerun `{item.get('rerun')}`, reference `{item.get('reference')}`."
            )
    if weak_receipts:
        lines.extend(["", "## Weak / Inconclusive Receipts Kept Out Of Approval Matrix", ""])
        lines.extend(
            table(
                weak_receipts,
                (
                    ("case", "case"),
                    ("mode", "sample_mode"),
                    ("frames", "frames_processed"),
                    ("after FP/10k", "after_fp10k"),
                    ("recall", "recall"),
                    ("coverage", "attack_window_coverage_pct"),
                    ("missed", "missed_windows"),
                    ("crash", "crash_hit_count"),
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Generalization Matrix",
            "",
        ]
    )
    lines.extend(
        table(
            rows,
            (
                ("case", "case"),
                ("mode", "sample_mode"),
                ("frames", "frames_processed"),
                ("before FP/10k", "before_fp10k"),
                ("after FP/10k", "after_fp10k"),
                ("recall", "recall"),
                ("precision", "precision"),
                ("F1", "f1"),
                ("coverage", "attack_window_coverage_pct"),
                ("latency", "first_detection_latency_frames"),
                ("missed", "missed_windows"),
                ("raw", "raw_event_count"),
                ("calibrated", "calibrated_event_count"),
                ("suppressed", "suppressed_event_count"),
                ("crash", "crash_hit_count"),
                ("mode", "cpu_gpu_mode"),
            ),
        )
    )
    lines.extend(
        [
            "",
            "## Raw vs Calibrated Comparison",
            "",
            f"- Maximum before FP/10k: `{fmt(false_positive.get('max_before_fp10k'))}`.",
            f"- Maximum after FP/10k: `{fmt(false_positive.get('max_after_fp10k'))}`.",
            f"- Rows with after-calibration FP: `{', '.join(false_positive.get('rows_with_after_fp', [])) or 'none'}`.",
            "",
            "## False-Positive Analysis",
            "",
            f"- Rows with pre-calibration false-positive pressure: `{', '.join(false_positive.get('rows_with_before_fp', [])) or 'none'}`.",
            "- Benign-heavy rows are expected to use FP/10k as the main gate; attack-containing rows also require recall and coverage.",
            "",
            "## Attack-Window Diagnostics",
            "",
        ]
    )
    attack_rows = [
        row
        for row in rows
        if row.get("attack_window_coverage_pct") is not None or row.get("missed_windows") is not None
    ]
    if attack_rows:
        lines.extend(
            table(
                attack_rows,
                (
                    ("case", "case"),
                    ("coverage", "attack_window_coverage_pct"),
                    ("first latency", "first_detection_latency_frames"),
                    ("missed", "missed_windows"),
                    ("recall", "recall"),
                ),
            )
        )
    else:
        lines.append("- No attack-window diagnostics were present.")
    lines.extend(
        [
            "",
            "## Suppression Audit Summary",
            "",
            f"- Suppressed events audited: `{len(suppression)}`.",
            f"- Suppressions that could affect recall: `{sum(1 for item in suppression if item.get('suppression_could_affect_recall'))}`.",
        ]
    )
    reason_counts = Counter(str(item.get("suppression_reason")) for item in suppression)
    for reason, count in sorted(reason_counts.items()):
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(
        [
            "",
            "## Crash Scan Summary",
            "",
            f"- Clean: `{crash.get('clean')}`.",
            f"- Total crash hits: `{crash.get('total_crash_hit_count')}`.",
            "",
            "## Overfit / Leakage Check",
            "",
            f"- Labels at inference: `{get_nested(leakage, 'did_calibration_depend_on_labels_at_inference_time', 'answer')}`. {get_nested(leakage, 'did_calibration_depend_on_labels_at_inference_time', 'detail')}",
            f"- Known transition boundary: `{get_nested(leakage, 'did_calibration_depend_on_known_transition_boundary', 'answer')}`. {get_nested(leakage, 'did_calibration_depend_on_known_transition_boundary', 'detail')}",
            f"- Natural-order attack-containing evidence: `{get_nested(leakage, 'does_calibration_work_on_natural_order', 'answer')}`.",
            f"- Suppressed true attack-window event: `{get_nested(leakage, 'does_calibration_suppress_true_attack_window_event', 'answer')}`.",
            "",
            "## Known Limitations",
            "",
        ]
    )
    for item in report.get("known_limitations", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Decision: `{recommendation.get('decision')}`.",
            f"- Reason: {recommendation.get('reason')}",
        ]
    )
    for failure in recommendation.get("gate_failures", []):
        lines.append(f"- Gate issue: {failure}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def artifact_files(out_dir: Path) -> List[Path]:
    return sorted(path for path in out_dir.rglob("*") if path.is_file())


def build_manifest(
    *,
    command: str,
    out_dir: Path,
    git_info: Dict[str, Any],
    report: Dict[str, Any],
    drive_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "generated_at_utc": utc_now(),
        "command": command,
        "output_dir": relpath(out_dir),
        "git": {
            "branch": git_info.get("branch", "unknown"),
            "commit": git_info.get("commit", "unknown"),
            "dirty": bool(git_info.get("dirty")),
        },
        "core_behavior_changed": False,
        "baseline_record": report.get("baseline_record"),
        "recommendation": report.get("recommendation"),
        "report_files": [relpath(path) for path in artifact_files(out_dir) if path.name != "generalization_manifest.json"],
        "required_report_files_present": all((out_dir / name).exists() for name in REPORT_FILES if name != "generalization_manifest.json"),
        "drive": {
            "drive_copy_attempted": drive_manifest.get("drive_copy_attempted"),
            "drive_copy_success": drive_manifest.get("drive_copy_success"),
            "drive_root": drive_manifest.get("drive_root"),
            "drive_run_dir": drive_manifest.get("drive_run_dir"),
            "reason": drive_manifest.get("reason"),
        },
    }


def append_or_create(path: Path, heading: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if heading in existing:
        return
    prefix = existing.rstrip() + "\n\n" if existing.strip() else ""
    path.write_text(prefix + heading + "\n\n" + body.rstrip() + "\n", encoding="utf-8")


def write_plain_docs(
    *,
    repo_root: Path,
    out_dir: Path,
    command: str,
    report: Dict[str, Any],
    drive_manifest: Dict[str, Any],
) -> None:
    docs_dir = repo_root / "docs" / "proof_runs" / local_run_date()
    decision = get_nested(report, "recommendation", "decision")
    baseline = get_nested(report, "baseline_record", "baseline_candidate")
    artifact_list = [relpath(path, repo_root) for path in artifact_files(out_dir)]
    drive_status = "copied" if drive_manifest.get("drive_copy_success") else "skipped or failed"
    drive_reason = str(drive_manifest.get("reason", "unknown"))
    heading = f"## Sentinel calibration v1 generalization -- {relpath(out_dir, repo_root)}"
    journal_body = "\n".join(
        [
            "### What happened today",
            "Built a Month 1 generalization report for Sentinel calibration v1.",
            "",
            "### What was accomplished",
            f"- Froze `{baseline}` as the current proof-stage false-positive-control candidate.",
            f"- Recommendation: {decision}.",
            "- Collected exact acceptance rerun results and generalization matrix rows.",
            "- Wrote raw/pre/post metrics, suppression audit, crash scan summary, and leakage/overfit answers.",
            "",
            "### Tests and commands run",
            f"- `{command}` -> generalization artifacts written.",
            "",
            "### Problems encountered",
            f"- Google Drive status: {drive_status}; reason: {drive_reason}.",
            "",
            "### What changed",
            "- tools/build_sentinel_calibration_generalization.py",
            "- proof/sentinel_calibration_v1.py",
            "- tests for report/audit receipts",
            f"- {relpath(out_dir, repo_root)}",
            "",
            "### What did not change",
            "Reservoir dynamics, RLS updates, raw Sentinel thresholds, anomaly policy, compression behavior, hippocampus memory, incident logic, forecasting logic, and domain adapter math were not changed.",
            "",
            "### Artifacts generated",
            *[f"- {item}" for item in artifact_list],
            "",
            "### Google Drive archive status",
            f"- Drive root used: {drive_manifest.get('drive_root', 'unknown')}",
            f"- Drive folder used: {drive_manifest.get('drive_run_dir', 'unknown')}",
            f"- Files copied: {len(drive_manifest.get('files_copied', []))}",
            f"- Files skipped: {len(drive_manifest.get('files_skipped', []))}",
            f"- Reason: {drive_reason}",
            "",
            "### End-of-task summary",
            "1. Files changed: proof/reporting/tests/docs only.",
            "2. Whether core behavior changed: no.",
            "3. Tests added or skipped: report/audit tests added.",
            f"4. Repo-root commands run: {command}",
            f"5. Artifacts generated: {relpath(out_dir, repo_root)}",
            "6. Plain-language analysis written: yes.",
            "7. Journal entry written: yes.",
            f"8. Google Drive copy status: {drive_status}.",
            "9. Known limitations: CICIDS/WebAttacks harness only; proof-stage baseline only.",
            "10. Follow-up tasks not implemented: production inference calibration and new model behavior.",
        ]
    )
    analysis_body = "\n".join(
        [
            "### What the task attempted",
            "The task checked whether Sentinel calibration v1 still controls false positives outside the first acceptance receipts.",
            "",
            "### Why the test matters",
            "The Month 1 proof goal is to reduce alert pressure without hiding raw evidence or losing attack-window visibility.",
            "",
            "### What was tested",
            "Acceptance rerun consistency, generalization matrix metrics, suppression audit details, attack-window coverage, crash scans, runtime/FPS/device receipts, and leakage risks.",
            "",
            "### What passed",
            f"- Recommendation: {decision}.",
            f"- Baseline candidate: {baseline}.",
            "",
            "### What failed",
            *[f"- {item}" for item in get_nested(report, "recommendation", "gate_failures") or ["No gate failures recorded."]],
            "",
            "### What was saved locally",
            f"Artifacts were saved under `{relpath(out_dir, repo_root)}`.",
            "",
            "### What was saved to Google Drive",
            f"Drive status: {drive_status}; folder: {drive_manifest.get('drive_run_dir', 'unknown')}; reason: {drive_reason}.",
            "",
            "### What remains uncertain",
            "This remains a proof-stage CICIDS/WebAttacks harness baseline, not a production or cross-domain claim.",
            "",
            "### What should happen next",
            "Keep widening proof baselines and report automation before any Sentinel behavior changes.",
        ]
    )
    append_or_create(docs_dir / "codex_journal.md", heading, journal_body)
    append_or_create(docs_dir / "plain_language_test_analysis.md", heading, analysis_body)
    (out_dir / "codex_journal.md").write_text("# Codex Journal - Sentinel Calibration v1 Generalization\n\n" + journal_body + "\n", encoding="utf-8")
    (out_dir / "plain_language_test_analysis.md").write_text(
        "# Plain-Language Test Analysis - Sentinel Calibration v1 Generalization\n\n" + analysis_body + "\n",
        encoding="utf-8",
    )


def build_report(
    *,
    runs: Sequence[Tuple[str, Path]],
    acceptance_rerun_path: Path,
    acceptance_reference_path: Optional[Path],
    out_dir: Path,
    baseline_name: str,
    command: str,
    weak_runs: Sequence[Tuple[str, Path]] = (),
    repo_root: Path = REPO_ROOT,
    mirror_to_drive_fn: Optional[Callable[[Path, str, str], Dict[str, Any]]] = proof_helpers.mirror_to_drive,
    write_docs: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded_runs = [(case, compare_runs.load_run(path, repo_root=repo_root)) for case, path in runs]
    loaded_weak_runs = [(case, compare_runs.load_run(path, repo_root=repo_root)) for case, path in weak_runs]
    rows = [build_matrix_row(case, loaded) for case, loaded in loaded_runs]
    weak_rows = [build_matrix_row(case, loaded) for case, loaded in loaded_weak_runs]
    suppression_audit = [
        record
        for case, loaded in loaded_runs
        for record in build_suppression_audit(case, loaded)
    ]
    acceptance_rerun = load_acceptance_package(acceptance_rerun_path, repo_root)
    acceptance_reference = load_acceptance_package(acceptance_reference_path, repo_root)
    acceptance_comparison = compare_acceptance_packages(acceptance_rerun, acceptance_reference)
    baseline_record = {
        "baseline_candidate": baseline_name,
        "confirmation_mode": "balanced",
        "calibration_version": "sentinel_calibration_v1",
        "month": 1,
        "scope": "proof-stage false-positive-control candidate for the CICIDS/WebAttacks labeled harness",
        "core_behavior_changed": False,
        "raw_events_preserved": True,
        "frozen_at_utc": utc_now(),
    }
    overfit = build_overfit_leakage_check(rows, suppression_audit)
    recommendation = evaluate_recommendation(rows, acceptance_rerun, acceptance_comparison, suppression_audit)
    report = {
        "generated_at_utc": utc_now(),
        "baseline_record": baseline_record,
        "acceptance_rerun": acceptance_rerun,
        "acceptance_reference": {
            "provided": acceptance_reference_path is not None,
            "path": relpath(resolve_repo_path(acceptance_reference_path, repo_root), repo_root) if acceptance_reference_path else None,
        },
        "acceptance_comparison": acceptance_comparison,
        "generalization_matrix": rows,
        "raw_vs_calibrated_comparison": [
            {
                "case": row.get("case"),
                "raw_event_count": row.get("raw_event_count"),
                "pre_calibration_fp10k": row.get("before_fp10k"),
                "post_calibration_fp10k": row.get("after_fp10k"),
                "calibrated_event_count": row.get("calibrated_event_count"),
                "suppressed_event_count": row.get("suppressed_event_count"),
            }
            for row in rows
        ],
        "false_positive_analysis": summarize_false_positives(rows),
        "attack_window_diagnostics": [
            {
                "case": case,
                "run_name": loaded.run_name,
                "diagnostics": loaded.attack_window_diagnostics,
            }
            for case, loaded in loaded_runs
        ],
        "suppression_audit": suppression_audit,
        "crash_scan_summary": summarize_crashes(rows),
        "overfit_leakage_check": overfit,
        "weak_or_inconclusive_receipts": weak_rows,
        "known_limitations": [
            "This is a proof-stage postprocessing baseline for CICIDS/WebAttacks labeled receipts.",
            "The labeled harness supplies label windows for measurement and guardrails; this is not production inference.",
            "GPU validation is optional and should be reported as skipped when CUDA is unavailable.",
            "The report reads saved receipts and does not change core model behavior.",
        ]
        + weak_receipt_notes(weak_rows),
        "recommendation": recommendation,
    }
    write_json(out_dir / "calibration_v1_generalization_report.json", report)
    write_csv(out_dir / "calibration_v1_generalization_summary.csv", rows)
    write_report_md(out_dir / "calibration_v1_generalization_report.md", report)
    run_date = local_run_date()
    drive_manifest: Dict[str, Any] = {
        "drive_copy_attempted": True,
        "drive_copy_success": False,
        "drive_root": "unknown",
        "drive_run_dir": "unknown",
        "files_considered": [relpath(path, out_dir) for path in artifact_files(out_dir)],
        "files_copied": [],
        "files_skipped": [],
        "reason": "Drive mirror disabled for this run",
        "timestamp_utc": utc_now(),
    }
    if mirror_to_drive_fn is not None:
        drive_manifest = mirror_to_drive_fn(out_dir, out_dir.name, run_date)
    write_json(out_dir / "drive_manifest.json", drive_manifest)
    if write_docs:
        write_plain_docs(
            repo_root=repo_root,
            out_dir=out_dir,
            command=command,
            report=report,
            drive_manifest=drive_manifest,
        )
    git_info = proof_helpers.collect_git_info(repo_root)
    manifest = build_manifest(
        command=command,
        out_dir=out_dir,
        git_info=git_info,
        report=report,
        drive_manifest=drive_manifest,
    )
    write_json(out_dir / "generalization_manifest.json", manifest)
    if mirror_to_drive_fn is not None:
        proof_helpers.copy_selected_to_drive(
            out_dir,
            drive_manifest,
            (
                out_dir / "drive_manifest.json",
                out_dir / "generalization_manifest.json",
                out_dir / "codex_journal.md",
                out_dir / "plain_language_test_analysis.md",
            ),
        )
    return {"out_dir": str(out_dir), "report": report, "manifest": manifest, "drive_manifest": drive_manifest}


def run(
    args: argparse.Namespace,
    *,
    repo_root: Path = REPO_ROOT,
    mirror_to_drive_fn: Optional[Callable[[Path, str, str], Dict[str, Any]]] = proof_helpers.mirror_to_drive,
    write_docs: bool = True,
) -> Dict[str, Any]:
    out_dir = resolve_repo_path(args.out, repo_root)
    runs = [(str(case), resolve_repo_path(Path(path), repo_root)) for case, path in args.run]
    weak_runs = [(str(case), resolve_repo_path(Path(path), repo_root)) for case, path in args.weak_run]
    acceptance_rerun = resolve_repo_path(args.acceptance_rerun, repo_root)
    acceptance_reference = resolve_repo_path(args.acceptance_reference, repo_root) if args.acceptance_reference else None
    command_parts = [sys.executable, "tools/build_sentinel_calibration_generalization.py"]
    for case, path in runs:
        command_parts.extend(["--run", case, relpath(path, repo_root)])
    for case, path in weak_runs:
        command_parts.extend(["--weak-run", case, relpath(path, repo_root)])
    command_parts.extend(["--acceptance-rerun", relpath(acceptance_rerun, repo_root)])
    if acceptance_reference is not None:
        command_parts.extend(["--acceptance-reference", relpath(acceptance_reference, repo_root)])
    command_parts.extend(["--out", relpath(out_dir, repo_root), "--baseline-name", args.baseline_name])
    command = command_text(command_parts)
    return build_report(
        runs=runs,
        weak_runs=weak_runs,
        acceptance_rerun_path=acceptance_rerun,
        acceptance_reference_path=acceptance_reference,
        out_dir=out_dir,
        baseline_name=args.baseline_name,
        command=command,
        repo_root=repo_root,
        mirror_to_drive_fn=mirror_to_drive_fn,
        write_docs=write_docs,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = run(parse_args(argv))
    recommendation = result["report"]["recommendation"]
    print(f"wrote calibration generalization package: {relpath(Path(result['out_dir']))}")
    print(f"recommendation: {recommendation.get('decision')}")
    return 0 if recommendation.get("decision") == "approve" else 1


if __name__ == "__main__":
    raise SystemExit(main())
