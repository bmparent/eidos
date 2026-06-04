"""Compare labeled-domain proof artifact folders.

This tool is reporting-only. It reads already-generated labeled proof receipts
and writes a side-by-side confirmation-mode evidence package without changing
reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy,
compression behavior, hippocampus memory, incident-card generation, or domain
adapter math.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_proof_baseline as proof_helpers


RECOMMENDATION_POLICIES = ("precision_first", "balanced_f1", "recall_first")
COMPARISON_FILES = (
    "comparison_summary.csv",
    "comparison_report.md",
    "recommended_confirmation_mode.json",
    "attack_window_comparison.md",
    "false_positive_taxonomy_summary.md",
    "failure_cases.md",
)
ARTIFACT_SPECS = (
    ("precision_ledger", "precision_ledger.json", "json"),
    ("labeled_metrics", "labeled_metrics.json", "json"),
    ("event_confirmation_report", "event_confirmation_report.json", "json"),
    ("sentinel_calibration_v1", "sentinel_calibration_v1.json", "json"),
    ("calibrated_precision_ledger", "calibrated_precision_ledger.json", "json"),
    ("event_summary", "event_summary.json", "json"),
    ("proof_digest", "proof_digest.json", "json"),
    ("run_manifest", "run_manifest.json", "json"),
    ("crash_scan", "crash_scan.json", "json"),
    ("benchmark_summary_csv", "benchmark_summary.csv", "csv"),
    ("benchmark_summary_json", "benchmark_summary.json", "json"),
)
CSV_COLUMNS = (
    "run_name",
    "run_path",
    "confirmation_mode",
    "calibration_enabled",
    "calibration_version",
    "sample_mode",
    "frames_processed",
    "raw_event_count",
    "merged_event_count",
    "deduped_event_count",
    "confirmed_event_count",
    "pre_calibration_confirmed_event_count",
    "post_calibration_confirmed_event_count",
    "calibration_suppressed_event_count",
    "raw_precision",
    "raw_recall",
    "raw_f1",
    "raw_false_positives_per_10k_frames",
    "pre_calibration_precision",
    "pre_calibration_recall",
    "pre_calibration_f1",
    "pre_calibration_false_positives_per_10k_frames",
    "calibrated_precision",
    "calibrated_recall",
    "calibrated_f1",
    "calibrated_false_positives_per_10k_frames",
    "precision",
    "recall",
    "f1",
    "false_positives",
    "false_positives_per_10k_frames",
    "attack_window_count",
    "attack_window_coverage_pct",
    "attack_window_mean_frame_coverage_pct",
    "first_detection_latency_frames",
    "mean_detection_latency_frames",
    "late_detection_count",
    "missed_attack_windows",
    "duplicate_event_count",
    "incident_card_count",
    "incident_card_coverage",
    "eidos_compression_ratio",
    "best_external_compression_baseline",
    "best_external_compression_ratio",
    "runtime_seconds",
    "fps",
    "selected_device",
    "cuda_available",
    "crash_hit_count",
    "git_commit",
    "config_hash",
    "missing_artifacts",
)


@dataclass
class LoadedRun:
    path: Path
    run_name: str
    artifacts: Dict[str, Any]
    missing_artifacts: List[str]
    artifact_errors: List[str]
    benchmark_rows: List[Dict[str, Any]]
    comparison_row: Dict[str, Any]
    attack_window_diagnostics: List[Dict[str, Any]]
    false_positive_taxonomy: Dict[str, int]
    ledger_false_positive_taxonomy: Dict[str, int]
    failure_cases: Dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def local_run_date() -> str:
    return datetime.now().date().isoformat()


def command_text(parts: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(parts))
    return " ".join(str(part) for part in parts)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="Labeled proof artifact folders to compare.")
    parser.add_argument("--out", type=Path, required=True, help="Output comparison artifact folder.")
    parser.add_argument("--recommendation-policy", choices=RECOMMENDATION_POLICIES, required=True)
    return parser.parse_args(argv)


def resolve_repo_path(path: Path, repo_root: Path = REPO_ROOT) -> Path:
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def relpath(path: Path, root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"{path.name}: {exc}"
    if isinstance(data, dict):
        return data, None
    return {"items": data}, None


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle)), None
    except Exception as exc:
        return [], f"{path.name}: {exc}"


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
    if parsed is None:
        return None
    return int(parsed)


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


def event_start(event: Dict[str, Any]) -> Optional[int]:
    return parse_int(first_present(event.get("start_frame"), get_nested(event, "score_detail", "start_frame")))


def event_end(event: Dict[str, Any]) -> Optional[int]:
    return parse_int(first_present(event.get("end_frame"), get_nested(event, "score_detail", "end_frame")))


def overlaps(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_start = event_start(left)
    left_end = event_end(left)
    right_start = event_start(right)
    right_end = event_end(right)
    if None in (left_start, left_end, right_start, right_end):
        return False
    return int(left_start) <= int(right_end) and int(right_start) <= int(left_end)


def coverage_percent(window: Dict[str, Any], events: Sequence[Dict[str, Any]]) -> Optional[float]:
    start = event_start(window)
    end = event_end(window)
    if start is None or end is None or end < start:
        return None
    intervals: List[Tuple[int, int]] = []
    for event in events:
        if not overlaps(event, window):
            continue
        event_left = event_start(event)
        event_right = event_end(event)
        if event_left is None or event_right is None:
            continue
        intervals.append((max(start, event_left), min(end, event_right)))
    if not intervals:
        return 0.0
    intervals.sort()
    merged: List[Tuple[int, int]] = []
    for left, right in intervals:
        if not merged or left > merged[-1][1] + 1:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
    covered = sum(right - left + 1 for left, right in merged)
    return round(covered * 100.0 / (end - start + 1), 6)


def attack_window_diagnostics(label_windows: Sequence[Dict[str, Any]], confirmed_events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    diagnostics: List[Dict[str, Any]] = []
    for index, window in enumerate(label_windows, start=1):
        inside = [event for event in confirmed_events if overlaps(event, window)]
        window_start = event_start(window)
        window_end = event_end(window)
        first_detection = None
        if window_start is not None:
            first_detection = min(
                (max(event_start(event) or window_start, window_start) for event in inside),
                default=None,
            )
        diagnostics.append(
            {
                "window_index": index,
                "start_frame": window_start,
                "end_frame": window_end,
                "first_detection_frame": first_detection,
                "detection_latency": (
                    first_detection - window_start if first_detection is not None and window_start is not None else None
                ),
                "coverage_percentage": coverage_percent(window, inside),
                "missed": first_detection is None,
                "detection_event_ids": [event.get("event_id") for event in inside],
                "label_distribution": window.get("label_distribution", {}),
            }
        )
    return diagnostics


def summarize_attack_windows(diagnostics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(diagnostics)
    detected = sum(1 for item in diagnostics if not item.get("missed"))
    missed = total - detected
    latencies = [parse_float(item.get("detection_latency")) for item in diagnostics if item.get("detection_latency") is not None]
    coverages = [parse_float(item.get("coverage_percentage")) for item in diagnostics if item.get("coverage_percentage") is not None]
    return {
        "attack_window_count": total,
        "detected_attack_windows": detected,
        "missed_attack_windows": missed,
        "attack_window_coverage_pct": round(detected * 100.0 / total, 6) if total else None,
        "attack_window_mean_frame_coverage_pct": round(sum(coverages) / len(coverages), 6) if coverages else None,
        "first_detection_latency_frames": min(latencies) if latencies else None,
        "mean_detection_latency_frames": round(sum(latencies) / len(latencies), 6) if latencies else None,
        "late_detection_count": sum(1 for value in latencies if value > 0),
    }


def false_positive_classification(event: Dict[str, Any]) -> str:
    return str(
        first_present(
            event.get("false_positive_classification"),
            get_nested(event, "score_detail", "false_positive_classification"),
            "fully_benign",
        )
    )


def confirmed_false_positive_events(
    confirmed_events: Sequence[Dict[str, Any]],
    label_windows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [event for event in confirmed_events if not any(overlaps(event, window) for window in label_windows)]


def taxonomy_for_events(events: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter(false_positive_classification(event) for event in events)
    return dict(sorted(counts.items()))


def ledger_taxonomy(precision_ledger: Dict[str, Any]) -> Dict[str, int]:
    counts = Counter(str(item.get("classification", "unknown")) for item in precision_ledger.get("false_positive_events", []))
    return dict(sorted(counts.items()))


def event_overlaps_attack_or_has_attack_frames(event: Dict[str, Any], label_windows: Sequence[Dict[str, Any]]) -> bool:
    overlap_frames = parse_float(first_present(event.get("overlap_attack_frames"), get_nested(event, "score_detail", "overlap_attack_frames")))
    if overlap_frames is not None and overlap_frames > 0:
        return True
    return any(overlaps(event, window) for window in label_windows)


def useful_suppressed_events(event_report: Dict[str, Any], label_windows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    suppressed = list(event_report.get("suppressed_events", []))
    if not suppressed:
        suppressed = [item for item in event_report.get("decisions", []) if str(item.get("decision")) != "confirmed"]
    return [event for event in suppressed if event_overlaps_attack_or_has_attack_frames(event, label_windows)]


def best_external_baseline(metrics: Dict[str, Any], digest: Dict[str, Any], benchmark_row: Dict[str, Any]) -> Tuple[Any, Any]:
    baseline = first_present(metrics.get("external_compression_baselines"), digest.get("external_compression_baselines"))
    if isinstance(baseline, dict):
        return (
            baseline.get("best_baseline"),
            baseline.get("best_baseline_compression_ratio"),
        )
    if isinstance(baseline, list):
        candidates = [item for item in baseline if isinstance(item, dict) and parse_float(item.get("best_baseline_compression_ratio")) is not None]
        if candidates:
            best = max(candidates, key=lambda item: parse_float(item.get("best_baseline_compression_ratio")) or -1.0)
            return best.get("best_baseline"), best.get("best_baseline_compression_ratio")
    return (
        first_present(benchmark_row.get("best_external_baseline"), benchmark_row.get("best_baseline")),
        first_present(benchmark_row.get("best_external_compression_ratio"), benchmark_row.get("best_baseline_compression_ratio")),
    )


def load_artifacts(run_path: Path) -> Tuple[Dict[str, Any], List[str], List[str], List[Dict[str, Any]]]:
    artifacts: Dict[str, Any] = {}
    missing: List[str] = []
    errors: List[str] = []
    benchmark_rows: List[Dict[str, Any]] = []
    for key, filename, kind in ARTIFACT_SPECS:
        path = run_path / filename
        if not path.exists():
            missing.append(filename)
            continue
        if kind == "json":
            data, error = read_json(path)
            if error:
                errors.append(error)
            artifacts[key] = data or {}
        elif kind == "csv":
            rows, error = read_csv_rows(path)
            if error:
                errors.append(error)
            artifacts[key] = rows
            benchmark_rows = rows
    if "benchmark_summary.csv" not in missing and "benchmark_summary.json" in missing:
        missing.remove("benchmark_summary.json")
    if "benchmark_summary.json" not in missing and "benchmark_summary.csv" in missing:
        missing.remove("benchmark_summary.csv")
    for optional_name in ("sentinel_calibration_v1.json", "calibrated_precision_ledger.json"):
        if optional_name in missing:
            missing.remove(optional_name)
    return artifacts, missing, errors, benchmark_rows


def extract_comparison_row(
    *,
    run_path: Path,
    artifacts: Dict[str, Any],
    missing_artifacts: List[str],
    benchmark_rows: Sequence[Dict[str, Any]],
    attack_summary: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = artifacts.get("labeled_metrics", {})
    event_summary = artifacts.get("event_summary", {})
    event_report = artifacts.get("event_confirmation_report", {})
    calibration_report = artifacts.get("sentinel_calibration_v1", {})
    calibrated_ledger = artifacts.get("calibrated_precision_ledger", {})
    precision_ledger = artifacts.get("precision_ledger", {})
    digest = artifacts.get("proof_digest", {})
    manifest = artifacts.get("run_manifest", {})
    crash_scan = artifacts.get("crash_scan", {})
    benchmark_row = benchmark_rows[0] if benchmark_rows else {}
    accounting = precision_ledger.get("incident_card_accounting", {}) if isinstance(precision_ledger, dict) else {}
    best_name, best_ratio = best_external_baseline(metrics, digest, benchmark_row)
    mode = first_present(
        metrics.get("confirmation_mode"),
        event_summary.get("confirmation_mode"),
        event_report.get("mode"),
        get_nested(manifest, "event_confirmation", "mode"),
        "unknown",
    )
    calibration_enabled = first_present(
        metrics.get("calibration_enabled"),
        event_summary.get("calibration_enabled"),
        calibration_report.get("calibration_enabled"),
        get_nested(manifest, "sentinel_calibration_v1", "enabled"),
        False,
    )
    calibration_version = first_present(
        metrics.get("calibration_version"),
        event_summary.get("calibration_version"),
        calibration_report.get("calibration_version"),
        get_nested(manifest, "sentinel_calibration_v1", "version"),
        "disabled",
    )
    raw_metric_view = metrics.get("raw_event_metrics") if isinstance(metrics.get("raw_event_metrics"), dict) else {}
    pre_metric_view = first_present(
        metrics.get("pre_calibration_confirmed_event_metrics"),
        get_nested(calibration_report, "before_metrics"),
        get_nested(calibrated_ledger, "before_after_metrics", "before"),
        {},
    )
    calibrated_metric_view = first_present(
        metrics.get("calibrated_event_metrics"),
        get_nested(calibration_report, "after_metrics"),
        get_nested(calibrated_ledger, "before_after_metrics", "after"),
        {},
    )
    return {
        "run_name": run_path.name,
        "run_path": relpath(run_path),
        "confirmation_mode": mode,
        "calibration_enabled": calibration_enabled,
        "calibration_version": calibration_version,
        "sample_mode": first_present(metrics.get("sample_mode"), digest.get("sample_mode"), benchmark_row.get("sample_mode")),
        "frames_processed": first_present(metrics.get("frames_processed"), digest.get("frames_processed"), benchmark_row.get("frames_processed")),
        "raw_event_count": first_present(metrics.get("proof_raw_event_count"), event_summary.get("raw_event_count"), event_report.get("raw_event_count"), accounting.get("proof_raw_event_count")),
        "merged_event_count": first_present(metrics.get("proof_merged_event_count"), event_summary.get("merged_event_count"), event_report.get("merged_event_count"), accounting.get("proof_merged_event_count")),
        "deduped_event_count": first_present(metrics.get("proof_deduped_event_count"), event_summary.get("deduped_event_count"), event_report.get("deduped_event_count"), accounting.get("proof_deduped_event_count")),
        "confirmed_event_count": first_present(metrics.get("proof_confirmed_event_count"), event_summary.get("confirmed_event_count"), event_report.get("confirmed_event_count"), metrics.get("confirmed_events")),
        "pre_calibration_confirmed_event_count": first_present(
            metrics.get("pre_calibration_confirmed_events"),
            event_summary.get("pre_calibration_confirmed_event_count"),
            get_nested(calibration_report, "counts", "pre_calibration_confirmed_events"),
        ),
        "post_calibration_confirmed_event_count": first_present(
            metrics.get("post_calibration_confirmed_events"),
            event_summary.get("post_calibration_confirmed_event_count"),
            get_nested(calibration_report, "counts", "post_calibration_confirmed_events"),
            metrics.get("confirmed_events"),
        ),
        "calibration_suppressed_event_count": first_present(
            metrics.get("calibration_suppressed_events"),
            event_summary.get("calibration_suppressed_event_count"),
            get_nested(calibration_report, "counts", "suppressed_events"),
            0,
        ),
        "raw_precision": raw_metric_view.get("precision"),
        "raw_recall": raw_metric_view.get("recall"),
        "raw_f1": raw_metric_view.get("f1"),
        "raw_false_positives_per_10k_frames": raw_metric_view.get("false_positives_per_10k_frames"),
        "pre_calibration_precision": pre_metric_view.get("precision") if isinstance(pre_metric_view, dict) else None,
        "pre_calibration_recall": pre_metric_view.get("recall") if isinstance(pre_metric_view, dict) else None,
        "pre_calibration_f1": pre_metric_view.get("f1") if isinstance(pre_metric_view, dict) else None,
        "pre_calibration_false_positives_per_10k_frames": (
            pre_metric_view.get("false_positives_per_10k_frames") if isinstance(pre_metric_view, dict) else None
        ),
        "calibrated_precision": calibrated_metric_view.get("precision") if isinstance(calibrated_metric_view, dict) else None,
        "calibrated_recall": calibrated_metric_view.get("recall") if isinstance(calibrated_metric_view, dict) else None,
        "calibrated_f1": calibrated_metric_view.get("f1") if isinstance(calibrated_metric_view, dict) else None,
        "calibrated_false_positives_per_10k_frames": (
            calibrated_metric_view.get("false_positives_per_10k_frames") if isinstance(calibrated_metric_view, dict) else None
        ),
        "precision": first_present(metrics.get("precision"), get_nested(metrics, "confirmed_event_metrics", "precision"), digest.get("precision"), benchmark_row.get("precision")),
        "recall": first_present(metrics.get("recall"), get_nested(metrics, "confirmed_event_metrics", "recall"), digest.get("recall"), benchmark_row.get("recall")),
        "f1": first_present(metrics.get("f1"), get_nested(metrics, "confirmed_event_metrics", "f1"), digest.get("f1"), benchmark_row.get("f1")),
        "false_positives": first_present(metrics.get("false_positives"), get_nested(metrics, "confirmed_event_metrics", "false_positives"), digest.get("false_positives"), benchmark_row.get("false_positives")),
        "false_positives_per_10k_frames": first_present(metrics.get("false_positives_per_10k_frames"), get_nested(metrics, "confirmed_event_metrics", "false_positives_per_10k_frames"), digest.get("false_positives_per_10k_frames"), benchmark_row.get("false_positives_per_10k_frames")),
        "attack_window_count": attack_summary.get("attack_window_count"),
        "attack_window_coverage_pct": attack_summary.get("attack_window_coverage_pct"),
        "attack_window_mean_frame_coverage_pct": attack_summary.get("attack_window_mean_frame_coverage_pct"),
        "first_detection_latency_frames": attack_summary.get("first_detection_latency_frames"),
        "mean_detection_latency_frames": attack_summary.get("mean_detection_latency_frames"),
        "late_detection_count": attack_summary.get("late_detection_count"),
        "missed_attack_windows": attack_summary.get("missed_attack_windows"),
        "duplicate_event_count": first_present(metrics.get("duplicate_event_count"), accounting.get("duplicate_event_count")),
        "incident_card_count": first_present(metrics.get("incident_card_count"), digest.get("incident_card_count"), accounting.get("incident_card_coverage_detail", {}).get("incident_cards_written")),
        "incident_card_coverage": first_present(metrics.get("incident_card_coverage"), accounting.get("incident_card_coverage")),
        "eidos_compression_ratio": first_present(metrics.get("eidos_compression_ratio"), digest.get("eidos_compression_ratio"), benchmark_row.get("eidos_compression_ratio")),
        "best_external_compression_baseline": best_name,
        "best_external_compression_ratio": best_ratio,
        "runtime_seconds": first_present(metrics.get("runtime_seconds"), digest.get("runtime_seconds"), get_nested(manifest, "device", "runtime_seconds"), benchmark_row.get("runtime_seconds")),
        "fps": first_present(metrics.get("frames_per_second"), digest.get("frames_per_second"), get_nested(manifest, "device", "frames_per_second"), benchmark_row.get("frames_per_second")),
        "selected_device": first_present(get_nested(manifest, "device", "selected_device"), "unknown"),
        "cuda_available": first_present(get_nested(manifest, "device", "cuda_available"), "unknown"),
        "crash_hit_count": first_present(crash_scan.get("crash_hit_count"), metrics.get("crash_hit_count"), get_nested(digest, "crash_scan", "crash_hit_count")),
        "git_commit": first_present(get_nested(manifest, "git", "commit"), digest.get("git_commit"), "unknown"),
        "config_hash": first_present(get_nested(manifest, "config", "config_hash_sha256"), metrics.get("config_hash_sha256")),
        "missing_artifacts": ";".join(missing_artifacts),
    }


def build_failure_cases(
    *,
    row: Dict[str, Any],
    diagnostics: Sequence[Dict[str, Any]],
    confirmed_fp_events: Sequence[Dict[str, Any]],
    useful_suppressed: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    duplicate_count = parse_int(row.get("duplicate_event_count")) or 0
    confirmed_count = parse_int(row.get("confirmed_event_count")) or 0
    deduped_count = parse_int(row.get("deduped_event_count")) or 0
    fp_count = parse_int(row.get("false_positives")) or 0
    fp10k = parse_float(row.get("false_positives_per_10k_frames")) or 0.0
    alert_spam_reasons: List[str] = []
    if fp_count > 0:
        alert_spam_reasons.append(f"{fp_count} confirmed false positive(s)")
    if fp10k > 0:
        alert_spam_reasons.append(f"{fp10k:.6g} false positives per 10k frames")
    if deduped_count and confirmed_count > max(10, deduped_count * 2):
        alert_spam_reasons.append(f"{confirmed_count} confirmed events versus {deduped_count} deduped events")
    duplicate_noise = [
        event
        for event in confirmed_fp_events
        if false_positive_classification(event) == "likely_duplicate_noise" or (parse_int(event.get("component_count")) or 0) > 1
    ]
    return {
        "missed_attack_windows": [item for item in diagnostics if item.get("missed")],
        "late_detections": [item for item in diagnostics if (parse_float(item.get("detection_latency")) or 0.0) > 0],
        "fully_benign_false_positives": [
            event for event in confirmed_fp_events if false_positive_classification(event) == "fully_benign"
        ],
        "duplicate_noise_clusters": duplicate_noise,
        "duplicate_event_count": duplicate_count,
        "suppressed_useful_detections": list(useful_suppressed),
        "alert_spam_reasons": alert_spam_reasons,
    }


def load_run(run_path: Path, repo_root: Path = REPO_ROOT) -> LoadedRun:
    resolved = resolve_repo_path(run_path, repo_root)
    if not resolved.is_dir():
        raise FileNotFoundError(f"run folder does not exist: {run_path}")
    artifacts, missing, errors, benchmark_rows = load_artifacts(resolved)
    event_summary = artifacts.get("event_summary", {})
    event_report = artifacts.get("event_confirmation_report", {})
    calibration_report = artifacts.get("sentinel_calibration_v1", {})
    precision_ledger = artifacts.get("precision_ledger", {})
    label_windows = list(event_summary.get("label_windows") or [])
    if not label_windows and precision_ledger.get("attack_window_diagnostics"):
        label_windows = [
            {
                "start_frame": item.get("start_frame"),
                "end_frame": item.get("end_frame"),
                "label_distribution": item.get("label_distribution", {}),
            }
            for item in precision_ledger.get("attack_window_diagnostics", [])
        ]
    confirmed_events = list(first_present(event_summary.get("confirmed_events"), event_report.get("confirmed_events"), []) or [])
    diagnostics = attack_window_diagnostics(label_windows, confirmed_events)
    attack_summary = summarize_attack_windows(diagnostics)
    confirmed_fps = confirmed_false_positive_events(confirmed_events, label_windows)
    taxonomy = taxonomy_for_events(confirmed_fps)
    row = extract_comparison_row(
        run_path=resolved,
        artifacts=artifacts,
        missing_artifacts=missing,
        benchmark_rows=benchmark_rows,
        attack_summary=attack_summary,
    )
    useful_suppressed = useful_suppressed_events(event_report, label_windows)
    calibration_useful_suppressed = [
        event
        for event in calibration_report.get("suppressed_events", [])
        if event_overlaps_attack_or_has_attack_frames(event, label_windows)
    ]
    useful_suppressed.extend(calibration_useful_suppressed)
    failures = build_failure_cases(
        row=row,
        diagnostics=diagnostics,
        confirmed_fp_events=confirmed_fps,
        useful_suppressed=useful_suppressed,
    )
    return LoadedRun(
        path=resolved,
        run_name=resolved.name,
        artifacts=artifacts,
        missing_artifacts=missing,
        artifact_errors=errors,
        benchmark_rows=list(benchmark_rows),
        comparison_row=row,
        attack_window_diagnostics=diagnostics,
        false_positive_taxonomy=taxonomy,
        ledger_false_positive_taxonomy=ledger_taxonomy(precision_ledger),
        failure_cases=failures,
    )


def metric(row: Dict[str, Any], key: str, default: float) -> float:
    parsed = parse_float(row.get(key))
    return parsed if parsed is not None else default


def mode_tie_score(policy: str, mode: str) -> int:
    if policy == "precision_first":
        order = {"low_noise": 4, "balanced": 3, "high_recall": 2, "off": 1}
    elif policy == "recall_first":
        order = {"high_recall": 4, "balanced": 3, "low_noise": 2, "off": 1}
    else:
        order = {"balanced": 4, "low_noise": 3, "high_recall": 2, "off": 1}
    return order.get(str(mode), 0)


def recommendation_sort_key(row: Dict[str, Any], policy: str) -> Tuple[float, ...]:
    mode = str(row.get("confirmation_mode", "unknown"))
    non_off = 1.0 if mode != "off" else 0.0
    tie = float(mode_tie_score(policy, mode))
    fp10k = metric(row, "false_positives_per_10k_frames", 1_000_000_000.0)
    precision = metric(row, "precision", -1.0)
    recall = metric(row, "recall", -1.0)
    f1 = metric(row, "f1", -1.0)
    coverage = metric(row, "attack_window_coverage_pct", -1.0)
    if policy == "precision_first":
        return (-fp10k, precision, recall, coverage, f1, non_off, tie)
    if policy == "recall_first":
        return (recall, coverage, precision, -fp10k, f1, non_off, tie)
    return (f1, -fp10k, precision, recall, coverage, non_off, tie)


def recommend_mode(rows: Sequence[Dict[str, Any]], policy: str) -> Dict[str, Any]:
    if not rows:
        raise ValueError("no runs available for recommendation")
    max_coverage = max(metric(row, "attack_window_coverage_pct", 0.0) for row in rows)
    coverage_floor = max_coverage * 0.5
    eligible = list(rows)
    eligibility_note = "all runs eligible"
    if policy == "precision_first":
        eligible = [
            row
            for row in rows
            if metric(row, "recall", 0.0) > 0.0 and metric(row, "attack_window_coverage_pct", 0.0) >= coverage_floor
        ]
        eligibility_note = (
            f"precision_first required nonzero recall and attack-window coverage >= {coverage_floor:.6g}% "
            f"(50% of the best observed coverage, {max_coverage:.6g}%)."
        )
        if not eligible:
            eligible = list(rows)
            eligibility_note += " No run met that floor, so all runs were ranked as a fallback."
    ranked = sorted(eligible, key=lambda row: recommendation_sort_key(row, policy), reverse=True)
    recommended = ranked[0]
    off_row = next((row for row in rows if str(row.get("confirmation_mode")) == "off"), None)
    tradeoffs = tradeoff_notes(recommended, off_row, rows)
    return {
        "generated_at_utc": utc_now(),
        "recommendation_policy": policy,
        "recommended_mode": recommended.get("confirmation_mode"),
        "recommended_calibration_enabled": recommended.get("calibration_enabled"),
        "recommended_calibration_version": recommended.get("calibration_version"),
        "recommended_display_mode": display_mode(recommended),
        "recommended_run_path": recommended.get("run_path"),
        "recommended_metrics": {key: recommended.get(key) for key in CSV_COLUMNS if key not in ("missing_artifacts",)},
        "eligibility_note": eligibility_note,
        "control_mode_note": (
            "The off mode is included as the no-confirmation control. It is only selected when it strictly wins "
            "the policy ranking after non-off tie-breaks."
        ),
        "ranking_reason": ranking_reason(policy, recommended),
        "tradeoffs": tradeoffs,
        "ranked_modes": [
            {
                "rank": index,
                "confirmation_mode": row.get("confirmation_mode"),
                "calibration_enabled": row.get("calibration_enabled"),
                "calibration_version": row.get("calibration_version"),
                "display_mode": display_mode(row),
                "run_path": row.get("run_path"),
                "score_tuple": list(recommendation_sort_key(row, policy)),
                "precision": row.get("precision"),
                "recall": row.get("recall"),
                "f1": row.get("f1"),
                "false_positives_per_10k_frames": row.get("false_positives_per_10k_frames"),
                "attack_window_coverage_pct": row.get("attack_window_coverage_pct"),
                "crash_hit_count": row.get("crash_hit_count"),
            }
            for index, row in enumerate(sorted(rows, key=lambda item: recommendation_sort_key(item, policy), reverse=True), start=1)
        ],
    }


def ranking_reason(policy: str, row: Dict[str, Any]) -> str:
    mode = row.get("confirmation_mode")
    if policy == "precision_first":
        return (
            f"{mode} ranked highest by lowest FP/10k first, then precision, recall, and attack-window coverage."
        )
    if policy == "recall_first":
        return f"{mode} ranked highest by recall and attack-window coverage, then precision."
    return f"{mode} ranked highest by F1, with FP/10k as the first tie-breaker."


def tradeoff_notes(recommended: Dict[str, Any], off_row: Optional[Dict[str, Any]], rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    best_recall = max(metric(row, "recall", -1.0) for row in rows)
    best_coverage = max(metric(row, "attack_window_coverage_pct", -1.0) for row in rows)
    notes: Dict[str, Any] = {
        "versus_best_recall": {
            "recommended_recall": recommended.get("recall"),
            "best_recall": best_recall,
            "recall_delta": (
                metric(recommended, "recall", 0.0) - best_recall if best_recall >= 0 else None
            ),
        },
        "versus_best_attack_window_coverage": {
            "recommended_coverage_pct": recommended.get("attack_window_coverage_pct"),
            "best_coverage_pct": best_coverage,
            "coverage_delta": (
                metric(recommended, "attack_window_coverage_pct", 0.0) - best_coverage if best_coverage >= 0 else None
            ),
        },
    }
    if off_row:
        notes["versus_off_control"] = {
            "off_false_positives_per_10k_frames": off_row.get("false_positives_per_10k_frames"),
            "recommended_false_positives_per_10k_frames": recommended.get("false_positives_per_10k_frames"),
            "fp10k_delta": metric(recommended, "false_positives_per_10k_frames", 0.0) - metric(off_row, "false_positives_per_10k_frames", 0.0),
            "off_confirmed_event_count": off_row.get("confirmed_event_count"),
            "recommended_confirmed_event_count": recommended.get("confirmed_event_count"),
            "confirmed_event_delta": metric(recommended, "confirmed_event_count", 0.0) - metric(off_row, "confirmed_event_count", 0.0),
            "off_f1": off_row.get("f1"),
            "recommended_f1": recommended.get("f1"),
            "f1_delta": metric(recommended, "f1", 0.0) - metric(off_row, "f1", 0.0),
        }
    return notes


def format_metric(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    parsed = parse_float(value)
    if parsed is not None:
        return f"{parsed:.6g}"
    return str(value)


def md_escape(value: Any) -> str:
    return format_metric(value).replace("|", "\\|")


def display_mode(row: Dict[str, Any]) -> str:
    mode = str(row.get("confirmation_mode", "unknown"))
    enabled = str(row.get("calibration_enabled")).lower() in {"true", "1", "yes"}
    if enabled:
        return f"{mode} + {row.get('calibration_version') or 'calibrated'}"
    return f"{mode} (uncalibrated)"


def markdown_event_ref(event: Dict[str, Any]) -> str:
    return "`{event}` {start}-{end}".format(
        event=md_escape(first_present(event.get("event_id"), event.get("candidate_id"), "unknown")),
        start=md_escape(first_present(event.get("start_frame"), get_nested(event, "score_detail", "start_frame"))),
        end=md_escape(first_present(event.get("end_frame"), get_nested(event, "score_detail", "end_frame"))),
    )


def write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_COLUMNS})


def write_comparison_report(path: Path, rows: Sequence[Dict[str, Any]], recommendation: Dict[str, Any], loaded_runs: Sequence[LoadedRun]) -> None:
    recommended = recommendation.get("recommended_metrics", {})
    off_row = next((row for row in rows if str(row.get("confirmation_mode")) == "off"), None)
    drive_or_missing = any(run.missing_artifacts or run.artifact_errors for run in loaded_runs)
    lines = [
        "# CICIDS/WebAttacks Confirmation Mode Comparison",
        "",
        "This report compares saved labeled-domain proof receipts only. It does not modify Eidos core behavior.",
        "",
        "## Recommendation",
        "",
        f"- Recommended confirmation/calibration view: `{recommendation.get('recommended_display_mode', recommendation.get('recommended_mode'))}`",
        f"- Recommendation policy: `{recommendation.get('recommendation_policy')}`",
        f"- Reason: {recommendation.get('ranking_reason')}",
        f"- Run folder: `{recommendation.get('recommended_run_path')}`",
        "",
        "## Decision Matrix",
        "",
        "| mode | calibration | raw | merged | deduped | pre-confirmed | confirmed | suppressed | raw FP/10k | pre FP/10k | calibrated FP/10k | precision | recall | F1 | attack windows covered | first latency | missed | crash hits |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {calibration} | {raw} | {merged} | {deduped} | {pre_confirmed} | {confirmed} | {suppressed} | {raw_fp10k} | {pre_fp10k} | {cal_fp10k} | {precision} | {recall} | {f1} | {coverage} | {latency} | {missed} | {crash} |".format(
                mode=md_escape(display_mode(row)),
                calibration=md_escape(row.get("calibration_enabled")),
                raw=md_escape(row.get("raw_event_count")),
                merged=md_escape(row.get("merged_event_count")),
                deduped=md_escape(row.get("deduped_event_count")),
                pre_confirmed=md_escape(row.get("pre_calibration_confirmed_event_count")),
                confirmed=md_escape(row.get("confirmed_event_count")),
                suppressed=md_escape(row.get("calibration_suppressed_event_count")),
                raw_fp10k=md_escape(row.get("raw_false_positives_per_10k_frames")),
                pre_fp10k=md_escape(row.get("pre_calibration_false_positives_per_10k_frames")),
                cal_fp10k=md_escape(row.get("calibrated_false_positives_per_10k_frames")),
                precision=md_escape(row.get("precision")),
                recall=md_escape(row.get("recall")),
                f1=md_escape(row.get("f1")),
                coverage=md_escape(row.get("attack_window_coverage_pct")),
                latency=md_escape(row.get("first_detection_latency_frames")),
                missed=md_escape(row.get("missed_attack_windows")),
                crash=md_escape(row.get("crash_hit_count")),
            )
        )
    lines.extend(["", "## Plain Answers", ""])
    lines.append(f"- Which mode performed best? `{recommendation.get('recommended_display_mode', recommendation.get('recommended_mode'))}` under `{recommendation.get('recommendation_policy')}`.")
    lines.append(f"- Why? {recommendation.get('ranking_reason')}")
    if off_row:
        notes = recommendation.get("tradeoffs", {}).get("versus_off_control", {})
        lines.append(
            "- Did it reduce false positives? Recommended FP/10k `{recommended}` versus off-control `{off}`; delta `{delta}`.".format(
                recommended=format_metric(notes.get("recommended_false_positives_per_10k_frames")),
                off=format_metric(notes.get("off_false_positives_per_10k_frames")),
                delta=format_metric(notes.get("fp10k_delta")),
            )
        )
        lines.append(
            "- What did it trade away? Confirmed events changed from off-control `{off}` to recommended `{recommended}`; F1 delta `{f1}`.".format(
                off=format_metric(notes.get("off_confirmed_event_count")),
                recommended=format_metric(notes.get("recommended_confirmed_event_count")),
                f1=format_metric(notes.get("f1_delta")),
            )
        )
    lines.append(
        "- Did it preserve attack-window detection? Recommended coverage `{coverage}%` with `{missed}` missed attack window(s).".format(
            coverage=format_metric(recommended.get("attack_window_coverage_pct")),
            missed=format_metric(recommended.get("missed_attack_windows")),
        )
    )
    crash_values = [parse_int(row.get("crash_hit_count")) or 0 for row in rows]
    lines.append(
        "- Did it keep crash scans clean? `{status}`; crash hits by mode are recorded in the matrix.".format(
            status="yes" if all(value == 0 for value in crash_values) else "no"
        )
    )
    lines.append(
        "- Is the result reproducible from artifact receipts? `{status}`; run paths, git commits, config hashes, device receipts, and missing-artifact notes are listed below.".format(
            status="yes" if not drive_or_missing else "mostly, with missing optional receipt notes"
        )
    )
    lines.extend(
        [
            "",
            "## Reproducibility Receipts",
            "",
            "| mode | calibration | version | run path | git commit | config hash | device | CUDA | missing artifacts |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {mode} | {calibration} | `{version}` | `{path}` | `{commit}` | `{config}` | {device} | {cuda} | {missing} |".format(
                mode=md_escape(display_mode(row)),
                calibration=md_escape(row.get("calibration_enabled")),
                version=md_escape(row.get("calibration_version")),
                path=md_escape(row.get("run_path")),
                commit=md_escape(row.get("git_commit")),
                config=md_escape(row.get("config_hash")),
                device=md_escape(row.get("selected_device")),
                cuda=md_escape(row.get("cuda_available")),
                missing=md_escape(row.get("missing_artifacts") or "none"),
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Core behavior changed: `false`.",
            "- This package is a decision surface for a later calibration step, not a tuning patch.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_attack_window_report(path: Path, loaded_runs: Sequence[LoadedRun]) -> None:
    lines = [
        "# Attack-Window Comparison",
        "",
        "| mode | windows | covered % | mean frame coverage % | first latency | mean latency | late detections | missed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in loaded_runs:
        row = run.comparison_row
        lines.append(
            "| {mode} | {windows} | {coverage} | {frame_coverage} | {first} | {mean} | {late} | {missed} |".format(
                mode=md_escape(display_mode(row)),
                windows=md_escape(row.get("attack_window_count")),
                coverage=md_escape(row.get("attack_window_coverage_pct")),
                frame_coverage=md_escape(row.get("attack_window_mean_frame_coverage_pct")),
                first=md_escape(row.get("first_detection_latency_frames")),
                mean=md_escape(row.get("mean_detection_latency_frames")),
                late=md_escape(row.get("late_detection_count")),
                missed=md_escape(row.get("missed_attack_windows")),
            )
        )
    lines.extend(["", "## Per-Window Details", ""])
    for run in loaded_runs:
        lines.append(f"### {display_mode(run.comparison_row)} - {run.run_name}")
        if not run.attack_window_diagnostics:
            lines.append("- No attack-window diagnostics were available.")
        for item in run.attack_window_diagnostics:
            lines.append(
                "- Window `{start}`-`{end}`: first detection `{first}`, latency `{latency}`, coverage `{coverage}%`, missed `{missed}`.".format(
                    start=md_escape(item.get("start_frame")),
                    end=md_escape(item.get("end_frame")),
                    first=md_escape(item.get("first_detection_frame")),
                    latency=md_escape(item.get("detection_latency")),
                    coverage=md_escape(item.get("coverage_percentage")),
                    missed=md_escape(item.get("missed")),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_false_positive_taxonomy(path: Path, loaded_runs: Sequence[LoadedRun]) -> None:
    categories = sorted({key for run in loaded_runs for key in run.false_positive_taxonomy})
    ledger_categories = sorted({key for run in loaded_runs for key in run.ledger_false_positive_taxonomy})
    lines = ["# False-Positive Taxonomy Summary", ""]
    if categories:
        lines.extend(
            [
                "## Confirmed-Event False Positives",
                "",
                "| mode | total | " + " | ".join(categories) + " |",
                "| --- | ---: | " + " | ".join("---:" for _ in categories) + " |",
            ]
        )
        for run in loaded_runs:
            total = sum(run.false_positive_taxonomy.values())
            cells = [str(run.false_positive_taxonomy.get(category, 0)) for category in categories]
            lines.append(f"| {md_escape(display_mode(run.comparison_row))} | {total} | " + " | ".join(cells) + " |")
    else:
        lines.append("- No confirmed-event false positives were found in the compared runs.")
    lines.extend(["", "## Precision-Ledger False Positives", ""])
    if ledger_categories:
        lines.extend(
            [
                "| mode | total | " + " | ".join(ledger_categories) + " |",
                "| --- | ---: | " + " | ".join("---:" for _ in ledger_categories) + " |",
            ]
        )
        for run in loaded_runs:
            total = sum(run.ledger_false_positive_taxonomy.values())
            cells = [str(run.ledger_false_positive_taxonomy.get(category, 0)) for category in ledger_categories]
            lines.append(f"| {md_escape(display_mode(run.comparison_row))} | {total} | " + " | ".join(cells) + " |")
    else:
        lines.append("- No precision-ledger false-positive taxonomy entries were available.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_failure_cases(path: Path, loaded_runs: Sequence[LoadedRun]) -> None:
    lines = ["# Failure Cases", ""]
    sections = (
        ("Missed Attack Windows", "missed_attack_windows"),
        ("Late Detections", "late_detections"),
        ("Fully Benign False Positives", "fully_benign_false_positives"),
        ("Duplicate/Noise Clusters", "duplicate_noise_clusters"),
        ("Modes That Suppressed Useful Detections", "suppressed_useful_detections"),
        ("Modes That Allowed Alert Spam", "alert_spam_reasons"),
    )
    for title, key in sections:
        lines.extend([f"## {title}", ""])
        wrote_any = False
        for run in loaded_runs:
            mode = display_mode(run.comparison_row)
            items = run.failure_cases.get(key, [])
            if key == "duplicate_noise_clusters" and run.failure_cases.get("duplicate_event_count"):
                lines.append(
                    f"- `{mode}` recorded `{run.failure_cases.get('duplicate_event_count')}` duplicate event(s) before dedupe."
                )
                wrote_any = True
            if not items:
                continue
            wrote_any = True
            if key == "alert_spam_reasons":
                for reason in items:
                    lines.append(f"- `{mode}`: {reason}.")
                continue
            for item in list(items)[:20]:
                if key in ("missed_attack_windows", "late_detections"):
                    lines.append(
                        "- `{mode}` window `{start}`-`{end}` latency `{latency}` missed `{missed}`.".format(
                            mode=md_escape(mode),
                            start=md_escape(item.get("start_frame")),
                            end=md_escape(item.get("end_frame")),
                            latency=md_escape(item.get("detection_latency")),
                            missed=md_escape(item.get("missed")),
                        )
                    )
                else:
                    lines.append(f"- `{mode}` {markdown_event_ref(item)} classification `{false_positive_classification(item)}`.")
            if len(items) > 20:
                lines.append(f"- `{mode}` omitted `{len(items) - 20}` additional item(s) from this compact report.")
        if not wrote_any:
            lines.append("- None found in the compared runs.")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_files(out_dir: Path) -> List[Path]:
    return sorted(path for path in out_dir.rglob("*") if path.is_file())


def write_plain_docs(
    *,
    repo_root: Path,
    out_dir: Path,
    run_date: str,
    command: str,
    rows: Sequence[Dict[str, Any]],
    recommendation: Dict[str, Any],
    drive_manifest: Optional[Dict[str, Any]] = None,
) -> None:
    drive_manifest = drive_manifest or {}
    drive_status = "copied" if drive_manifest.get("drive_copy_success") else "skipped or failed"
    drive_reason = str(drive_manifest.get("reason", "Drive copy has not run yet."))
    drive_root = str(drive_manifest.get("drive_root", "unknown"))
    drive_folder = str(drive_manifest.get("drive_run_dir", "unknown"))
    artifact_list = [relpath(path, repo_root) for path in artifact_files(out_dir)]
    docs_dir = repo_root / "docs" / "proof_runs" / run_date
    docs_dir.mkdir(parents=True, exist_ok=True)
    heading = f"## Labeled confirmation comparison -- {relpath(out_dir, repo_root)}"
    modes = ", ".join(display_mode(dict(row)) for row in rows)
    journal_body = "\n".join(
        [
            "### What happened today",
            "Built a comparison/reporting package for CICIDS/WebAttacks labeled-domain confirmation modes.",
            "",
            "### What was accomplished",
            f"- Compared confirmation modes: {modes}.",
            f"- Recommended `{recommendation.get('recommended_display_mode', recommendation.get('recommended_mode'))}` using `{recommendation.get('recommendation_policy')}`.",
            "- Wrote comparison CSV, Markdown reports, recommendation JSON, failure cases, and artifact manifests.",
            "- Kept Eidos core behavior untouched.",
            "",
            "### Tests and commands run",
            f"- `{command}` -> comparison artifacts written.",
            "",
            "### Problems encountered",
            f"- Google Drive status: {drive_status}; reason: {drive_reason}.",
            "- Missing optional source artifacts, if any, are listed in the comparison report.",
            "",
            "### What changed",
            "- tools/compare_labeled_domain_runs.py",
            "- tests/test_labeled_domain_run_comparison.py",
            f"- {relpath(out_dir, repo_root)}",
            "",
            "### What did not change",
            "Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, hippocampus memory, incident-card generation, and domain adapter math were not changed.",
            "",
            "### Artifacts generated",
            *[f"- {item}" for item in artifact_list],
            "",
            "### Google Drive archive status",
            f"- Drive root used: {drive_root}",
            f"- Drive folder used: {drive_folder}",
            f"- Files copied: {len(drive_manifest.get('files_copied', []))}",
            f"- Files skipped: {len(drive_manifest.get('files_skipped', []))}",
            f"- Reason: {drive_reason}",
            "",
            "### End-of-task summary",
            "1. Files changed: tools/compare_labeled_domain_runs.py; tests/test_labeled_domain_run_comparison.py; comparison artifacts and docs.",
            "2. Whether core behavior changed: no.",
            "3. Tests added or skipped: comparison tests added; full validation run separately.",
            f"4. Repo-root commands run: `{command}`.",
            f"5. Artifacts generated: {len(artifact_list)} files under `{relpath(out_dir, repo_root)}`.",
            "6. Plain-language analysis written: yes.",
            "7. Journal entry written: yes.",
            f"8. Google Drive copy status: {drive_status}; {drive_reason}.",
            "9. Known limitations: recommendation depends on available saved receipts; no threshold tuning was attempted.",
            "10. Follow-up tasks not implemented: Sentinel calibration or brain behavior changes.",
        ]
    )
    analysis_body = "\n".join(
        [
            "### What the task attempted",
            "The task compared saved CICIDS/WebAttacks labeled proof runs across event-confirmation modes.",
            "",
            "### Why the test matters",
            "The comparison turns separate proof runs into one decision surface for choosing a later calibration direction.",
            "",
            "### What was tested",
            "The tool read labeled metrics, event summaries, confirmation reports, precision ledgers, run manifests, crash scans, proof digests, and benchmark summaries when present.",
            "",
            "### What passed",
            f"- Recommended mode: {recommendation.get('recommended_display_mode', recommendation.get('recommended_mode'))}",
            f"- Policy: {recommendation.get('recommendation_policy')}",
            f"- Compared modes: {modes}",
            "",
            "### What failed or remains uncertain",
            "- Missing optional artifacts are disclosed in the reports instead of being silently ignored.",
            "- The comparison cannot prove a mode outside the receipts it was given.",
            "",
            "### What was saved locally",
            f"Artifacts were saved under `{relpath(out_dir, repo_root)}`.",
            "",
            "### What was saved to Google Drive",
            f"Drive status: {drive_status}; folder: {drive_folder}; reason: {drive_reason}.",
            "",
            "### What should happen next",
            "Use the comparison report to choose whether a separately gated Sentinel calibration task is worth doing.",
        ]
    )
    append_or_create(docs_dir / "codex_journal.md", heading, journal_body)
    append_or_create(docs_dir / "plain_language_test_analysis.md", heading, analysis_body)
    (out_dir / "codex_journal.md").write_text("# Codex Journal - Labeled Confirmation Comparison\n\n" + journal_body + "\n", encoding="utf-8")
    (out_dir / "plain_language_test_analysis.md").write_text(
        "# Plain-Language Test Analysis - Labeled Confirmation Comparison\n\n" + analysis_body + "\n",
        encoding="utf-8",
    )


def append_or_create(path: Path, heading: str, body: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if heading in existing:
        return
    prefix = existing.rstrip() + "\n\n" if existing.strip() else ""
    path.write_text(prefix + heading + "\n\n" + body.rstrip() + "\n", encoding="utf-8")


def build_manifest(
    *,
    command: str,
    out_dir: Path,
    rows: Sequence[Dict[str, Any]],
    loaded_runs: Sequence[LoadedRun],
    recommendation: Dict[str, Any],
    git_info: Dict[str, Any],
    drive_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    output_files = [
        {
            "path": relpath(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in artifact_files(out_dir)
        if path.name != "comparison_manifest.json"
    ]
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
        "core_behavior_boundaries": {
            "reservoir_dynamics_changed": False,
            "rls_updates_changed": False,
            "sentinel_thresholds_changed": False,
            "anomaly_policy_changed": False,
            "compression_behavior_changed": False,
            "hippocampus_memory_changed": False,
            "incident_card_generation_changed": False,
            "domain_adapter_math_changed": False,
        },
        "recommendation": recommendation,
        "runs": [
            {
                "run_path": relpath(run.path),
                "confirmation_mode": run.comparison_row.get("confirmation_mode"),
                "calibration_enabled": run.comparison_row.get("calibration_enabled"),
                "calibration_version": run.comparison_row.get("calibration_version"),
                "display_mode": display_mode(run.comparison_row),
                "missing_artifacts": run.missing_artifacts,
                "artifact_errors": run.artifact_errors,
            }
            for run in loaded_runs
        ],
        "comparison_rows": list(rows),
        "outputs": output_files,
        "drive": {
            "drive_copy_attempted": drive_manifest.get("drive_copy_attempted") if drive_manifest else None,
            "drive_copy_success": drive_manifest.get("drive_copy_success") if drive_manifest else None,
            "drive_root": drive_manifest.get("drive_root") if drive_manifest else None,
            "drive_run_dir": drive_manifest.get("drive_run_dir") if drive_manifest else None,
            "reason": drive_manifest.get("reason") if drive_manifest else None,
        },
    }


def write_package(
    *,
    out_dir: Path,
    loaded_runs: Sequence[LoadedRun],
    policy: str,
    command: str,
    repo_root: Path,
    git_info: Dict[str, Any],
    mirror_to_drive_fn: Optional[Callable[[Path, str, str], Dict[str, Any]]] = proof_helpers.mirror_to_drive,
    write_docs: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [run.comparison_row for run in loaded_runs]
    recommendation = recommend_mode(rows, policy)
    write_summary_csv(out_dir / "comparison_summary.csv", rows)
    write_json(out_dir / "recommended_confirmation_mode.json", recommendation)
    write_comparison_report(out_dir / "comparison_report.md", rows, recommendation, loaded_runs)
    write_attack_window_report(out_dir / "attack_window_comparison.md", loaded_runs)
    write_false_positive_taxonomy(out_dir / "false_positive_taxonomy_summary.md", loaded_runs)
    write_failure_cases(out_dir / "failure_cases.md", loaded_runs)
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
            run_date=run_date,
            command=command,
            rows=rows,
            recommendation=recommendation,
            drive_manifest=drive_manifest,
        )
    manifest = build_manifest(
        command=command,
        out_dir=out_dir,
        rows=rows,
        loaded_runs=loaded_runs,
        recommendation=recommendation,
        git_info=git_info,
        drive_manifest=drive_manifest,
    )
    write_json(out_dir / "comparison_manifest.json", manifest)
    if mirror_to_drive_fn is not None:
        proof_helpers.copy_selected_to_drive(
            out_dir,
            drive_manifest,
            (
                out_dir / "drive_manifest.json",
                out_dir / "comparison_manifest.json",
                out_dir / "codex_journal.md",
                out_dir / "plain_language_test_analysis.md",
            ),
        )
    return {
        "rows": rows,
        "recommendation": recommendation,
        "drive_manifest": drive_manifest,
        "manifest": manifest,
        "out_dir": str(out_dir),
    }


def run(
    args: argparse.Namespace,
    *,
    repo_root: Path = REPO_ROOT,
    mirror_to_drive_fn: Optional[Callable[[Path, str, str], Dict[str, Any]]] = proof_helpers.mirror_to_drive,
    write_docs: bool = True,
) -> Dict[str, Any]:
    out_dir = resolve_repo_path(args.out, repo_root)
    run_paths = [resolve_repo_path(path, repo_root) for path in args.runs]
    loaded_runs = [load_run(path, repo_root=repo_root) for path in run_paths]
    command = command_text([sys.executable, "tools/compare_labeled_domain_runs.py", "--runs", *[relpath(path, repo_root) for path in run_paths], "--out", relpath(out_dir, repo_root), "--recommendation-policy", args.recommendation_policy])
    git_info = proof_helpers.collect_git_info(repo_root)
    return write_package(
        out_dir=out_dir,
        loaded_runs=loaded_runs,
        policy=args.recommendation_policy,
        command=command,
        repo_root=repo_root,
        git_info=git_info,
        mirror_to_drive_fn=mirror_to_drive_fn,
        write_docs=write_docs,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run(args)
    recommendation = result["recommendation"]
    print(f"wrote comparison package: {relpath(Path(result['out_dir']))}")
    print(f"recommended confirmation mode: {recommendation.get('recommended_mode')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
