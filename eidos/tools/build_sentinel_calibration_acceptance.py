"""Build a Sentinel calibration v1 acceptance package.

This tool is reporting-only. It reads saved uncalibrated and calibrated
CICIDS/WebAttacks labeled proof receipts, groups them by sample family, and
writes an approve/decline/needs-more-evidence package for the Month 1 false
positive control gate. It does not change reservoir dynamics, RLS updates,
Sentinel thresholds, anomaly policy, compression behavior, hippocampus memory,
incident-card generation, or domain adapter math.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import compare_labeled_domain_runs as compare_runs
from tools import run_proof_baseline as proof_helpers


ACCEPTANCE_FILES = (
    "calibration_v1_acceptance.json",
    "calibration_v1_acceptance.md",
    "calibration_v1_acceptance_summary.csv",
    "guardrail_summary.csv",
    "acceptance_manifest.json",
    "drive_manifest.json",
    "codex_journal.md",
    "plain_language_test_analysis.md",
)
SUMMARY_COLUMNS = (
    "sample",
    "sample_mode",
    "frames_processed",
    "uncalibrated_run_path",
    "calibrated_run_path",
    "confirmation_mode",
    "calibration_enabled",
    "calibration_version",
    "uncalibrated_confirmed_events",
    "pre_calibration_confirmed_events",
    "post_calibration_confirmed_events",
    "suppressed_events",
    "uncalibrated_fp10k",
    "pre_calibration_fp10k",
    "post_calibration_fp10k",
    "fp10k_delta_vs_uncalibrated",
    "uncalibrated_recall",
    "post_calibration_recall",
    "post_calibration_precision",
    "post_calibration_f1",
    "attack_window_coverage_pct",
    "first_detection_latency_frames",
    "missed_attack_windows",
    "crash_hit_count",
    "guardrails_passed",
    "raw_pre_post_metrics_visible",
    "drive_status_explicit",
    "drive_copy_success",
    "status",
)
GUARDRAIL_COLUMNS = (
    "sample",
    "raw_pre_post_metrics_visible",
    "calibration_guardrails_passed",
    "fp10k_not_worse",
    "fp10k_reduced_when_baseline_had_fp",
    "recall_not_below_uncalibrated",
    "transition_attack_window_coverage_100",
    "transition_first_detection_latency_zero",
    "transition_missed_windows_zero",
    "crash_hits_zero",
    "raw_artifacts_visible",
    "drive_status_explicit",
    "passed",
)


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
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("SAMPLE", "UNCALIBRATED_RUN", "CALIBRATED_RUN"),
        required=True,
        help="A sample name plus uncalibrated and calibrated artifact folders. Repeat for each sample family.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output acceptance artifact folder.")
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


def parse_float(value: Any) -> Optional[float]:
    if value in (None, "", "NA", "NaN", "nan"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


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


def read_json_file(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        return {"_read_error": str(exc)}
    return data if isinstance(data, dict) else {"items": data}


def read_drive_manifest(run_path: Path) -> Dict[str, Any]:
    return read_json_file(run_path / "drive_manifest.json")


def artifact_files(out_dir: Path) -> List[Path]:
    return sorted(path for path in out_dir.rglob("*") if path.is_file())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_block(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    block = data.get(key)
    return block if isinstance(block, dict) else {}


def calibrated_metric_view(calibrated: compare_runs.LoadedRun, name: str) -> Dict[str, Any]:
    metrics = calibrated.artifacts.get("labeled_metrics", {})
    calibration = calibrated.artifacts.get("sentinel_calibration_v1", {})
    ledger = calibrated.artifacts.get("calibrated_precision_ledger", {})
    if name == "before":
        return first_present(
            metric_block(metrics, "pre_calibration_confirmed_event_metrics"),
            metric_block(calibration, "before_metrics"),
            get_nested(ledger, "before_after_metrics", "before"),
            {},
        )
    return first_present(
        metric_block(metrics, "calibrated_event_metrics"),
        metric_block(calibration, "after_metrics"),
        get_nested(ledger, "before_after_metrics", "after"),
        {},
    )


def required_metrics_visible(row: Dict[str, Any]) -> bool:
    required = (
        row.get("uncalibrated_fp10k"),
        row.get("pre_calibration_fp10k"),
        row.get("post_calibration_fp10k"),
        row.get("pre_calibration_confirmed_events"),
        row.get("post_calibration_confirmed_events"),
        row.get("suppressed_events"),
    )
    return all(value is not None for value in required)


def compare_values_not_worse(after: Any, before: Any) -> Optional[bool]:
    after_float = parse_float(after)
    before_float = parse_float(before)
    if after_float is None or before_float is None:
        return None
    return after_float <= before_float


def compare_values_not_below(after: Any, before: Any) -> Optional[bool]:
    after_float = parse_float(after)
    before_float = parse_float(before)
    if after_float is None or before_float is None:
        return None
    return after_float >= before_float


def drive_status_explicit(drive: Dict[str, Any]) -> bool:
    if not drive:
        return False
    return (
        "drive_copy_attempted" in drive
        and "drive_copy_success" in drive
        and bool(first_present(drive.get("drive_root"), "unknown"))
        and bool(first_present(drive.get("drive_run_dir"), "unknown"))
        and bool(first_present(drive.get("reason"), "unknown"))
    )


def build_pair_result(
    *,
    sample: str,
    uncalibrated: compare_runs.LoadedRun,
    calibrated: compare_runs.LoadedRun,
) -> Dict[str, Any]:
    uncal_row = uncalibrated.comparison_row
    cal_row = calibrated.comparison_row
    cal_before = calibrated_metric_view(calibrated, "before")
    cal_after = calibrated_metric_view(calibrated, "after")
    calibration = calibrated.artifacts.get("sentinel_calibration_v1", {})
    cal_metrics = calibrated.artifacts.get("labeled_metrics", {})
    cal_event_summary = calibrated.artifacts.get("event_summary", {})
    cal_drive = read_drive_manifest(calibrated.path)
    attack_window_count = parse_float(cal_row.get("attack_window_count")) or 0.0
    uncal_fp10k = first_present(uncal_row.get("false_positives_per_10k_frames"), uncal_row.get("calibrated_false_positives_per_10k_frames"))
    pre_fp10k = first_present(
        cal_before.get("false_positives_per_10k_frames"),
        cal_row.get("pre_calibration_false_positives_per_10k_frames"),
    )
    post_fp10k = first_present(
        cal_after.get("false_positives_per_10k_frames"),
        cal_row.get("calibrated_false_positives_per_10k_frames"),
        cal_row.get("false_positives_per_10k_frames"),
    )
    fp_delta = (
        parse_float(post_fp10k) - parse_float(uncal_fp10k)
        if parse_float(post_fp10k) is not None and parse_float(uncal_fp10k) is not None
        else None
    )
    row = {
        "sample": sample,
        "sample_mode": cal_row.get("sample_mode"),
        "frames_processed": cal_row.get("frames_processed"),
        "uncalibrated_run_path": relpath(uncalibrated.path),
        "calibrated_run_path": relpath(calibrated.path),
        "confirmation_mode": cal_row.get("confirmation_mode"),
        "calibration_enabled": cal_row.get("calibration_enabled"),
        "calibration_version": cal_row.get("calibration_version"),
        "uncalibrated_confirmed_events": uncal_row.get("confirmed_event_count"),
        "pre_calibration_confirmed_events": first_present(
            cal_row.get("pre_calibration_confirmed_event_count"),
            cal_metrics.get("pre_calibration_confirmed_events"),
            get_nested(calibration, "counts", "pre_calibration_confirmed_events"),
        ),
        "post_calibration_confirmed_events": first_present(
            cal_row.get("post_calibration_confirmed_event_count"),
            cal_metrics.get("post_calibration_confirmed_events"),
            get_nested(calibration, "counts", "post_calibration_confirmed_events"),
        ),
        "suppressed_events": first_present(
            cal_row.get("calibration_suppressed_event_count"),
            cal_metrics.get("calibration_suppressed_events"),
            get_nested(calibration, "counts", "suppressed_events"),
        ),
        "uncalibrated_fp10k": uncal_fp10k,
        "pre_calibration_fp10k": pre_fp10k,
        "post_calibration_fp10k": post_fp10k,
        "fp10k_delta_vs_uncalibrated": fp_delta,
        "uncalibrated_recall": uncal_row.get("recall"),
        "post_calibration_recall": first_present(cal_after.get("recall"), cal_row.get("recall")),
        "post_calibration_precision": first_present(cal_after.get("precision"), cal_row.get("precision")),
        "post_calibration_f1": first_present(cal_after.get("f1"), cal_row.get("f1")),
        "attack_window_count": cal_row.get("attack_window_count"),
        "attack_window_coverage_pct": cal_row.get("attack_window_coverage_pct"),
        "first_detection_latency_frames": cal_row.get("first_detection_latency_frames"),
        "missed_attack_windows": cal_row.get("missed_attack_windows"),
        "crash_hit_count": cal_row.get("crash_hit_count"),
        "guardrails_passed": get_nested(calibration, "guardrails", "passed"),
        "raw_event_count": cal_row.get("raw_event_count"),
        "merged_event_count": cal_row.get("merged_event_count"),
        "deduped_event_count": cal_row.get("deduped_event_count"),
        "drive_status_explicit": drive_status_explicit(cal_drive),
        "drive_copy_success": cal_drive.get("drive_copy_success"),
        "drive_root": cal_drive.get("drive_root"),
        "drive_run_dir": cal_drive.get("drive_run_dir"),
        "drive_reason": cal_drive.get("reason"),
        "suppressed_reason_counts": get_nested(calibration, "suppressed_reason_counts") or {},
        "calibration_config_hash": first_present(
            cal_metrics.get("calibration_config_hash_sha256"),
            calibration.get("config_hash_sha256"),
            get_nested(calibrated.artifacts.get("run_manifest", {}), "sentinel_calibration_v1", "config_hash_sha256"),
        ),
    }
    fp_not_worse = compare_values_not_worse(row.get("post_calibration_fp10k"), row.get("uncalibrated_fp10k"))
    baseline_fp = parse_float(row.get("uncalibrated_fp10k"))
    post_fp = parse_float(row.get("post_calibration_fp10k"))
    fp_reduced = True if baseline_fp is not None and baseline_fp <= 0 else (post_fp is not None and baseline_fp is not None and post_fp < baseline_fp)
    recall_not_below = compare_values_not_below(row.get("post_calibration_recall"), row.get("uncalibrated_recall"))
    if attack_window_count <= 0:
        recall_not_below = True
    checks = {
        "raw_pre_post_metrics_visible": required_metrics_visible(row),
        "calibration_guardrails_passed": parse_bool(row.get("guardrails_passed")) is True,
        "fp10k_not_worse": fp_not_worse is True,
        "fp10k_reduced_when_baseline_had_fp": fp_reduced is True,
        "recall_not_below_uncalibrated": recall_not_below is True,
        "transition_attack_window_coverage_100": (
            attack_window_count <= 0
            or parse_float(row.get("attack_window_coverage_pct")) == 100.0
        ),
        "transition_first_detection_latency_zero": (
            attack_window_count <= 0
            or parse_float(row.get("first_detection_latency_frames")) == 0.0
        ),
        "transition_missed_windows_zero": (
            attack_window_count <= 0
            or parse_float(row.get("missed_attack_windows")) == 0.0
        ),
        "crash_hits_zero": parse_float(row.get("crash_hit_count")) == 0.0,
        "raw_artifacts_visible": all(row.get(name) is not None for name in ("raw_event_count", "merged_event_count", "deduped_event_count")),
        "drive_status_explicit": row.get("drive_status_explicit") is True,
    }
    row["raw_pre_post_metrics_visible"] = checks["raw_pre_post_metrics_visible"]
    row["status"] = "passed" if all(checks.values()) else "failed"
    return {"row": row, "checks": checks, "passed": all(checks.values())}


def build_recommendation(pair_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not pair_results:
        return {
            "decision": "declined",
            "recommended_baseline": None,
            "reason": "No calibrated/un-calibrated receipt pairs were provided.",
        }
    failed = [item for item in pair_results if not item.get("passed")]
    if failed:
        return {
            "decision": "declined",
            "recommended_baseline": "balanced + sentinel_calibration_v1",
            "reason": "One or more calibration acceptance guardrails failed.",
            "failed_samples": [item["row"].get("sample") for item in failed],
        }
    rows = [item["row"] for item in pair_results]
    reduced_samples = [
        row.get("sample")
        for row in rows
        if (parse_float(row.get("uncalibrated_fp10k")) or 0.0) > 0.0
        and parse_float(row.get("post_calibration_fp10k")) is not None
        and parse_float(row.get("post_calibration_fp10k")) < (parse_float(row.get("uncalibrated_fp10k")) or 0.0)
    ]
    transition_rows = [row for row in rows if parse_float(row.get("attack_window_count")) and parse_float(row.get("attack_window_count")) > 0]
    has_transition_guard = bool(transition_rows)
    has_benign_pressure = any((parse_float(row.get("attack_window_count")) or 0.0) == 0.0 for row in rows)
    if not reduced_samples or not has_transition_guard:
        return {
            "decision": "needs_more_evidence",
            "recommended_baseline": "balanced + sentinel_calibration_v1",
            "reason": "Guardrails passed, but the package does not yet include enough FP-reduction and transition evidence.",
            "fp_reduction_samples": reduced_samples,
        }
    return {
        "decision": "approved",
        "recommended_baseline": "balanced + sentinel_calibration_v1",
        "promotion_scope": "proof-stage baseline candidate for Month 1 false-positive control",
        "reason": (
            "All provided calibrated receipts passed guardrails; calibrated balanced reduced FP/10k where the "
            "uncalibrated balanced baseline had confirmed false positives while preserving transition attack-window detection."
        ),
        "fp_reduction_samples": reduced_samples,
        "transition_samples": [row.get("sample") for row in transition_rows],
        "benign_pressure_samples": [row.get("sample") for row in rows if (parse_float(row.get("attack_window_count")) or 0.0) == 0.0],
    }


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    parsed = parse_float(value)
    if parsed is not None:
        return f"{parsed:.6g}"
    return str(value)


def write_acceptance_md(path: Path, package: Dict[str, Any]) -> None:
    recommendation = package.get("recommendation", {})
    rows = package.get("summary_rows", [])
    guardrails = package.get("guardrail_rows", [])
    lines = [
        "# Sentinel Calibration v1 Acceptance",
        "",
        "This is a Month 1 / Week 2 proof-stage false-positive control gate. It reads saved receipts only.",
        "",
        "## Decision",
        "",
        f"- Decision: `{recommendation.get('decision')}`",
        f"- Recommended baseline: `{recommendation.get('recommended_baseline')}`",
        f"- Scope: `{recommendation.get('promotion_scope', 'not approved')}`",
        f"- Reason: {recommendation.get('reason')}",
        "",
        "## Summary",
        "",
        "| sample | frames | uncal FP/10k | pre FP/10k | post FP/10k | suppressed | precision | recall | F1 | crash | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {sample} | {frames} | {uncal_fp} | {pre_fp} | {post_fp} | {suppressed} | {precision} | {recall} | {f1} | {crash} | {status} |".format(
                sample=row.get("sample"),
                frames=fmt(row.get("frames_processed")),
                uncal_fp=fmt(row.get("uncalibrated_fp10k")),
                pre_fp=fmt(row.get("pre_calibration_fp10k")),
                post_fp=fmt(row.get("post_calibration_fp10k")),
                suppressed=fmt(row.get("suppressed_events")),
                precision=fmt(row.get("post_calibration_precision")),
                recall=fmt(row.get("post_calibration_recall")),
                f1=fmt(row.get("post_calibration_f1")),
                crash=fmt(row.get("crash_hit_count")),
                status=row.get("status"),
            )
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "| sample | metrics visible | guardrails | FP not worse | recall held | coverage 100 | latency 0 | missed 0 | crash 0 | Drive explicit | passed |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in guardrails:
        lines.append(
            "| {sample} | `{metrics}` | `{guard}` | `{fp}` | `{recall}` | `{coverage}` | `{latency}` | `{missed}` | `{crash}` | `{drive}` | `{passed}` |".format(
                sample=row.get("sample"),
                metrics=row.get("raw_pre_post_metrics_visible"),
                guard=row.get("calibration_guardrails_passed"),
                fp=row.get("fp10k_not_worse"),
                recall=row.get("recall_not_below_uncalibrated"),
                coverage=row.get("transition_attack_window_coverage_100"),
                latency=row.get("transition_first_detection_latency_zero"),
                missed=row.get("transition_missed_windows_zero"),
                crash=row.get("crash_hits_zero"),
                drive=row.get("drive_status_explicit"),
                passed=row.get("passed"),
            )
        )
    lines.extend(
        [
            "",
            "## What This Approves",
            "",
            "- `balanced + sentinel_calibration_v1` is approved only as a proof-stage baseline candidate when the decision is `approved`.",
            "- This does not approve production tuning.",
            "- This does not change Eidos core behavior.",
            "- Raw, merged, deduped, pre-calibration, post-calibration, and suppressed-event views remain reviewable.",
            "",
            "## What It Does Not Prove",
            "",
            "- It does not prove general performance outside these saved CICIDS/WebAttacks receipts.",
            "- It does not create a 10k transition receipt; that sample is infeasible for this CSV because transition sampling needs a 50/50 benign/attack split and only 2,180 attack rows are available.",
            "- The `natural2k` receipt is a benign-only pressure check, not an attack-detection proof.",
            "",
            "## 90-Day Plan Alignment",
            "",
            "- Month 1 / Week 2: false-positive control and calibrated Sentinel modes.",
            "- No Month 2 or Month 3 feature expansion was attempted.",
            "- Next 90-day step remains broader baselines and proof-report automation, not new brain layers.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


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
    run_date: str,
    command: str,
    package: Dict[str, Any],
    drive_manifest: Dict[str, Any],
) -> None:
    docs_dir = repo_root / "docs" / "proof_runs" / run_date
    docs_dir.mkdir(parents=True, exist_ok=True)
    recommendation = package.get("recommendation", {})
    drive_status = "copied" if drive_manifest.get("drive_copy_success") else "skipped or failed"
    drive_reason = str(drive_manifest.get("reason", "unknown"))
    artifact_list = [relpath(path, repo_root) for path in artifact_files(out_dir)]
    heading = f"## Sentinel calibration v1 acceptance -- {relpath(out_dir, repo_root)}"
    journal_body = "\n".join(
        [
            "### What happened today",
            "Built a Month 1 / Week 2 acceptance package for Sentinel calibration v1.",
            "",
            "### What was accomplished",
            f"- Decision: {recommendation.get('decision')}.",
            f"- Recommended baseline: {recommendation.get('recommended_baseline')}.",
            "- Compared saved uncalibrated balanced receipts against calibrated balanced receipts by sample family.",
            "- Wrote guardrail summaries, acceptance Markdown, JSON, CSV, and manifest receipts.",
            "- Stayed within the 90-day proof plan false-positive-control track.",
            "",
            "### Tests and commands run",
            f"- `{command}` -> acceptance artifacts written.",
            "",
            "### Problems encountered",
            f"- Google Drive status: {drive_status}; reason: {drive_reason}.",
            "",
            "### What changed",
            "- tools/build_sentinel_calibration_acceptance.py",
            "- tests/test_sentinel_calibration_acceptance.py",
            f"- {relpath(out_dir, repo_root)}",
            "",
            "### What did not change",
            "Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, hippocampus memory, incident-card generation, and domain adapter math were not changed.",
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
        ]
    )
    analysis_body = "\n".join(
        [
            "### What the task attempted",
            "This task turned Sentinel calibration v1 into a reviewable acceptance gate instead of a loose proof-run success claim.",
            "",
            "### Why the test matters",
            "The 90-day plan says Month 1 should make Eidos trustworthy by reducing false positives without destroying anomaly recall.",
            "",
            "### What was tested",
            "The acceptance package checked FP/10k reduction, recall preservation, attack-window coverage, first latency, missed windows, crash scans, raw/pre/post metric visibility, and Drive status.",
            "",
            "### What passed",
            f"- Decision: {recommendation.get('decision')}",
            f"- Recommended baseline: {recommendation.get('recommended_baseline')}",
            "",
            "### What remains uncertain",
            "- This is still CICIDS/WebAttacks receipt evidence, not proof across every domain.",
            "- The natural2k sample is benign-only pressure evidence.",
            "",
            "### What was saved locally",
            f"Artifacts were saved under `{relpath(out_dir, repo_root)}`.",
            "",
            "### What was saved to Google Drive",
            f"Drive status: {drive_status}; folder: {drive_manifest.get('drive_run_dir', 'unknown')}; reason: {drive_reason}.",
            "",
            "### What should happen next",
            "Move to broader Month 1 baselines and proof-report automation before adding Month 2 features.",
        ]
    )
    append_or_create(docs_dir / "codex_journal.md", heading, journal_body)
    append_or_create(docs_dir / "plain_language_test_analysis.md", heading, analysis_body)
    (out_dir / "codex_journal.md").write_text("# Codex Journal - Sentinel Calibration v1 Acceptance\n\n" + journal_body + "\n", encoding="utf-8")
    (out_dir / "plain_language_test_analysis.md").write_text(
        "# Plain-Language Test Analysis - Sentinel Calibration v1 Acceptance\n\n" + analysis_body + "\n",
        encoding="utf-8",
    )


def build_manifest(
    *,
    command: str,
    out_dir: Path,
    pair_results: Sequence[Dict[str, Any]],
    package: Dict[str, Any],
    git_info: Dict[str, Any],
    drive_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    output_files = [
        {
            "path": relpath(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in artifact_files(out_dir)
        if path.name != "acceptance_manifest.json"
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
        "ninety_day_plan_alignment": {
            "month": 1,
            "week": 2,
            "track": "Fix normal-stream false positives and preserve anomaly recall.",
            "features_added_beyond_week_2": False,
        },
        "recommendation": package.get("recommendation"),
        "pairs": [
            {
                "sample": item["row"].get("sample"),
                "uncalibrated_run_path": item["row"].get("uncalibrated_run_path"),
                "calibrated_run_path": item["row"].get("calibrated_run_path"),
                "passed": item.get("passed"),
            }
            for item in pair_results
        ],
        "outputs": output_files,
        "drive": {
            "drive_copy_attempted": drive_manifest.get("drive_copy_attempted"),
            "drive_copy_success": drive_manifest.get("drive_copy_success"),
            "drive_root": drive_manifest.get("drive_root"),
            "drive_run_dir": drive_manifest.get("drive_run_dir"),
            "reason": drive_manifest.get("reason"),
        },
    }


def build_package(
    *,
    pairs: Sequence[Tuple[str, Path, Path]],
    out_dir: Path,
    command: str,
    repo_root: Path = REPO_ROOT,
    mirror_to_drive_fn: Optional[Callable[[Path, str, str], Dict[str, Any]]] = proof_helpers.mirror_to_drive,
    write_docs: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_results = []
    for sample, uncal_path, cal_path in pairs:
        uncalibrated = compare_runs.load_run(uncal_path, repo_root=repo_root)
        calibrated = compare_runs.load_run(cal_path, repo_root=repo_root)
        pair_results.append(
            build_pair_result(sample=sample, uncalibrated=uncalibrated, calibrated=calibrated)
        )
    summary_rows = [item["row"] for item in pair_results]
    guardrail_rows = [
        {"sample": item["row"].get("sample"), **item["checks"], "passed": item.get("passed")}
        for item in pair_results
    ]
    recommendation = build_recommendation(pair_results)
    package = {
        "generated_at_utc": utc_now(),
        "recommendation": recommendation,
        "summary_rows": summary_rows,
        "guardrail_rows": guardrail_rows,
        "core_behavior_changed": False,
        "ninety_day_plan_alignment": {
            "month": 1,
            "week": 2,
            "purpose": "false-positive control acceptance gate",
        },
    }
    write_json(out_dir / "calibration_v1_acceptance.json", package)
    write_acceptance_md(out_dir / "calibration_v1_acceptance.md", package)
    write_csv(out_dir / "calibration_v1_acceptance_summary.csv", SUMMARY_COLUMNS, summary_rows)
    write_csv(out_dir / "guardrail_summary.csv", GUARDRAIL_COLUMNS, guardrail_rows)
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
            package=package,
            drive_manifest=drive_manifest,
        )
    git_info = proof_helpers.collect_git_info(repo_root)
    manifest = build_manifest(
        command=command,
        out_dir=out_dir,
        pair_results=pair_results,
        package=package,
        git_info=git_info,
        drive_manifest=drive_manifest,
    )
    write_json(out_dir / "acceptance_manifest.json", manifest)
    if mirror_to_drive_fn is not None:
        proof_helpers.copy_selected_to_drive(
            out_dir,
            drive_manifest,
            (
                out_dir / "drive_manifest.json",
                out_dir / "acceptance_manifest.json",
                out_dir / "codex_journal.md",
                out_dir / "plain_language_test_analysis.md",
            ),
        )
    return {
        "out_dir": str(out_dir),
        "package": package,
        "manifest": manifest,
        "drive_manifest": drive_manifest,
    }


def run(
    args: argparse.Namespace,
    *,
    repo_root: Path = REPO_ROOT,
    mirror_to_drive_fn: Optional[Callable[[Path, str, str], Dict[str, Any]]] = proof_helpers.mirror_to_drive,
    write_docs: bool = True,
) -> Dict[str, Any]:
    out_dir = resolve_repo_path(args.out, repo_root)
    pairs = [
        (
            str(sample),
            resolve_repo_path(Path(uncalibrated), repo_root),
            resolve_repo_path(Path(calibrated), repo_root),
        )
        for sample, uncalibrated, calibrated in args.pair
    ]
    command_parts = [sys.executable, "tools/build_sentinel_calibration_acceptance.py"]
    for sample, uncalibrated, calibrated in pairs:
        command_parts.extend(["--pair", sample, relpath(uncalibrated, repo_root), relpath(calibrated, repo_root)])
    command_parts.extend(["--out", relpath(out_dir, repo_root)])
    command = command_text(command_parts)
    return build_package(
        pairs=pairs,
        out_dir=out_dir,
        command=command,
        repo_root=repo_root,
        mirror_to_drive_fn=mirror_to_drive_fn,
        write_docs=write_docs,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = run(parse_args(argv))
    package = result["package"]
    recommendation = package["recommendation"]
    print(f"wrote calibration acceptance package: {relpath(Path(result['out_dir']))}")
    print(f"decision: {recommendation.get('decision')}")
    print(f"recommended baseline: {recommendation.get('recommended_baseline')}")
    return 0 if recommendation.get("decision") == "approved" else 1


if __name__ == "__main__":
    raise SystemExit(main())
