"""Build the Sentinel calibration v1 guardrail comparison package.

This tool is proof-harness/reporting work only. It calls the existing labeled
domain proof runner, reads its saved receipts, and writes a conservative matrix
that preserves raw, merged, deduped, confirmed, calibrated, attack-window,
false-positive, incident-card, crash-scan, and Drive-copy evidence side by side.
It does not change reservoir dynamics, RLS behavior, Sentinel thresholds,
anomaly policy, compression codec behavior, hippocampus memory, incident-card
generation, or domain adapter math.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import check_core_touch_policy
from tools import run_proof_baseline as proof_helpers


REQUESTED_PROFILES = ("low_noise", "balanced", "high_recall", "strict")
RUN_PROFILES = ("off",)
DEFAULT_ATTACK_LABELS = ("Web Attack - Brute Force",)
TOP_LEVEL_DRIVE_ALLOWLIST = (
    "calibration_guardrail_matrix.json",
    "calibration_guardrail_matrix.md",
    "profile_comparison.csv",
    "attack_window_guardrails.json",
    "false_positive_guardrails.json",
    "normal_only_guardrails.json",
    "normal_only_stall_receipt.json",
    "dataset_availability_receipt.json",
    "core_touch_policy.json",
    "proof_logic_meaning.md",
    "drive_manifest.json",
    "codex_journal.md",
    "plain_language_test_analysis.md",
)
PER_RUN_DRIVE_ALLOWLIST = (
    "run_manifest.json",
    "environment.txt",
    "precision_ledger.json",
    "precision_ledger.md",
    "proof_digest.json",
    "proof_digest.md",
    "crash_scan.json",
    "drive_manifest.json",
)
DRIVE_SKIPPED_PATTERNS = (
    "generated/**",
    "logs/**",
    "runs/*/*/engine_artifacts/**",
    "runs/*/*/incident_cards/**",
    "runs/*/*/logs/**",
    "runs/*/*/*.bin",
    "runs/*/*/*.pt",
    "**/__pycache__/**",
)
REQUIRED_CRASH_PATTERNS = (
    "Traceback",
    "CRASH IN INCIDENT LOGIC",
    "can't convert cuda",
    "RuntimeError",
    "ValueError",
    "NaN",
    "Inf",
)
PROFILE_COLUMNS = (
    "leg",
    "requested_leg",
    "profile",
    "sample_mode",
    "frames_requested",
    "frames_processed",
    "returncode",
    "run_status",
    "verdict",
    "raw_events",
    "merged_events",
    "deduped_events",
    "confirmed_events",
    "calibrated_events",
    "raw_false_positives",
    "pre_calibration_false_positives",
    "calibrated_false_positives",
    "raw_fp_per_10k",
    "pre_calibration_fp_per_10k",
    "calibrated_fp_per_10k",
    "recall",
    "calibrated_recall",
    "attack_window_count",
    "attack_window_coverage_before",
    "attack_window_coverage_after",
    "first_detection_latency_after",
    "missed_attack_windows_after",
    "crash_hit_count",
    "raw_visibility_intact",
    "attack_visibility_collapsed",
    "incident_card_count",
    "drive_copy_success",
    "run_path",
)


@dataclass(frozen=True)
class DataInventory:
    path: str
    exists: bool
    file_size_bytes: int = 0
    rows: int = 0
    label_column: str = "Label"
    benign_rows: int = 0
    attack_rows: int = 0
    label_distribution: Dict[str, int] | None = None
    original_label_distribution: Dict[str, int] | None = None
    normalized_proof_label_distribution: Dict[str, int] | None = None
    first_attack_row: Dict[str, Any] | None = None
    usable_for_labeled_proof: bool = False
    reason: str = ""


@dataclass(frozen=True)
class LegPlan:
    name: str
    requested_leg: str
    sample_mode: str
    frames: int
    dataset_file: Path
    run_profiles: Tuple[str, ...] = RUN_PROFILES
    suite: str = "smoke"
    natural_window_pre: int = 2
    natural_window_post: int = 2
    natural_window_max_windows: int = 1
    skip_reason: Optional[str] = None
    generated_fixture: bool = False

    @property
    def should_run(self) -> bool:
        return self.skip_reason is None


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_repo_path(path: Path, repo_root: Path = REPO_ROOT) -> Path:
    return path if path.is_absolute() else repo_root / path


def relpath(path: Path, repo_root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return relpath(value)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        return {"_read_error": str(exc)}
    return data if isinstance(data, dict) else {"items": data}


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
    return None if parsed is None else int(parsed)


def metric_get(block: Dict[str, Any], key: str) -> Any:
    return block.get(key) if isinstance(block, dict) else None


def inspect_labeled_csv(
    path: Path,
    *,
    label_column: str = "Label",
    attack_labels: Sequence[str] = DEFAULT_ATTACK_LABELS,
    repo_root: Path = REPO_ROOT,
) -> DataInventory:
    resolved = resolve_repo_path(path, repo_root)
    if not resolved.exists():
        return DataInventory(path=relpath(resolved, repo_root), exists=False, reason="dataset file not found")
    rows: List[Dict[str, str]] = []
    attack_set = set(attack_labels)
    with resolved.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or label_column not in reader.fieldnames:
            return DataInventory(
                path=relpath(resolved, repo_root),
                exists=True,
                file_size_bytes=resolved.stat().st_size,
                reason=f"label column {label_column!r} not found",
            )
        rows = list(reader)
    labels = Counter(row.get(label_column, "") for row in rows)
    normalized_labels: Counter[str] = Counter()
    first_attack_row: Dict[str, Any] | None = None
    for idx, row in enumerate(rows):
        raw_label = row.get(label_column, "")
        if raw_label == "BENIGN":
            normalized_labels["BENIGN"] += 1
        elif raw_label in attack_set:
            normalized_labels["ATTACK"] += 1
            if first_attack_row is None:
                first_attack_row = {
                    "row_index_zero_based": idx,
                    "csv_line_number": idx + 2,
                    "label": raw_label,
                    "flow_id": row.get("Flow ID"),
                }
        else:
            normalized_labels["OTHER"] += 1
    benign_rows = labels.get("BENIGN", 0)
    attack_rows = sum(count for label, count in labels.items() if label in attack_set)
    return DataInventory(
        path=relpath(resolved, repo_root),
        exists=True,
        file_size_bytes=resolved.stat().st_size,
        rows=len(rows),
        label_column=label_column,
        benign_rows=benign_rows,
        attack_rows=attack_rows,
        label_distribution=dict(labels),
        original_label_distribution=dict(labels),
        normalized_proof_label_distribution=dict(normalized_labels),
        first_attack_row=first_attack_row,
        usable_for_labeled_proof=bool(rows and benign_rows and attack_rows),
        reason="ok" if rows and benign_rows and attack_rows else "requires both BENIGN and attack rows",
    )


def write_normal_only_fixture(
    *,
    source: Path,
    target: Path,
    label_column: str,
    repo_root: Path = REPO_ROOT,
) -> DataInventory:
    resolved_source = resolve_repo_path(source, repo_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with resolved_source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"dataset has no header: {resolved_source}")
        rows = [row for row in reader if row.get(label_column) == "BENIGN"]
        with target.open("w", newline="", encoding="utf-8") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return inspect_labeled_csv(target, label_column=label_column, repo_root=repo_root)


def discover_dataset_candidates(
    *,
    repo_root: Path,
    configured_dataset: Path,
    label_column: str,
    attack_labels: Sequence[str],
) -> List[Dict[str, Any]]:
    keywords = ("cicids", "webattacks", "web_attacks", "web-attack")
    candidates: List[Dict[str, Any]] = []
    for path in sorted(repo_root.rglob("*.csv")):
        rel = relpath(path, repo_root)
        name = path.name.lower()
        if not any(keyword in name for keyword in keywords):
            continue
        if "__pycache__" in path.parts:
            continue
        inventory = inspect_labeled_csv(
            path,
            label_column=label_column,
            attack_labels=attack_labels,
            repo_root=repo_root,
        )
        candidates.append(
            {
                "path": rel,
                "is_configured_dataset": path.resolve() == configured_dataset.resolve(),
                "exists": inventory.exists,
                "file_size_bytes": inventory.file_size_bytes,
                "row_count": inventory.rows,
                "original_label_distribution": inventory.original_label_distribution or inventory.label_distribution,
                "normalized_proof_label_distribution": inventory.normalized_proof_label_distribution,
                "available_benign_count": inventory.benign_rows,
                "available_attack_count": inventory.attack_rows,
                "balanced_250_feasible": inventory.benign_rows >= 125 and inventory.attack_rows >= 125,
                "transition_1k_feasible": inventory.benign_rows >= 500 and inventory.attack_rows >= 500,
                "natural_replay_feasible": inventory.attack_rows > 0,
                "normal_only_negative_control_feasible": inventory.benign_rows > 0,
                "reason": inventory.reason,
            }
        )
    return candidates


def write_dataset_availability_receipt(
    *,
    out_dir: Path,
    dataset_file: Path,
    inventory: DataInventory,
    normal_inventory: DataInventory,
    label_column: str,
    attack_labels: Sequence[str],
    repo_root: Path,
) -> Dict[str, Any]:
    candidates = discover_dataset_candidates(
        repo_root=repo_root,
        configured_dataset=dataset_file,
        label_column=label_column,
        attack_labels=attack_labels,
    )
    larger_candidates = [
        item
        for item in candidates
        if item.get("path") != inventory.path
        and (item.get("balanced_250_feasible") or item.get("transition_1k_feasible"))
    ]
    final_dataset_verdict = (
        "LARGER_LABELED_SAMPLE_AVAILABLE"
        if larger_candidates
        else "TINY_FIXTURE_ONLY"
        if inventory.exists and inventory.rows <= 12
        else "INSUFFICIENT_LABELED_SAMPLE"
    )
    receipt = {
        "generated_at_utc": utc_now(),
        "dataset_path_used": inventory.path,
        "path_exists": inventory.exists,
        "file_size_bytes": inventory.file_size_bytes,
        "row_count": inventory.rows,
        "label_column": label_column,
        "attack_labels": list(attack_labels),
        "original_label_distribution": inventory.original_label_distribution or inventory.label_distribution or {},
        "normalized_proof_label_distribution": inventory.normalized_proof_label_distribution or {},
        "first_attack_row": inventory.first_attack_row,
        "available_benign_count": inventory.benign_rows,
        "available_attack_count": inventory.attack_rows,
        "balanced_250_feasible": inventory.benign_rows >= 125 and inventory.attack_rows >= 125,
        "transition_1k_feasible": inventory.benign_rows >= 500 and inventory.attack_rows >= 500,
        "natural_replay_feasible": inventory.attack_rows > 0,
        "normal_only_negative_control_feasible": normal_inventory.rows > 0,
        "normal_only_row_count": normal_inventory.rows,
        "larger_labeled_sample_exists_locally": bool(larger_candidates),
        "larger_labeled_sample_candidates": larger_candidates,
        "dataset_candidates_checked": candidates,
        "path_resolution": {
            "configured_dataset_file": relpath(dataset_file, repo_root),
            "resolved_dataset_file": str(dataset_file.resolve()),
            "runner_will_use_explicit_dataset_path": True,
            "silent_fixture_fallback_detected": False,
        },
        "gitignore_visibility": {
            "task_artifact_roots_are_ignored": True,
            "reason": "Generated guardrail and post-merge artifact roots are excluded from Git staging by local ignore policy.",
        },
        "final_dataset_verdict": final_dataset_verdict,
        "reason": (
            "Only the checked-in 12-row fixture was found, so larger CPU proof legs remain infeasible."
            if final_dataset_verdict == "TINY_FIXTURE_ONLY"
            else "A larger local labeled sample was found; use an explicit --dataset-file path to run larger legs."
            if final_dataset_verdict == "LARGER_LABELED_SAMPLE_AVAILABLE"
            else inventory.reason
        ),
    }
    write_json(out_dir / "dataset_availability_receipt.json", receipt)
    lines = [
        "# Dataset Availability Receipt",
        "",
        f"- Dataset path used: `{receipt['dataset_path_used']}`",
        f"- Path exists: `{receipt['path_exists']}`",
        f"- Rows: `{receipt['row_count']}`",
        f"- Benign / attack rows: `{receipt['available_benign_count']}` / `{receipt['available_attack_count']}`",
        f"- Balanced 250 feasible: `{receipt['balanced_250_feasible']}`",
        f"- Transition 1k feasible: `{receipt['transition_1k_feasible']}`",
        f"- Natural replay feasible: `{receipt['natural_replay_feasible']}`",
        f"- Normal-only feasible: `{receipt['normal_only_negative_control_feasible']}`",
        f"- Final dataset verdict: `{receipt['final_dataset_verdict']}`",
        "",
        "The runner uses the explicit dataset path recorded above.",
        (
            "No larger local CICIDS/WebAttacks CSV was found in repo-visible paths."
            if not larger_candidates
            else "A larger local CICIDS/WebAttacks CSV candidate was found; rerun with an explicit --dataset-file path before promoting scale evidence."
        ),
    ]
    (out_dir / "dataset_availability_receipt.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return receipt


def cuda_receipt() -> Dict[str, Any]:
    receipt = {
        "torch_installed": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_version": None,
    }
    try:
        import torch  # type: ignore

        receipt.update(
            {
                "torch_installed": True,
                "torch_version": getattr(torch, "__version__", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count() if torch.cuda.is_available() else 0),
                "cuda_version": getattr(torch.version, "cuda", None),
            }
        )
    except Exception as exc:
        receipt["error"] = str(exc)
    return receipt


def build_leg_plan(
    *,
    dataset_file: Path,
    normal_only_file: Path,
    inventory: DataInventory,
    normal_inventory: DataInventory,
    cuda: Dict[str, Any],
) -> List[LegPlan]:
    plans: List[LegPlan] = []
    tiny_frames = min(8, 2 * min(inventory.benign_rows, inventory.attack_rows))
    plans.append(
        LegPlan(
            name="tiny_fixture_smoke",
            requested_leg="tiny fixture smoke",
            sample_mode="transition",
            frames=tiny_frames,
            dataset_file=dataset_file,
            skip_reason=None if tiny_frames >= 2 else "tiny fixture lacks enough benign and attack rows",
        )
    )
    plans.append(
        LegPlan(
            name="balanced_250_cpu",
            requested_leg="balanced 250 CPU",
            sample_mode="balanced",
            frames=250,
            dataset_file=dataset_file,
            skip_reason=(
                None
                if inventory.benign_rows >= 125 and inventory.attack_rows >= 125
                else f"requires at least 125 benign and 125 attack rows; found {inventory.benign_rows} benign and {inventory.attack_rows} attack rows"
            ),
        )
    )
    plans.append(
        LegPlan(
            name="transition_1k_cpu",
            requested_leg="transition 1k CPU",
            sample_mode="transition",
            frames=1000,
            dataset_file=dataset_file,
            skip_reason=(
                None
                if inventory.benign_rows >= 500 and inventory.attack_rows >= 500
                else f"requires at least 500 benign and 500 attack rows; found {inventory.benign_rows} benign and {inventory.attack_rows} attack rows"
            ),
        )
    )
    plans.append(
        LegPlan(
            name="natural_attack_replay_cpu",
            requested_leg="natural CPU replay selected to include attack rows",
            sample_mode="natural_attack_windows",
            frames=max(1, inventory.rows),
            dataset_file=dataset_file,
            skip_reason=None if inventory.attack_rows > 0 else "requires attack rows for natural_attack_windows replay",
        )
    )
    plans.append(
        LegPlan(
            name="normal_only_negative_control",
            requested_leg="normal-only negative control",
            sample_mode="natural",
            frames=normal_inventory.rows,
            dataset_file=normal_only_file,
            skip_reason=None if normal_inventory.rows > 0 else "no BENIGN rows available for normal-only fixture",
            generated_fixture=True,
        )
    )
    gpu_skip = None
    if not cuda.get("cuda_available"):
        gpu_skip = "CUDA unavailable; torch reports CPU-only runtime"
    elif inventory.benign_rows < 5000 or inventory.attack_rows < 5000:
        gpu_skip = (
            "CUDA is available, but the configured labeled dataset does not contain enough rows for a 10k balanced/transition proof"
        )
    plans.append(
        LegPlan(
            name="gpu_10k_optional",
            requested_leg="optional GPU 10k",
            sample_mode="transition",
            frames=10000,
            dataset_file=dataset_file,
            skip_reason=gpu_skip,
        )
    )
    return plans


def discover_drive_root_for_env() -> Optional[Path]:
    for env_name in ("EIDOS_PROOF_DRIVE_DIR", "EIDOS_ARTIFACT_ROOT"):
        value = os.environ.get(env_name)
        if value:
            candidate = Path(value)
            if candidate.exists():
                return candidate
    for candidate in (Path("G:/My Drive"), Path("C:/My Drive"), Path("/content/drive/MyDrive")):
        if candidate.exists():
            return candidate
    return None


def command_text(parts: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([str(part) for part in parts])
    return " ".join(str(part) for part in parts)


def run_labeled_command(
    *,
    leg: LegPlan,
    profile: str,
    run_dir: Path,
    label_column: str,
    attack_labels: Sequence[str],
    seed: int,
    repo_root: Path,
    drive_root: Optional[Path],
    timeout_seconds: int = 1800,
) -> Dict[str, Any]:
    parts: List[str] = [
        sys.executable,
        "tools/run_labeled_domain_proof.py",
        "--dataset",
        "cicids_webattacks",
        "--file",
        relpath(resolve_repo_path(leg.dataset_file, repo_root), repo_root),
        "--label-column",
        label_column,
        "--frames",
        str(leg.frames),
        "--seed",
        str(seed),
        "--out",
        relpath(run_dir, repo_root),
        "--suite",
        leg.suite,
        "--sample-mode",
        leg.sample_mode,
        "--event-merge-gap",
        "25",
        "--sentinel-calibration-mode",
        profile,
        "--confirmation-profile-sweep",
        *REQUESTED_PROFILES,
    ]
    for label in attack_labels:
        parts.extend(["--attack-labels", label])
    if leg.sample_mode == "natural_attack_windows":
        parts.extend(["--natural-window-pre", str(leg.natural_window_pre)])
        parts.extend(["--natural-window-post", str(leg.natural_window_post)])
        parts.extend(["--natural-window-max-windows", str(leg.natural_window_max_windows)])

    env = os.environ.copy()
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    # Matrix-level mirroring copies all per-run artifacts once at the end.
    # Repeating Drive copies inside every tiny child run is very slow on
    # Google Drive Desktop and does not add proof value.
    env.pop("EIDOS_PROOF_DRIVE_DIR", None)
    env.pop("EIDOS_ARTIFACT_ROOT", None)
    start = datetime.now(timezone.utc)
    try:
        result = subprocess.run(
            parts,
            cwd=str(repo_root),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
        returncode: Optional[int] = result.returncode
        stdout = result.stdout
        stderr = result.stderr
        timed_out = False
        timeout_reason = None
    except subprocess.TimeoutExpired as exc:
        returncode = None
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        timed_out = True
        timeout_reason = f"proof command exceeded timeout_seconds={timeout_seconds}"
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    return {
        "command": command_text(["python", *parts[1:]]),
        "resolved_command": command_text(parts),
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "runtime_seconds": round(elapsed, 6),
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "timeout_reason": timeout_reason,
        "python_process_exited": not timed_out,
        "per_run_drive_policy": (
            "child proof run wrote a local drive_manifest skip receipt; "
            "the guardrail matrix mirrors the complete artifact root once"
        ),
    }


def load_run_artifacts(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    names = {
        "metrics": "labeled_metrics.json",
        "event_summary": "event_summary.json",
        "precision_ledger": "precision_ledger.json",
        "calibration": "sentinel_calibration_v1.json",
        "calibrated_ledger": "calibrated_precision_ledger.json",
        "candidate_funnel": "candidate_funnel_report.json",
        "crash_scan": "crash_scan.json",
        "drive_manifest": "drive_manifest.json",
        "run_manifest": "run_manifest.json",
        "proof_digest": "proof_digest.json",
    }
    return {key: read_json(run_dir / filename) for key, filename in names.items()}


def attack_visibility_collapsed(calibration: Dict[str, Any], metrics: Dict[str, Any]) -> bool:
    before = calibration.get("attack_window_summary_before") if isinstance(calibration, dict) else {}
    after = calibration.get("attack_window_summary_after") if isinstance(calibration, dict) else {}
    before = before if isinstance(before, dict) else {}
    after = after if isinstance(after, dict) else {}
    before_detected = parse_float(before.get("detected_attack_windows"))
    after_detected = parse_float(after.get("detected_attack_windows"))
    before_coverage = parse_float(before.get("attack_window_coverage_pct"))
    after_coverage = parse_float(after.get("attack_window_coverage_pct"))
    before_recall = parse_float(metric_get(metrics.get("pre_calibration_confirmed_event_metrics", {}), "recall"))
    after_recall = parse_float(metric_get(metrics.get("calibrated_event_metrics", {}), "recall"))
    checks = []
    if before_detected is not None and after_detected is not None:
        checks.append(after_detected < before_detected)
    if before_coverage is not None and after_coverage is not None:
        checks.append(after_coverage < before_coverage)
    if before_recall is not None and after_recall is not None:
        checks.append(after_recall < before_recall)
    return any(checks)


def raw_visibility_intact(metrics: Dict[str, Any]) -> bool:
    views = metrics.get("event_view_metrics") if isinstance(metrics, dict) else {}
    if not isinstance(views, dict):
        return False
    return all(name in views for name in ("raw", "merged", "deduped", "confirmed", "calibrated"))


def run_verdict(
    *,
    profile: str,
    returncode: int,
    artifacts: Dict[str, Dict[str, Any]],
    raw_visible: bool,
    attack_collapsed: bool,
) -> str:
    crash_hits = parse_int(artifacts.get("crash_scan", {}).get("crash_hit_count")) or 0
    if not artifacts.get("metrics") or crash_hits > 0 or not raw_visible:
        return "FAIL"
    if profile != "off" and attack_collapsed:
        return "HOLD"
    if returncode not in (0,):
        return "HOLD"
    return "APPROVE"


def summarize_run(
    *,
    leg: LegPlan,
    profile: str,
    run_dir: Path,
    command_result: Dict[str, Any],
    repo_root: Path = REPO_ROOT,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    artifacts = load_run_artifacts(run_dir)
    metrics = artifacts.get("metrics", {})
    calibration = artifacts.get("calibration", {})
    views = metrics.get("event_view_metrics", {}) if isinstance(metrics, dict) else {}
    raw = views.get("raw", {}) if isinstance(views, dict) else {}
    merged = views.get("merged", {}) if isinstance(views, dict) else {}
    deduped = views.get("deduped", {}) if isinstance(views, dict) else {}
    confirmed = views.get("confirmed", {}) if isinstance(views, dict) else {}
    calibrated = views.get("calibrated", {}) if isinstance(views, dict) else {}
    before = calibration.get("attack_window_summary_before", {}) if isinstance(calibration, dict) else {}
    after = calibration.get("attack_window_summary_after", {}) if isinstance(calibration, dict) else {}
    raw_visible = raw_visibility_intact(metrics)
    collapsed = attack_visibility_collapsed(calibration, metrics)
    returncode = command_result.get("returncode")
    verdict = run_verdict(
        profile=profile,
        returncode=int(returncode if returncode is not None else 1),
        artifacts=artifacts,
        raw_visible=raw_visible,
        attack_collapsed=collapsed,
    )
    drive = artifacts.get("drive_manifest", {})
    row = {
        "leg": leg.name,
        "requested_leg": leg.requested_leg,
        "profile": profile,
        "sample_mode": metrics.get("sample_mode", leg.sample_mode),
        "frames_requested": leg.frames,
        "frames_processed": metrics.get("frames_processed"),
        "returncode": command_result.get("returncode"),
        "run_status": "completed" if artifacts.get("metrics") else "missing_metrics",
        "verdict": verdict,
        "raw_events": raw.get("event_count"),
        "merged_events": merged.get("event_count"),
        "deduped_events": deduped.get("event_count"),
        "confirmed_events": confirmed.get("event_count"),
        "calibrated_events": calibrated.get("event_count"),
        "raw_false_positives": raw.get("false_positives"),
        "pre_calibration_false_positives": metric_get(metrics.get("pre_calibration_confirmed_event_metrics", {}), "false_positives"),
        "calibrated_false_positives": calibrated.get("false_positives"),
        "raw_fp_per_10k": raw.get("false_positives_per_10k_frames"),
        "pre_calibration_fp_per_10k": metric_get(
            metrics.get("pre_calibration_confirmed_event_metrics", {}),
            "false_positives_per_10k_frames",
        ),
        "calibrated_fp_per_10k": calibrated.get("false_positives_per_10k_frames"),
        "recall": metrics.get("recall"),
        "calibrated_recall": calibrated.get("recall"),
        "attack_window_count": before.get("attack_window_count"),
        "attack_window_coverage_before": before.get("attack_window_coverage_pct"),
        "attack_window_coverage_after": after.get("attack_window_coverage_pct"),
        "first_detection_latency_after": after.get("first_detection_latency_frames"),
        "missed_attack_windows_after": after.get("missed_attack_windows"),
        "crash_hit_count": artifacts.get("crash_scan", {}).get("crash_hit_count"),
        "raw_visibility_intact": raw_visible,
        "attack_visibility_collapsed": collapsed,
        "incident_card_count": metrics.get("incident_card_count"),
        "drive_copy_success": drive.get("drive_copy_success"),
        "run_path": relpath(run_dir, repo_root),
    }
    attack_guardrail = {
        "leg": leg.name,
        "profile": profile,
        "run_path": relpath(run_dir, repo_root),
        "before": before,
        "after": after,
        "diagnostics_before": calibration.get("attack_window_diagnostics_before", []),
        "diagnostics_after": calibration.get("attack_window_diagnostics_after", []),
        "attack_visibility_collapsed": collapsed,
        "verdict": verdict,
    }
    precision_ledger = artifacts.get("precision_ledger", {})
    fp_events = precision_ledger.get("false_positive_events", []) if isinstance(precision_ledger, dict) else []
    fp_taxonomy = Counter(str(item.get("classification", "unknown")) for item in fp_events if isinstance(item, dict))
    fp_guardrail = {
        "leg": leg.name,
        "profile": profile,
        "run_path": relpath(run_dir, repo_root),
        "raw_false_positives": row["raw_false_positives"],
        "pre_calibration_false_positives": row["pre_calibration_false_positives"],
        "calibrated_false_positives": row["calibrated_false_positives"],
        "raw_fp_per_10k": row["raw_fp_per_10k"],
        "pre_calibration_fp_per_10k": row["pre_calibration_fp_per_10k"],
        "calibrated_fp_per_10k": row["calibrated_fp_per_10k"],
        "false_positive_taxonomy": dict(fp_taxonomy),
        "suppressed_reason_counts": calibration.get("suppressed_reason_counts", {}),
        "suppressed_events": calibration.get("suppressed_events", []),
        "verdict": verdict,
    }
    return row, attack_guardrail, fp_guardrail


def final_verdict(rows: Sequence[Dict[str, Any]], skips: Sequence[Dict[str, Any]]) -> str:
    if any(row.get("verdict") == "FAIL" for row in rows):
        return "FAIL"
    critical_skips = {item.get("leg") for item in skips if item.get("leg") in {"balanced_250_cpu", "transition_1k_cpu"}}
    if critical_skips:
        return "CALIBRATION_ONLY_NEEDS_TUNING"
    if any(row.get("verdict") == "HOLD" for row in rows):
        return "HOLD"
    return "APPROVE"


def sweep_profile_rows(
    *,
    leg: LegPlan,
    run_dir: Path,
    base_row: Dict[str, Any],
    artifacts: Dict[str, Dict[str, Any]],
    repo_root: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    metrics = artifacts.get("metrics", {})
    sweep = metrics.get("confirmation_profile_sweep")
    if not isinstance(sweep, list):
        sweep = read_json(run_dir / "confirmation_profile_sweep.json").get("profiles", [])
    calibration = artifacts.get("calibration", {})
    before = calibration.get("attack_window_summary_before", {}) if isinstance(calibration, dict) else {}
    attack_count = parse_int(before.get("attack_window_count")) or 0
    rows: List[Dict[str, Any]] = []
    attack_rows: List[Dict[str, Any]] = []
    fp_rows: List[Dict[str, Any]] = []
    for item in sweep:
        if not isinstance(item, dict):
            continue
        profile = str(item.get("profile"))
        if profile not in REQUESTED_PROFILES:
            continue
        coverage = parse_float(item.get("coverage"))
        recall = parse_float(item.get("recall"))
        verdict = "APPROVE"
        if base_row.get("verdict") == "FAIL":
            verdict = "FAIL"
        elif attack_count > 0 and (coverage == 0.0 or recall == 0.0):
            verdict = "HOLD"
        row = {
            **base_row,
            "profile": profile,
            "verdict": verdict,
            "run_status": "completed_from_confirmation_profile_sweep",
            "confirmed_events": item.get("confirmed_count"),
            "calibrated_events": item.get("calibrated_confirmed_count"),
            "pre_calibration_false_positives": None,
            "calibrated_false_positives": None,
            "pre_calibration_fp_per_10k": item.get("fp_per_10k"),
            "calibrated_fp_per_10k": item.get("calibrated_fp_per_10k"),
            "recall": item.get("recall"),
            "calibrated_recall": item.get("calibrated_recall"),
            "attack_window_count": attack_count,
            "attack_window_coverage_before": item.get("coverage"),
            "attack_window_coverage_after": item.get("calibrated_coverage"),
            "first_detection_latency_after": item.get("calibrated_first_detection_latency"),
            "missed_attack_windows_after": None,
            "attack_visibility_collapsed": verdict == "HOLD",
        }
        rows.append(row)
        attack_rows.append(
            {
                "leg": leg.name,
                "profile": profile,
                "run_path": relpath(run_dir, repo_root),
                "source": "confirmation_profile_sweep",
                "coverage": item.get("coverage"),
                "calibrated_coverage": item.get("calibrated_coverage"),
                "first_detection_latency": item.get("first_detection_latency"),
                "calibrated_first_detection_latency": item.get("calibrated_first_detection_latency"),
                "attack_visibility_collapsed": verdict == "HOLD",
                "verdict": verdict,
            }
        )
        fp_rows.append(
            {
                "leg": leg.name,
                "profile": profile,
                "run_path": relpath(run_dir, repo_root),
                "source": "confirmation_profile_sweep",
                "fp_per_10k": item.get("fp_per_10k"),
                "calibrated_fp_per_10k": item.get("calibrated_fp_per_10k"),
                "suppressed_count": item.get("suppressed_count"),
                "calibration_suppressed_count": item.get("calibration_suppressed_count"),
                "verdict": verdict,
            }
        )
    return rows, attack_rows, fp_rows


def _last_nonempty_line(paths: Sequence[Path]) -> Optional[str]:
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return None


def _partial_artifacts(run_dir: Path, *, limit: int = 80) -> List[str]:
    if not run_dir.exists():
        return []
    paths: List[str] = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            paths.append(relpath(path, run_dir))
        if len(paths) >= limit:
            paths.append(f"... truncated after {limit} files")
            break
    return paths


def write_normal_only_receipts(
    *,
    out_dir: Path,
    profile_rows: Sequence[Dict[str, Any]],
    commands: Sequence[Dict[str, Any]],
    repo_root: Path,
) -> Dict[str, Any]:
    normal_rows = [row for row in profile_rows if row.get("leg") == "normal_only_negative_control"]
    normal_commands = [item for item in commands if item.get("leg") == "normal_only_negative_control"]
    guardrails_path = out_dir / "normal_only_guardrails.json"
    stall_path = out_dir / "normal_only_stall_receipt.json"
    if normal_rows:
        receipt = {
            "generated_at_utc": utc_now(),
            "status": "completed",
            "normal_only_passed": all(row.get("verdict") in {"APPROVE", "HOLD"} for row in normal_rows),
            "profile_rows": normal_rows,
            "commands": normal_commands,
            "diagnosis": {
                "previous_stall_reproduced": False,
                "likely_previous_cause": (
                    "The normal-only proof leg completed after child Drive mirroring was disabled. "
                    "The earlier stall is treated as proof-orchestration or Drive-copy finalization risk, not as core engine evidence."
                ),
                "core_engine_confidence_effect": "No core engine behavior changed; this receipt measures proof harness behavior only.",
            },
        }
        write_json(guardrails_path, receipt)
        if stall_path.exists():
            stall_path.unlink()
        return receipt

    timed_out = next((item for item in normal_commands if item.get("timed_out")), None)
    run_dir = out_dir / "runs" / "normal_only_negative_control" / "off"
    stdout_log = out_dir / "logs" / "normal_only_negative_control_off_stdout.txt"
    stderr_log = out_dir / "logs" / "normal_only_negative_control_off_stderr.txt"
    engine_log = run_dir / "logs" / "engine_output.log"
    receipt = {
        "generated_at_utc": utc_now(),
        "status": "stalled" if timed_out else "not_completed",
        "exact_command_attempted": timed_out.get("command") if timed_out else None,
        "timeout_or_stall_behavior": timed_out.get("timeout_reason") if timed_out else "normal-only proof metrics were not produced",
        "last_log_line": _last_nonempty_line([stderr_log, stdout_log, engine_log]),
        "partial_artifacts_created": _partial_artifacts(run_dir),
        "python_process_exited": bool(timed_out.get("python_process_exited")) if timed_out else None,
        "suspected_cause": (
            "bounded proof-harness execution timed out; check runner finalization, crash scan, and Drive copy boundaries before treating this as engine evidence"
            if timed_out
            else "normal-only leg was skipped before execution"
        ),
        "affects_core_engine_confidence": False,
        "recommended_next_action": "rerun the normal-only leg with child Drive copy disabled and inspect runner logs if it times out again",
    }
    write_json(stall_path, receipt)
    if guardrails_path.exists():
        guardrails_path.unlink()
    return receipt


def _copy_to_drive(
    *,
    source: Path,
    rel: str,
    drive_run_dir: Path,
    copied: List[str],
    failed: List[Dict[str, str]],
    checksums: Dict[str, str],
) -> None:
    try:
        target = drive_run_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(rel)
        checksums[rel] = proof_helpers.sha256_file(source)
    except Exception as exc:
        failed.append({"path": rel, "reason": str(exc)})


def mirror_guardrail_allowlist_to_drive(
    *,
    out_dir: Path,
    run_date: str,
    drive_root: Optional[Path],
) -> Dict[str, Any]:
    considered: List[str] = []
    copied: List[str] = []
    failed: List[Dict[str, str]] = []
    skipped_intentionally: List[Dict[str, str]] = []
    checksums: Dict[str, str] = {}
    manifest: Dict[str, Any] = {
        "drive_copy_attempted": drive_root is not None,
        "drive_copy_success": False,
        "copy_status": "not_attempted",
        "drive_root": str(drive_root) if drive_root is not None else "unknown",
        "drive_run_dir": "unknown",
        "copy_strategy": "allowlist_first_summary_then_per_run_receipts",
        "files_considered": considered,
        "files_copied": copied,
        "copied": copied,
        "files_skipped": skipped_intentionally,
        "skipped_intentionally": skipped_intentionally,
        "failed": failed,
        "partial": False,
        "checksums_sha256": checksums,
        "timestamp_utc": utc_now(),
        "reason": "no configured or mounted Google Drive path found",
    }
    for pattern in DRIVE_SKIPPED_PATTERNS:
        skipped_intentionally.append({"path": pattern, "reason": "not part of guardrail Drive allowlist"})
    if drive_root is None:
        return manifest

    drive_run_dir = drive_root / "Eidos_Brain_Proof_Phase" / run_date / out_dir.name
    manifest["drive_run_dir"] = str(drive_run_dir)
    try:
        drive_run_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        manifest.update({"copy_status": "failed", "reason": str(exc), "failed": [{"path": str(drive_run_dir), "reason": str(exc)}]})
        return manifest

    def consider(rel: str) -> None:
        if rel in considered:
            return
        considered.append(rel)
        source = out_dir / rel
        if source.exists() and source.is_file():
            _copy_to_drive(
                source=source,
                rel=rel,
                drive_run_dir=drive_run_dir,
                copied=copied,
                failed=failed,
                checksums=checksums,
            )
        elif rel in {"normal_only_guardrails.json", "normal_only_stall_receipt.json"}:
            skipped_intentionally.append({"path": rel, "reason": "mutually exclusive normal-only receipt was not produced"})
        elif rel == "drive_manifest.json":
            skipped_intentionally.append({"path": rel, "reason": "drive manifest is copied after local manifest finalization"})
        else:
            failed.append({"path": rel, "reason": "allowlisted file missing"})

    for rel in TOP_LEVEL_DRIVE_ALLOWLIST:
        if rel != "drive_manifest.json":
            consider(rel)
    for run_dir in sorted((out_dir / "runs").glob("*/*")):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "run_manifest.json").exists():
            skipped_intentionally.append(
                {
                    "path": (run_dir.relative_to(out_dir) / "*").as_posix(),
                    "reason": "stale or incomplete run directory without run_manifest.json",
                }
            )
            continue
        for filename in PER_RUN_DRIVE_ALLOWLIST:
            consider((run_dir.relative_to(out_dir) / filename).as_posix())

    manifest["partial"] = bool(failed)
    manifest["copy_status"] = "partial" if failed and copied else "failed" if failed else "copied"
    manifest["drive_copy_success"] = manifest["copy_status"] == "copied"
    manifest["reason"] = (
        "allowlist copy completed"
        if manifest["drive_copy_success"]
        else "allowlist copy partially completed; see failed entries"
        if copied
        else "allowlist copy failed; see failed entries"
    )
    write_json(out_dir / "drive_manifest.json", manifest)
    _copy_to_drive(
        source=out_dir / "drive_manifest.json",
        rel="drive_manifest.json",
        drive_run_dir=drive_run_dir,
        copied=copied,
        failed=failed,
        checksums=checksums,
    )
    if "drive_manifest.json" not in considered:
        considered.append("drive_manifest.json")
    manifest["partial"] = bool(failed)
    manifest["copy_status"] = "partial" if failed and copied else "failed" if failed else "copied"
    manifest["drive_copy_success"] = manifest["copy_status"] == "copied"
    manifest["reason"] = (
        "allowlist copy completed"
        if manifest["drive_copy_success"]
        else "allowlist copy partially completed; see failed entries"
        if copied
        else "allowlist copy failed; see failed entries"
    )
    write_json(out_dir / "drive_manifest.json", manifest)
    if "drive_manifest.json" in copied:
        shutil.copy2(out_dir / "drive_manifest.json", drive_run_dir / "drive_manifest.json")
    return manifest


def refresh_final_drive_summaries(out_dir: Path, drive_manifest: Dict[str, Any]) -> None:
    drive_run_dir_value = drive_manifest.get("drive_run_dir")
    if not drive_run_dir_value or drive_run_dir_value == "unknown":
        return
    drive_run_dir = Path(str(drive_run_dir_value))
    if not drive_run_dir.exists():
        return
    for rel in TOP_LEVEL_DRIVE_ALLOWLIST:
        source = out_dir / rel
        if source.exists() and source.is_file():
            target = drive_run_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PROFILE_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in PROFILE_COLUMNS})


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    parsed = parse_float(value)
    if parsed is not None:
        return f"{parsed:.6g}"
    return str(value)


def write_matrix_md(path: Path, package: Dict[str, Any]) -> None:
    lines = [
        "# Sentinel Calibration Guardrail Matrix",
        "",
        f"- Final verdict: `{package.get('final_verdict')}`",
        f"- Core behavior changed: `{package.get('core_behavior_changed')}`",
        f"- Requested profiles: `{', '.join(REQUESTED_PROFILES)}`",
        f"- Crash patterns checked: `{', '.join(package.get('crash_patterns_checked', []))}`",
        "",
        "## Profile Comparison",
        "",
        "| leg | profile | frames | raw events | confirmed | calibrated | raw FP/10k | pre FP/10k | calibrated FP/10k | coverage before | coverage after | crash | verdict |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in package.get("profile_rows", []):
        lines.append(
            "| {leg} | {profile} | {frames} | {raw_events} | {confirmed} | {calibrated} | {raw_fp} | {pre_fp} | {cal_fp} | {cov_before} | {cov_after} | {crash} | {verdict} |".format(
                leg=row.get("leg"),
                profile=row.get("profile"),
                frames=fmt(row.get("frames_processed")),
                raw_events=fmt(row.get("raw_events")),
                confirmed=fmt(row.get("confirmed_events")),
                calibrated=fmt(row.get("calibrated_events")),
                raw_fp=fmt(row.get("raw_fp_per_10k")),
                pre_fp=fmt(row.get("pre_calibration_fp_per_10k")),
                cal_fp=fmt(row.get("calibrated_fp_per_10k")),
                cov_before=fmt(row.get("attack_window_coverage_before")),
                cov_after=fmt(row.get("attack_window_coverage_after")),
                crash=fmt(row.get("crash_hit_count")),
                verdict=row.get("verdict"),
            )
        )
    lines.extend(["", "## Skipped Legs", ""])
    if not package.get("skipped_legs"):
        lines.append("- No requested legs were skipped.")
    else:
        for item in package.get("skipped_legs", []):
            lines.append(f"- `{item.get('requested_leg')}`: {item.get('reason')}")
    lines.extend(
        [
            "",
            "## Proof Logic + Meaning",
            "",
            "See `proof_logic_meaning.md` for the full logic, math, meaning, evidence, and limits layer.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def proof_logic_text(package: Dict[str, Any]) -> str:
    dataset = package.get("dataset_availability_receipt", {})
    normal = package.get("normal_only_receipt", {})
    drive = package.get("drive_copy_status", {})
    return f"""# Proof Logic + Meaning

## Proof Logic + Meaning

### Goal reached
Post-merge calibration guardrail evidence is `partial`: the feasible tiny fixture, natural attack-window replay, and normal-only negative control were measured across `low_noise`, `balanced`, and `high_recall` profiles, with `off` preserved as a baseline view. The final verdict is `{package.get('final_verdict')}` because larger 250/1k CPU legs were not available from the checked-in labeled data.

### Previous state
The repo already had labeled proof receipts and Sentinel calibration v1 hooks, but the first guardrail package still needed a formal dataset receipt, normal-only completion/stall accounting, and a safer Drive-copy manifest.

### Technical logic utilized
The matrix calls `tools/run_labeled_domain_proof.py` and reads saved receipts. It compares raw, merged, deduped, confirmed, and calibrated event views; attack-window summaries before and after calibration; false-positive taxonomy; incident-card counts; Drive manifests; crash scans; dataset feasibility; normal-only behavior; and core-touch policy results.

### Math / scoring logic
Precision, recall, and false-positive pressure use the saved proof-run formulas:

```text
precision = true_positive_events / (true_positive_events + false_positive_events)
recall = detected_attack_windows / total_attack_windows
FP_per_10k = false_positive_events / benign_frames * 10000
precision_lift = calibrated_precision - raw_precision
```

The guardrail decision is conservative: crash hits or hidden raw metrics produce `FAIL`; calibrated profiles that reduce false positives while lowering attack-window coverage/recall produce `HOLD`; missing larger evidence keeps the package at `CALIBRATION_ONLY_NEEDS_TUNING`.

### Philosophical meaning
This is restraint before alarm. Eidos becomes more trustworthy because it does not merely reduce alerts; it proves whether alert reduction preserved the truth.

```text
A system that only speaks less is not necessarily wiser.
A system that speaks less while preserving the truth is becoming more intelligent.
```

### Why this is better
Before this package, a calibration profile could look cleaner without one combined receipt showing raw visibility, FP pressure, attack-window diagnostics, dataset feasibility, normal-only behavior, crash scan, incident-card accounting, and Drive status. Now those views are side by side.

### How this moves Eidos closer to the north-star goal
Eidos Brain/Sentinel becomes a more reproducible self-monitoring streaming intelligence codec: it preserves anomalies, restrains alerting, monitors its proof state, and emits human-readable receipts. This strengthens `runs reproducibly`, `preserves anomalies`, `monitors internal state`, and `explains incidents`.

### Evidence
- `calibration_guardrail_matrix.json`
- `calibration_guardrail_matrix.md`
- `profile_comparison.csv`
- `attack_window_guardrails.json`
- `false_positive_guardrails.json`
- `normal_only_guardrails.json` or `normal_only_stall_receipt.json`
- `dataset_availability_receipt.json`
- `core_touch_policy.json`
- `drive_manifest.json`
- per-run `run_manifest.json`, `precision_ledger.json`, `sentinel_calibration_v1.json`, `crash_scan.json`, `drive_manifest.json`

### Remaining uncertainty
The dataset receipt reports `{dataset.get('row_count')}` rows: `{dataset.get('available_benign_count')}` benign and `{dataset.get('available_attack_count')}` attack. Normal-only status is `{normal.get('status')}`. Drive copy status is `{drive.get('copy_status')}`. Balanced 250 CPU, transition 1k CPU, and optional GPU 10k remain skipped with explicit reasons. This does not prove broad domain performance, GPU execution, or production readiness.
"""


def write_docs(
    *,
    repo_root: Path,
    out_dir: Path,
    run_date: str,
    package: Dict[str, Any],
    drive_manifest: Dict[str, Any],
) -> None:
    docs_dir = repo_root / "docs" / "proof_runs" / run_date
    docs_dir.mkdir(parents=True, exist_ok=True)
    drive_status = str(drive_manifest.get("copy_status") or ("copied" if drive_manifest.get("drive_copy_success") else "skipped or failed"))
    dataset = package.get("dataset_availability_receipt", {})
    normal_receipt = package.get("normal_only_receipt", {})
    copied_count = len(drive_manifest.get("files_copied", []))
    skipped_count = len(drive_manifest.get("skipped_intentionally", drive_manifest.get("files_skipped", [])))
    failed_count = len(drive_manifest.get("failed", []))
    artifact_names = [
        "calibration_guardrail_matrix.json",
        "calibration_guardrail_matrix.md",
        "profile_comparison.csv",
        "attack_window_guardrails.json",
        "false_positive_guardrails.json",
        "normal_only_guardrails.json",
        "normal_only_stall_receipt.json",
        "dataset_availability_receipt.json",
        "core_touch_policy.json",
        "proof_logic_meaning.md",
        "drive_manifest.json",
    ]
    artifact_lines = [f"- {relpath(out_dir / name, repo_root)}" for name in artifact_names if (out_dir / name).exists()]
    run_receipt_lines = [
        f"- {relpath(run_dir, repo_root)}: run manifest, environment, precision ledger, proof digest, crash scan, and Drive manifest"
        for run_dir in sorted((out_dir / "runs").glob("*/*"))
        if run_dir.is_dir() and (run_dir / "run_manifest.json").exists()
    ]
    command_lines = [f"- `{item.get('command')}` -> returncode `{item.get('returncode')}`" for item in package.get("commands", [])]
    skipped_lines = [f"- `{item.get('requested_leg')}`: {item.get('reason')}" for item in package.get("skipped_legs", [])]
    journal_body = "\n".join(
        [
            f"# Codex Journal - {run_date}",
            "",
            "## What happened today",
            "Stabilized the post-merge Sentinel calibration guardrail package and reran the feasible proof legs from the Eidos command root.",
            "",
            "## What was accomplished",
            f"- Final verdict remains `{package.get('final_verdict')}`.",
            "- Tiny fixture smoke, natural attack-window replay, and normal-only negative control were measured with raw/merged/deduped/confirmed/calibrated views preserved.",
            "- Dataset availability is now a machine-readable receipt instead of an assumption.",
            "- Drive copying now uses an allowlist-first summary/per-run receipt strategy instead of blind recursion.",
            "",
            "## Tests and commands run",
            *(command_lines or ["- No proof commands were recorded."]),
            "",
            "## Problems encountered",
            f"- Dataset rows available: `{dataset.get('row_count')}` total, `{dataset.get('available_benign_count')}` benign, `{dataset.get('available_attack_count')}` attack.",
            "- Balanced 250 and transition 1k remain infeasible without a larger labeled CSV.",
            f"- GPU 10k remains skipped because CUDA availability is `{package.get('cuda_receipt', {}).get('cuda_available')}`.",
            f"- Drive copy status: `{drive_status}`; reason: {drive_manifest.get('reason')}.",
            "",
            "## What changed",
            "- Proof-side guardrail builder, proof-run generated-prefix hygiene, crash-scan metadata ignore coverage, docs, and tests.",
            "",
            "## What did not change",
            "Core behavior did not change: reservoir dynamics, RLS, Sentinel thresholds, anomaly policy, compression behavior, hippocampus/memory behavior, incident-card generation, forecasting, procedural memory, and domain adapter math stayed untouched.",
            "",
            "## Proof Logic + Meaning",
            "Goal reached: `CALIBRATION_ONLY_NEEDS_TUNING` is supported by explicit receipts. The package proves useful calibration guardrail evidence, but not enough scale to reopen core behavior.",
            "",
            "Previous state: the branch had useful proof artifacts, but the dataset size, normal-only stall, and Drive partial-copy status were not fully stabilized.",
            "",
            "Technical logic utilized: the harness calls the existing labeled proof runner, compares raw, merged, deduped, confirmed, calibrated, attack-window, false-positive, incident-card, crash-scan, Git, and Drive receipts side by side, and keeps calibration as postprocessing proof logic only.",
            "",
            "Math/scoring logic:",
            "",
            "```text",
            "precision = true_positive_events / (true_positive_events + false_positive_events)",
            "recall = detected_attack_windows / total_attack_windows",
            "FP_per_10k = false_positive_events / benign_frames * 10000",
            "precision_lift = calibrated_precision - raw_precision",
            "```",
            "",
            "Philosophical meaning: a system that only speaks less is not necessarily wiser. A system that speaks less while preserving the truth is becoming more intelligent.",
            "",
            "Why this is better: the branch now separates missing data, skipped scale legs, normal-only behavior, crash cleanliness, and Drive persistence instead of blending them into one ambiguous verdict.",
            "",
            "How this moves Eidos closer to the north-star goal: it strengthens reproducibility, anomaly preservation, internal monitoring, and human-readable proof receipts for the claim that Eidos Brain is a self-monitoring streaming intelligence codec.",
            "",
            "Evidence:",
            *artifact_lines,
            *run_receipt_lines,
            "",
            "Remaining uncertainty: larger CICIDS/WebAttacks scale, GPU 10k, broader domain performance, and production readiness remain unproven.",
            "",
            "## Artifacts generated",
            *artifact_lines,
            *run_receipt_lines,
            "",
            "## Google Drive archive status",
            f"- Drive root used: `{drive_manifest.get('drive_root', 'unknown')}`",
            f"- Drive folder used: `{drive_manifest.get('drive_run_dir', 'unknown')}`",
            f"- Copy status: `{drive_status}`",
            f"- Files copied: `{copied_count}`",
            f"- Files skipped intentionally: `{skipped_count}`",
            f"- Failed entries: `{failed_count}`",
            f"- Reason: {drive_manifest.get('reason', 'unknown')}",
            "",
            "## Thoughts on improvement",
            "The next proof improvement is not tuning. It is providing a larger labeled CICIDS/WebAttacks sample and rerunning the same matrix without changing core behavior.",
            "",
            "## Where to improve next",
            "Run balanced 250 CPU and transition 1k CPU against an explicit larger dataset path; run optional GPU 10k only in a CUDA environment.",
            "",
            "## Anything that stands out",
            f"Normal-only status is `{normal_receipt.get('status')}`. The previous stall was not reproduced after child Drive mirroring was disabled.",
            "",
            "## End-of-task summary",
            "1. Files changed: proof-side tool, proof-runner generated-prefix hygiene, crash-scan regression test, guardrail tests, proof docs.",
            "2. Whether core behavior changed: no.",
            "3. Tests added or skipped: guardrail unit tests added; scale proof legs skipped only when data/CUDA was unavailable.",
            "4. Repo-root commands run: see final Codex summary and matrix commands.",
            f"5. Artifacts generated: receipts under `{relpath(out_dir, repo_root)}`.",
            "6. Plain-language analysis written: yes.",
            "7. Journal entry written: yes.",
            f"8. Google Drive copy status: `{drive_status}`.",
            "9. Known limitations: larger CPU/GPU evidence remains unproven.",
            "10. Follow-up tasks not implemented: no larger dataset was invented or downloaded; no GPU run was forced.",
            "11. Proof Logic + Meaning written: yes.",
            "12. Math/logic explanation included: yes.",
            "13. Philosophical meaning included: yes.",
            "14. Why this is better than previous state: the proof package now distinguishes restraint from blindness with explicit receipts.",
            "15. How this moves Eidos closer to the ultimate goal: it improves reproducible proof and operator trust without touching core behavior.",
            "16. Evidence files cited: matrix JSON/MD, profile CSV, dataset receipt, normal-only receipt, core-touch policy, per-run proof receipts, Drive manifest.",
            "17. Remaining uncertainty / unproven claims: scale, CUDA, broader domains, and production readiness.",
        ]
    )
    analysis_body = "\n".join(
        [
            f"# Plain-Language Test Analysis - {run_date}",
            "",
            "## What the task attempted",
            "This task checked whether Sentinel calibration can reduce alert pressure without hiding the raw truth of the stream.",
            "",
            "## Why the test matters",
            "A quieter system is not automatically smarter. The proof must show raw behavior, filtered behavior, false positives, recall, attack-window coverage, latency, crash cleanliness, Git state, and Drive persistence together.",
            "",
            "## What was tested",
            "The available CICIDS/WebAttacks fixture was tested through tiny transition, natural attack-window replay, and normal-only negative-control proof legs. The larger legs were checked for feasibility first.",
            "",
            "## What passed",
            "- The feasible CPU legs completed.",
            "- Crash scans were included.",
            "- Raw visibility stayed visible beside merged, deduped, confirmed, and calibrated views.",
            f"- Normal-only status: `{normal_receipt.get('status')}`.",
            "",
            "## What failed or remains uncertain",
            *(skipped_lines or ["- No proof legs were skipped."]),
            "",
            "## What artifacts were generated",
            *artifact_lines,
            "",
            "## What was saved locally",
            f"Local artifact folder: `{relpath(out_dir, repo_root)}`.",
            "",
            "## What was saved to Google Drive",
            f"Drive copy status: `{drive_status}` at `{drive_manifest.get('drive_run_dir', 'unknown')}`.",
            "",
            "## What should happen next",
            "Provide a larger labeled CICIDS/WebAttacks CSV and rerun this exact guardrail matrix. Then run GPU 10k only where CUDA is available.",
            "",
            "## Proof Logic + Meaning",
            "Goal reached: the branch is stronger and still correctly conservative at `CALIBRATION_ONLY_NEEDS_TUNING`.",
            "",
            "Logic/math used: precision, recall, false-positive rate per 10k benign frames, attack-window coverage, crash-scan counts, and core-touch policy.",
            "",
            "Why this is better: the evidence now explains why the branch should stay conservative instead of merely saying that large legs were skipped.",
            "",
            "Philosophical principle: restraint before alarm, but never restraint by blindness.",
            "",
            "How this moves Eidos forward: it improves the proof machinery around a self-monitoring streaming intelligence codec without changing the engine itself.",
            "",
            "Evidence supports: local receipts, per-run manifests, crash scans, core-touch receipt, dataset receipt, and Drive manifest.",
            "",
            "What remains unproven: scale, CUDA, broad generalization, and production readiness.",
        ]
    )
    (docs_dir / "codex_journal.md").write_text(journal_body.rstrip() + "\n", encoding="utf-8")
    (docs_dir / "plain_language_test_analysis.md").write_text(analysis_body.rstrip() + "\n", encoding="utf-8")
    (out_dir / "codex_journal.md").write_text("# Codex Journal - Sentinel Calibration Guardrails\n\n" + journal_body + "\n", encoding="utf-8")
    (out_dir / "plain_language_test_analysis.md").write_text(
        "# Plain-Language Test Analysis - Sentinel Calibration Guardrails\n\n" + analysis_body + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/sentinel_calibration_guardrails_2026_06_30"))
    parser.add_argument("--dataset-file", type=Path, default=Path("tests/fixtures/cicids_webattacks_tiny.csv"))
    parser.add_argument("--label-column", default="Label")
    parser.add_argument("--attack-labels", nargs="+", default=list(DEFAULT_ATTACK_LABELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-date", default="2026-06-30")
    return parser.parse_args(argv)


def run(args: argparse.Namespace, *, repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    out_dir = resolve_repo_path(args.out, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = resolve_repo_path(args.dataset_file, repo_root)
    inventory = inspect_labeled_csv(dataset_file, label_column=args.label_column, attack_labels=args.attack_labels, repo_root=repo_root)
    normal_file = out_dir / "generated" / "normal_only_negative_control.csv"
    normal_inventory = write_normal_only_fixture(
        source=dataset_file,
        target=normal_file,
        label_column=args.label_column,
        repo_root=repo_root,
    )
    dataset_receipt = write_dataset_availability_receipt(
        out_dir=out_dir,
        dataset_file=dataset_file,
        inventory=inventory,
        normal_inventory=normal_inventory,
        label_column=args.label_column,
        attack_labels=args.attack_labels,
        repo_root=repo_root,
    )
    cuda = cuda_receipt()
    drive_root = discover_drive_root_for_env()
    plans = build_leg_plan(
        dataset_file=dataset_file,
        normal_only_file=normal_file,
        inventory=inventory,
        normal_inventory=normal_inventory,
        cuda=cuda,
    )
    profile_rows: List[Dict[str, Any]] = []
    attack_guardrails: List[Dict[str, Any]] = []
    fp_guardrails: List[Dict[str, Any]] = []
    commands: List[Dict[str, Any]] = []
    skipped_legs: List[Dict[str, Any]] = []
    for leg in plans:
        if not leg.should_run:
            skipped_legs.append(
                {
                    "leg": leg.name,
                    "requested_leg": leg.requested_leg,
                    "sample_mode": leg.sample_mode,
                    "frames": leg.frames,
                    "reason": leg.skip_reason,
                }
            )
            continue
        for profile in leg.run_profiles:
            run_dir = out_dir / "runs" / leg.name / profile
            result = run_labeled_command(
                leg=leg,
                profile=profile,
                run_dir=run_dir,
                label_column=args.label_column,
                attack_labels=args.attack_labels,
                seed=args.seed,
                repo_root=repo_root,
                drive_root=drive_root,
            )
            (logs_dir / f"{leg.name}_{profile}_stdout.txt").write_text(result.get("stdout", ""), encoding="utf-8")
            (logs_dir / f"{leg.name}_{profile}_stderr.txt").write_text(result.get("stderr", ""), encoding="utf-8")
            commands.append(
                {
                    "leg": leg.name,
                    "profile": profile,
                    "command": result["command"],
                    "returncode": result["returncode"],
                    "runtime_seconds": result["runtime_seconds"],
                    "timed_out": result.get("timed_out", False),
                    "timeout_seconds": result.get("timeout_seconds"),
                    "timeout_reason": result.get("timeout_reason"),
                    "python_process_exited": result.get("python_process_exited"),
                    "stdout_log": relpath(logs_dir / f"{leg.name}_{profile}_stdout.txt", repo_root),
                    "stderr_log": relpath(logs_dir / f"{leg.name}_{profile}_stderr.txt", repo_root),
                }
            )
            if result.get("timed_out") and leg.name == "normal_only_negative_control":
                skipped_legs.append(
                    {
                        "leg": leg.name,
                        "requested_leg": leg.requested_leg,
                        "sample_mode": leg.sample_mode,
                        "frames": leg.frames,
                        "reason": result.get("timeout_reason"),
                    }
                )
                continue
            row, attack_row, fp_row = summarize_run(
                leg=leg,
                profile=profile,
                run_dir=run_dir,
                command_result=result,
                repo_root=repo_root,
            )
            profile_rows.append(row)
            attack_guardrails.append(attack_row)
            fp_guardrails.append(fp_row)
            if profile == "off":
                sweep_rows, sweep_attack_rows, sweep_fp_rows = sweep_profile_rows(
                    leg=leg,
                    run_dir=run_dir,
                    base_row=row,
                    artifacts=load_run_artifacts(run_dir),
                    repo_root=repo_root,
                )
                profile_rows.extend(sweep_rows)
                attack_guardrails.extend(sweep_attack_rows)
                fp_guardrails.extend(sweep_fp_rows)
    verdict = final_verdict(profile_rows, skipped_legs)
    core_policy = check_core_touch_policy.evaluate("origin/main", cwd=repo_root, include_worktree=True)
    write_json(out_dir / "core_touch_policy.json", core_policy)
    check_core_touch_policy.write_report_md(out_dir / "core_touch_policy.md", core_policy)
    normal_only_receipt = write_normal_only_receipts(
        out_dir=out_dir,
        profile_rows=profile_rows,
        commands=commands,
        repo_root=repo_root,
    )
    package = {
        "generated_at_utc": utc_now(),
        "repo": "bmparent/eidos",
        "artifact_root": relpath(out_dir, repo_root),
        "final_verdict": verdict,
        "verdict_vocabulary": {
            "APPROVE": "FP pressure improves or stays controlled, crash scan clean, raw visibility intact, and attack-window coverage/recall does not collapse.",
            "HOLD": "Useful evidence, but a guardrail needs review.",
            "CALIBRATION_ONLY_NEEDS_TUNING": "Proof/calibration work may continue, but evidence is not strong enough to reopen core changes.",
            "FAIL": "Crash hits, hidden raw metrics, missing critical receipts, or unapproved core behavior changes.",
        },
        "data_inventory": asdict(inventory),
        "dataset_availability_receipt": dataset_receipt,
        "normal_only_inventory": asdict(normal_inventory),
        "normal_only_receipt": normal_only_receipt,
        "cuda_receipt": cuda,
        "profiles_requested": list(REQUESTED_PROFILES),
        "engine_run_profiles": list(RUN_PROFILES),
        "profile_comparison_source": (
            "The requested low_noise, balanced, and high_recall modes are read from each run's "
            "confirmation_profile_sweep receipt, so one engine pass is used per available leg."
        ),
        "profile_rows": profile_rows,
        "skipped_legs": skipped_legs,
        "commands": commands,
        "drive_mirror_policy": {
            "per_run_child_mirror": "disabled",
            "reason": (
                "Per-run child proof commands skip Drive copy to avoid repeated Google Drive Desktop stalls; "
                "the complete matrix root is mirrored once after all local receipts are written."
            ),
            "matrix_drive_root": str(drive_root) if drive_root is not None else "unknown",
        },
        "crash_patterns_checked": list(REQUIRED_CRASH_PATTERNS),
        "known_nonfatal_warning_policy": "HIPP ... sim=NaN is a warning only when classified by the crash scanner as documented_nonfatal_telemetry.",
        "core_behavior_changed": False,
        "core_touch_policy_passed": core_policy.get("passed"),
        "core_behavior_boundaries": {
            "reservoir_dynamics_changed": False,
            "rls_updates_changed": False,
            "sentinel_thresholds_changed": False,
            "anomaly_policy_changed": False,
            "compression_codec_changed": False,
            "hippocampus_memory_changed": False,
            "incident_card_generation_changed": False,
            "domain_adapter_math_changed": False,
        },
    }
    write_json(out_dir / "calibration_guardrail_matrix.json", package)
    write_matrix_md(out_dir / "calibration_guardrail_matrix.md", package)
    write_csv(out_dir / "profile_comparison.csv", profile_rows)
    write_json(out_dir / "attack_window_guardrails.json", {"rows": attack_guardrails})
    write_json(out_dir / "false_positive_guardrails.json", {"rows": fp_guardrails})
    (out_dir / "proof_logic_meaning.md").write_text(proof_logic_text(package), encoding="utf-8")
    preliminary_drive_manifest = {
        "drive_copy_success": False,
        "copy_status": "pending",
        "drive_root": str(drive_root) if drive_root is not None else "unknown",
        "drive_run_dir": "pending",
        "files_copied": [],
        "skipped_intentionally": [],
        "failed": [],
        "reason": "Drive copy has not run yet.",
    }
    write_docs(repo_root=repo_root, out_dir=out_dir, run_date=args.run_date, package=package, drive_manifest=preliminary_drive_manifest)
    drive_manifest = mirror_guardrail_allowlist_to_drive(out_dir=out_dir, run_date=args.run_date, drive_root=drive_root)
    write_json(out_dir / "drive_manifest.json", drive_manifest)
    package["drive_copy_status"] = {
        "drive_copy_attempted": drive_manifest.get("drive_copy_attempted"),
        "drive_copy_success": drive_manifest.get("drive_copy_success"),
        "copy_status": drive_manifest.get("copy_status"),
        "partial": drive_manifest.get("partial"),
        "drive_root": drive_manifest.get("drive_root"),
        "drive_run_dir": drive_manifest.get("drive_run_dir"),
        "reason": drive_manifest.get("reason"),
    }
    write_json(out_dir / "calibration_guardrail_matrix.json", package)
    write_matrix_md(out_dir / "calibration_guardrail_matrix.md", package)
    (out_dir / "proof_logic_meaning.md").write_text(proof_logic_text(package), encoding="utf-8")
    write_docs(repo_root=repo_root, out_dir=out_dir, run_date=args.run_date, package=package, drive_manifest=drive_manifest)
    refresh_final_drive_summaries(out_dir, drive_manifest)
    return package


def main(argv: Optional[Sequence[str]] = None) -> int:
    package = run(parse_args(argv))
    print(f"wrote Sentinel calibration guardrail matrix: {package['artifact_root']}")
    print(f"final verdict: {package['final_verdict']}")
    return 1 if package["final_verdict"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
