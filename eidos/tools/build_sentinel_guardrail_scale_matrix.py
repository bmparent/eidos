"""Build the larger labeled Sentinel guardrail scale matrix.

This is proof-harness work only. It discovers bounded local CICIDS/WebAttacks
datasets, records a registry, runs the existing labeled proof runner where data
allows, and writes conservative matrix reports. It does not change reservoir
dynamics, RLS behavior, raw Sentinel anomaly policy, compression codec behavior,
hippocampus memory behavior, or core incident-card generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from tools import build_sentinel_calibration_guardrails as guardrails
from tools import check_core_touch_policy


RUN_DATE = "2026-07-04"
ARTIFACT_ROOT = Path("artifacts/sentinel_guardrail_scale_matrix_2026_07_04")
REGISTRY_JSON = Path("docs/proof/dataset_registry.json")
REGISTRY_MD = Path("docs/proof/dataset_registry.md")
DOC_RUN_DIR = Path("docs/proof_runs/2026-07-04")
DOC_REPORT = DOC_RUN_DIR / "sentinel_guardrail_scale_matrix.md"
CURRENT_CALIBRATED_PROFILE = "low_noise"
TUNED_CALIBRATED_PROFILE = "strict"
REQUESTED_PROFILES = (CURRENT_CALIBRATED_PROFILE, "balanced", "high_recall", TUNED_CALIBRATED_PROFILE)
ENGINE_RUN_PROFILES = ("off",)
PARTIAL_RECEIPT_NAMES = ("config.json", "environment.txt", "git_commit.txt")
NORMAL_ONLY_FP_PER_10K_TARGET = 10.0
NORMAL_ONLY_FP_PER_10K_STRETCH_TARGET = 5.0
DEFAULT_ATTACK_LABELS = (
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Web Attack - Sql Injection",
)
LABEL_COLUMN_ALIASES = ("Label", " Label", "label")
SEARCH_KEYWORDS = (
    "cicids",
    "webattack",
    "webattacks",
    "web_attack",
    "web-attack",
    "workinghours",
    "iscx",
)
SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "engine_artifacts",
    "generated",
    "logs",
    "runs",
}
SCALE_COLUMNS = (
    "run_name",
    "profile",
    "dataset_path",
    "dataset_sha256",
    "sample_mode",
    "frame_count",
    "benign_frames",
    "attack_frames",
    "attack_windows",
    "raw_events",
    "merged_events",
    "deduped_events",
    "confirmed_events",
    "calibrated_events",
    "suppressed_events",
    "false_positive_events",
    "raw_false_positive_events",
    "fp_per_10k_benign_frames",
    "precision",
    "recall",
    "f1",
    "attack_window_coverage",
    "first_detection_latency",
    "crash_hits",
    "runtime_seconds",
    "fps",
    "selected_device",
    "cuda_available",
    "core_touch_result",
    "raw_visibility_intact",
    "attack_visibility_collapsed",
    "run_status",
    "verdict",
    "verdict_reason",
    "run_path",
)


@dataclass(frozen=True)
class DatasetCandidate:
    path: str
    resolved_path: str
    file_type: str
    exists: bool
    file_size_bytes: int = 0
    sha256: Optional[str] = None
    row_count: int = 0
    label_column: Optional[str] = None
    requested_label_column: str = "Label"
    raw_label_distribution: Dict[str, int] | None = None
    normalized_label_distribution: Dict[str, int] | None = None
    benign_count: int = 0
    attack_count: int = 0
    attack_labels_detected: List[str] | None = None
    first_attack_row: Dict[str, Any] | None = None
    usable_for_tiny: bool = False
    usable_for_balanced_250: bool = False
    usable_for_transition_1k: bool = False
    usable_for_natural_replay: bool = False
    usable_for_gpu_10k_rows: bool = False
    usable_for_labeled_proof: bool = False
    reason: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def relpath(path: Path, repo_root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def resolve_path(path: Path, repo_root: Path = REPO_ROOT) -> Path:
    return path if path.is_absolute() else repo_root / path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        return {"_read_error": str(exc)}
    return data if isinstance(data, dict) else {"items": data}


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


def normalize_name(value: Any) -> str:
    return " ".join(str(value if value is not None else "").replace("\ufeff", "").strip().lower().replace("_", " ").split())


def normalize_label(value: Any) -> str:
    return str(value if value is not None else "").replace("\ufeff", "").strip()


def label_key(value: str) -> str:
    normalized = normalize_label(value).lower()
    normalized = normalized.replace("\ufffd", "-")
    for char in ("_", "/", "\\", ":", ";", ",", ".", "(", ")", "[", "]"):
        normalized = normalized.replace(char, " ")
    normalized = normalized.replace("-", " ")
    return " ".join(normalized.split())


def is_benign_label(label: str) -> bool:
    return label_key(label) in {"", "0", "false", "no", "normal", "benign", "none"}


def is_attack_label(label: str, attack_labels: Sequence[str]) -> bool:
    key = label_key(label)
    explicit = {label_key(item) for item in attack_labels}
    if key in explicit:
        return True
    return "web attack" in key and any(token in key for token in ("brute force", "xss", "sql injection", "sql"))


def resolve_label_column(fieldnames: Sequence[str], requested: str) -> Optional[str]:
    if requested in fieldnames:
        return requested
    requested_norm = normalize_name(requested)
    for field in fieldnames:
        if normalize_name(field) == requested_norm:
            return field
    for alias in LABEL_COLUMN_ALIASES:
        alias_norm = normalize_name(alias)
        for field in fieldnames:
            if normalize_name(field) == alias_norm:
                return field
    return None


def sha256_file(path: Path, *, max_bytes: Optional[int] = None) -> str:
    digest = hashlib.sha256()
    remaining = max_bytes
    with path.open("rb") as handle:
        while True:
            chunk_size = 1024 * 1024
            if remaining is not None:
                if remaining <= 0:
                    break
                chunk_size = min(chunk_size, remaining)
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


def inspect_csv_dataset(
    path: Path,
    *,
    requested_label_column: str = "Label",
    attack_labels: Sequence[str] = DEFAULT_ATTACK_LABELS,
    repo_root: Path = REPO_ROOT,
    hash_file: bool = True,
) -> DatasetCandidate:
    resolved = resolve_path(path, repo_root)
    if not resolved.exists():
        return DatasetCandidate(
            path=relpath(resolved, repo_root),
            resolved_path=str(resolved),
            file_type=resolved.suffix.lower().lstrip(".") or "unknown",
            exists=False,
            requested_label_column=requested_label_column,
            reason="file not found",
        )

    raw_labels: Counter[str] = Counter()
    normalized_labels: Counter[str] = Counter()
    attack_seen: Counter[str] = Counter()
    first_attack: Optional[Dict[str, Any]] = None
    row_count = 0
    label_column: Optional[str] = None

    try:
        with resolved.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return DatasetCandidate(
                    path=relpath(resolved, repo_root),
                    resolved_path=str(resolved),
                    file_type="csv",
                    exists=True,
                    file_size_bytes=resolved.stat().st_size,
                    requested_label_column=requested_label_column,
                    reason="csv has no header",
                )
            label_column = resolve_label_column(reader.fieldnames, requested_label_column)
            if label_column is None:
                return DatasetCandidate(
                    path=relpath(resolved, repo_root),
                    resolved_path=str(resolved),
                    file_type="csv",
                    exists=True,
                    file_size_bytes=resolved.stat().st_size,
                    sha256=sha256_file(resolved) if hash_file else None,
                    requested_label_column=requested_label_column,
                    reason=f"label column {requested_label_column!r} not found",
                )
            for row in reader:
                row_count += 1
                raw_label = normalize_label(row.get(label_column))
                raw_labels[raw_label] += 1
                if is_benign_label(raw_label):
                    normalized_labels["BENIGN"] += 1
                elif is_attack_label(raw_label, attack_labels):
                    normalized_labels["ATTACK"] += 1
                    attack_seen[raw_label] += 1
                    if first_attack is None:
                        first_attack = {
                            "row_index_zero_based": row_count - 1,
                            "csv_line_number": row_count + 1,
                            "label": raw_label,
                        }
                else:
                    normalized_labels["OTHER"] += 1
    except OSError as exc:
        return DatasetCandidate(
            path=relpath(resolved, repo_root),
            resolved_path=str(resolved),
            file_type="csv",
            exists=True,
            file_size_bytes=resolved.stat().st_size,
            requested_label_column=requested_label_column,
            reason=f"read failed: {exc}",
        )

    benign_count = int(normalized_labels.get("BENIGN", 0))
    attack_count = int(normalized_labels.get("ATTACK", 0))
    return DatasetCandidate(
        path=relpath(resolved, repo_root),
        resolved_path=str(resolved),
        file_type="csv",
        exists=True,
        file_size_bytes=resolved.stat().st_size,
        sha256=sha256_file(resolved) if hash_file else None,
        row_count=row_count,
        label_column=label_column,
        requested_label_column=requested_label_column,
        raw_label_distribution=dict(raw_labels),
        normalized_label_distribution=dict(normalized_labels),
        benign_count=benign_count,
        attack_count=attack_count,
        attack_labels_detected=sorted(attack_seen),
        first_attack_row=first_attack,
        usable_for_tiny=benign_count >= 1 and attack_count >= 1,
        usable_for_balanced_250=benign_count >= 125 and attack_count >= 125,
        usable_for_transition_1k=benign_count >= 500 and attack_count >= 500,
        usable_for_natural_replay=attack_count > 0,
        usable_for_gpu_10k_rows=benign_count >= 5000 and attack_count >= 5000,
        usable_for_labeled_proof=benign_count > 0 and attack_count > 0,
        reason="ok" if benign_count > 0 and attack_count > 0 else "requires both BENIGN and WebAttack rows",
    )


def inspect_parquet_dataset(
    path: Path,
    *,
    requested_label_column: str = "Label",
    repo_root: Path = REPO_ROOT,
) -> DatasetCandidate:
    resolved = resolve_path(path, repo_root)
    return DatasetCandidate(
        path=relpath(resolved, repo_root),
        resolved_path=str(resolved),
        file_type="parquet",
        exists=resolved.exists(),
        file_size_bytes=resolved.stat().st_size if resolved.exists() else 0,
        requested_label_column=requested_label_column,
        reason="parquet candidate recorded, but this proof runner currently accepts CSV input only",
    )


def default_search_roots(repo_root: Path = REPO_ROOT) -> List[Path]:
    roots = [
        repo_root / "artifacts",
        repo_root / "tests" / "fixtures",
        Path("/tmp/eidos_proof_data"),
        Path("C:/Users/bmpar/OneDrive/Documents"),
        Path("E:/agent data"),
    ]
    # This older checkout is explicitly referenced by recent proof receipts and
    # is where the real WebAttacks CSV was previously materialized.
    roots.append(Path("C:/Users/bmpar/OneDrive/Documents/eidos-brain/eidos/artifacts/cicids_webattacks_samples"))
    return roots


def should_consider_path(path: Path) -> bool:
    lower = str(path).lower()
    if any(part.lower() in SKIP_DIR_NAMES for part in path.parts):
        return False
    return any(keyword in lower for keyword in SEARCH_KEYWORDS)


def discover_candidate_paths(search_roots: Sequence[Path]) -> List[Path]:
    seen: set[str] = set()
    paths: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root]
        else:
            candidates = []
            for pattern in ("*.csv", "*.parquet"):
                try:
                    candidates.extend(root.rglob(pattern))
                except OSError:
                    continue
        for path in candidates:
            if not path.is_file() or not should_consider_path(path):
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return sorted(paths, key=lambda item: str(item).lower())


def discover_datasets(
    *,
    search_roots: Sequence[Path],
    requested_label_column: str = "Label",
    attack_labels: Sequence[str] = DEFAULT_ATTACK_LABELS,
    repo_root: Path = REPO_ROOT,
) -> List[DatasetCandidate]:
    candidates: List[DatasetCandidate] = []
    for path in discover_candidate_paths(search_roots):
        suffix = path.suffix.lower()
        if suffix == ".csv":
            candidates.append(
                inspect_csv_dataset(
                    path,
                    requested_label_column=requested_label_column,
                    attack_labels=attack_labels,
                    repo_root=repo_root,
                )
            )
        elif suffix == ".parquet":
            candidates.append(inspect_parquet_dataset(path, requested_label_column=requested_label_column, repo_root=repo_root))
    return candidates


def choose_larger_dataset(candidates: Sequence[DatasetCandidate]) -> Optional[DatasetCandidate]:
    usable = [item for item in candidates if item.exists and item.file_type == "csv" and item.usable_for_labeled_proof]
    larger = [item for item in usable if item.row_count > 12 and item.usable_for_balanced_250 and item.usable_for_transition_1k]
    if not larger:
        return None
    return sorted(
        larger,
        key=lambda item: (
            item.usable_for_gpu_10k_rows,
            item.row_count,
            item.attack_count,
            item.file_size_bytes,
        ),
        reverse=True,
    )[0]


def write_dataset_registry(
    *,
    candidates: Sequence[DatasetCandidate],
    selected: Optional[DatasetCandidate],
    search_roots: Sequence[Path],
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    registry = {
        "generated_at_utc": utc_now(),
        "scope": "Bounded local CICIDS/WebAttacks labeled dataset registry for Eidos proof runs.",
        "search_roots": [str(root) for root in search_roots],
        "candidate_count": len(candidates),
        "selected_dataset_path": selected.path if selected else None,
        "selected_dataset_sha256": selected.sha256 if selected else None,
        "verdict": "larger_labeled_dataset_available" if selected else "DATASET_MISSING_HOLD",
        "candidates": [asdict(item) for item in candidates],
    }
    write_json(repo_root / REGISTRY_JSON, registry)
    write_dataset_registry_md(repo_root / REGISTRY_MD, registry)
    return registry


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(parsed):
        return "NA"
    return f"{parsed:.6g}"


def write_dataset_registry_md(path: Path, registry: Dict[str, Any]) -> None:
    lines = [
        "# Eidos CICIDS/WebAttacks Dataset Registry",
        "",
        f"- Generated at UTC: `{registry.get('generated_at_utc')}`",
        f"- Verdict: `{registry.get('verdict')}`",
        f"- Selected dataset: `{registry.get('selected_dataset_path') or 'none'}`",
        "",
        "## Search Roots",
        "",
    ]
    for root in registry.get("search_roots", []):
        lines.append(f"- `{root}`")
    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| path | type | rows | benign | attack | label column | balanced 250 | transition 1k | GPU 10k rows | reason |",
            "| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for item in registry.get("candidates", []):
        lines.append(
            "| {path} | {typ} | {rows} | {benign} | {attack} | {label} | {balanced} | {transition} | {gpu} | {reason} |".format(
                path=item.get("path"),
                typ=item.get("file_type"),
                rows=fmt(item.get("row_count")),
                benign=fmt(item.get("benign_count")),
                attack=fmt(item.get("attack_count")),
                label=item.get("label_column") or "NA",
                balanced=item.get("usable_for_balanced_250"),
                transition=item.get("usable_for_transition_1k"),
                gpu=item.get("usable_for_gpu_10k_rows"),
                reason=str(item.get("reason", "")).replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "## Proof Logic + Meaning",
            "",
            "The registry turns local dataset availability into a receipt. This prevents the proof harness from quietly falling back to a tiny fixture when a larger labeled CSV is missing or when a discovered file has a label-column problem.",
            "",
            "Known limits: the current proof runner consumes CSV input. Parquet candidates are recorded but not used unless converted or supported by a future proof-side adapter.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_normal_only_fixture(
    *,
    source: Path,
    target: Path,
    requested_label_column: str,
    repo_root: Path = REPO_ROOT,
) -> DatasetCandidate:
    resolved = resolve_path(source, repo_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"dataset has no header: {resolved}")
        actual_label = resolve_label_column(reader.fieldnames, requested_label_column)
        if actual_label is None:
            raise ValueError(f"label column {requested_label_column!r} not found in {resolved}")
        with target.open("w", newline="", encoding="utf-8") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if is_benign_label(normalize_label(row.get(actual_label))):
                    writer.writerow(row)
    return inspect_csv_dataset(target, requested_label_column=requested_label_column, repo_root=repo_root)


def cuda_receipt() -> Dict[str, Any]:
    return guardrails.cuda_receipt()


def build_scale_leg_plan(
    *,
    selected: Optional[DatasetCandidate],
    normal_candidate: Optional[DatasetCandidate],
    tiny_candidate: DatasetCandidate,
    cuda: Dict[str, Any],
    natural_frames: int,
    normal_frames: int,
    repo_root: Path = REPO_ROOT,
) -> List[guardrails.LegPlan]:
    plans: List[guardrails.LegPlan] = []
    tiny_path = resolve_path(Path(tiny_candidate.resolved_path), repo_root)
    tiny_frames = min(8, max(0, 2 * min(tiny_candidate.benign_count, tiny_candidate.attack_count)))
    plans.append(
        guardrails.LegPlan(
            name="tiny_fixture_smoke",
            requested_leg="tiny fixture smoke",
            sample_mode="transition",
            frames=tiny_frames,
            dataset_file=tiny_path,
            skip_reason=None if tiny_frames >= 2 else "tiny fixture lacks both benign and attack rows",
        )
    )
    if selected is None:
        missing = "no larger real labeled CICIDS/WebAttacks CSV was found"
        plans.extend(
            [
                guardrails.LegPlan("balanced_250_cpu", "balanced 250 CPU", "balanced", 250, tiny_path, skip_reason=missing),
                guardrails.LegPlan("transition_1k_cpu", "transition 1k CPU", "transition", 1000, tiny_path, skip_reason=missing),
                guardrails.LegPlan("natural_larger_replay_cpu", "natural larger replay CPU", "natural", natural_frames, tiny_path, skip_reason=missing),
            ]
        )
    else:
        dataset_path = Path(selected.resolved_path)
        plans.append(
            guardrails.LegPlan(
                "balanced_250_cpu",
                "balanced 250 CPU",
                "balanced",
                250,
                dataset_path,
                suite="full",
                skip_reason=None if selected.usable_for_balanced_250 else "larger dataset lacks 125 benign and 125 attack rows",
            )
        )
        plans.append(
            guardrails.LegPlan(
                "transition_1k_cpu",
                "transition 1k CPU",
                "transition",
                1000,
                dataset_path,
                suite="full",
                skip_reason=None if selected.usable_for_transition_1k else "larger dataset lacks 500 benign and 500 attack rows",
            )
        )
        first_attack = selected.first_attack_row or {}
        first_attack_line = int(first_attack.get("row_index_zero_based", 0) or 0) + 1 if first_attack else 1
        requested_frames = min(selected.row_count, max(natural_frames, first_attack_line + 1000))
        plans.append(
            guardrails.LegPlan(
                "natural_larger_replay_cpu",
                "natural larger attack-window replay CPU",
                "natural_attack_windows",
                requested_frames,
                dataset_path,
                suite="full",
                natural_window_pre=250,
                natural_window_post=250,
                natural_window_max_windows=1,
                skip_reason=None if selected.usable_for_natural_replay else "larger dataset has no attack rows",
            )
        )
    if normal_candidate is not None:
        planned_normal_frames = min(normal_candidate.row_count, max(0, normal_frames))
        plans.append(
            guardrails.LegPlan(
                "normal_only_negative_control",
                "normal-only negative control",
                "natural",
                planned_normal_frames,
                Path(normal_candidate.resolved_path),
                suite="full" if planned_normal_frames >= 250 else "smoke",
                skip_reason=None if planned_normal_frames > 0 else "no BENIGN rows available for normal-only fixture",
                generated_fixture=True,
            )
        )
    else:
        plans.append(
            guardrails.LegPlan(
                "normal_only_negative_control",
                "normal-only negative control",
                "natural",
                0,
                tiny_path,
                skip_reason="normal-only fixture was not generated",
                generated_fixture=True,
            )
        )
    gpu_skip = None
    gpu_dataset = selected
    if not cuda.get("cuda_available"):
        gpu_skip = "CUDA unavailable; torch reports CPU-only runtime"
    elif gpu_dataset is None or not gpu_dataset.usable_for_gpu_10k_rows:
        gpu_skip = "CUDA is available, but no discovered dataset has enough benign and attack rows for a 10k proof"
    plans.append(
        guardrails.LegPlan(
            "gpu_10k_optional",
            "optional GPU/CUDA 10k",
            "transition",
            10000,
            Path(gpu_dataset.resolved_path) if gpu_dataset else tiny_path,
            suite="full",
            skip_reason=gpu_skip,
        )
    )
    return plans


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


def label_counts_from_metrics(metrics: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    distribution = metrics.get("normalized_label_distribution")
    if not isinstance(distribution, dict):
        distribution = metrics.get("scored_normalized_label_distribution")
    if not isinstance(distribution, dict):
        return None, None
    return parse_int(distribution.get("BENIGN")), parse_int(distribution.get("ATTACK"))


def parse_device_receipt(environment_path: Path) -> Dict[str, Any]:
    receipt: Dict[str, Any] = {}
    if not environment_path.exists():
        return receipt
    for line in environment_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {"cuda_available", "selected_device", "runtime_seconds", "frames_per_second"}:
            receipt[key] = value
    if "cuda_available" in receipt:
        receipt["cuda_available"] = str(receipt["cuda_available"]).lower() == "true"
    for key in ("runtime_seconds", "frames_per_second"):
        if key in receipt:
            receipt[key] = parse_float(receipt[key])
    return receipt


def candidate_for_leg(leg: guardrails.LegPlan, candidates: Sequence[DatasetCandidate]) -> Optional[DatasetCandidate]:
    resolved = resolve_path(leg.dataset_file).resolve()
    for candidate in candidates:
        try:
            if Path(candidate.resolved_path).resolve() == resolved:
                return candidate
        except OSError:
            continue
    return None


def safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def metric_view(metrics: Dict[str, Any], name: str) -> Dict[str, Any]:
    views = metrics.get("event_view_metrics", {})
    if isinstance(views, dict) and isinstance(views.get(name), dict):
        return views[name]
    return {}


def scale_row_from_run(
    *,
    leg: guardrails.LegPlan,
    profile: str,
    run_dir: Path,
    command_result: Dict[str, Any],
    candidate: Optional[DatasetCandidate],
    core_touch_result: str,
    sweep_item: Optional[Dict[str, Any]] = None,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    artifacts = guardrails.load_run_artifacts(run_dir)
    metrics = artifacts.get("metrics", {})
    calibration = artifacts.get("calibration", {})
    raw = metric_view(metrics, "raw")
    merged = metric_view(metrics, "merged")
    deduped = metric_view(metrics, "deduped")
    confirmed = metric_view(metrics, "confirmed")
    calibrated = metric_view(metrics, "calibrated")
    before = calibration.get("attack_window_summary_before", {}) if isinstance(calibration, dict) else {}
    after = calibration.get("attack_window_summary_after", {}) if isinstance(calibration, dict) else {}
    device = parse_device_receipt(run_dir / "environment.txt")
    benign_frames, attack_frames = label_counts_from_metrics(metrics)
    crash_hits = parse_int(artifacts.get("crash_scan", {}).get("crash_hit_count")) or 0
    raw_visible = guardrails.raw_visibility_intact(metrics)
    collapsed = guardrails.attack_visibility_collapsed(calibration, metrics)
    if sweep_item is None:
        precision = calibrated.get("precision")
        recall = calibrated.get("recall")
        f1 = calibrated.get("f1")
        fp_per_10k = calibrated.get("false_positives_per_10k_frames")
        confirmed_events = confirmed.get("event_count")
        calibrated_events = calibrated.get("event_count")
        false_positive_events = calibrated.get("false_positives")
        suppressed_events = metrics.get("calibration_suppressed_events", calibration.get("counts", {}).get("suppressed_events"))
        coverage = after.get("attack_window_coverage_pct")
        latency = after.get("first_detection_latency_frames")
    else:
        precision = sweep_item.get("calibrated_precision", sweep_item.get("precision"))
        recall = sweep_item.get("calibrated_recall", sweep_item.get("recall"))
        f1 = sweep_item.get("calibrated_f1", sweep_item.get("f1"))
        fp_per_10k = sweep_item.get("calibrated_fp_per_10k", sweep_item.get("fp_per_10k"))
        confirmed_events = sweep_item.get("confirmed_count")
        calibrated_events = sweep_item.get("calibrated_confirmed_count")
        false_positive_events = sweep_item.get("calibrated_false_positive_count", sweep_item.get("false_positive_count"))
        suppressed_events = sweep_item.get("calibration_suppressed_count", sweep_item.get("suppressed_count"))
        coverage = sweep_item.get("calibrated_coverage", sweep_item.get("coverage"))
        latency = sweep_item.get("calibrated_first_detection_latency", sweep_item.get("first_detection_latency"))
        if parse_float(sweep_item.get("calibrated_coverage")) is not None and parse_float(sweep_item.get("coverage")) is not None:
            collapsed = bool(parse_float(sweep_item.get("calibrated_coverage")) < parse_float(sweep_item.get("coverage")))
        if parse_float(sweep_item.get("calibrated_recall")) is not None and parse_float(sweep_item.get("recall")) is not None:
            collapsed = collapsed or bool(parse_float(sweep_item.get("calibrated_recall")) < parse_float(sweep_item.get("recall")))

    verdict = "ROW_PASS"
    verdict_reason = "row completed with raw visibility and no gate hold"
    if core_touch_result != "passed":
        verdict = "ROW_CORE_TOUCH_FAIL"
        verdict_reason = "core-touch policy did not pass"
    elif not metrics or crash_hits > 0 or not raw_visible or command_result.get("returncode") not in (0,):
        verdict = "ROW_CRASH"
        verdict_reason = "metrics missing, crash hits found, raw visibility hidden, or proof command returned nonzero"
    elif collapsed:
        verdict = "ROW_HOLD_RECALL"
        verdict_reason = "calibrated profile reduced attack-window visibility or recall"
    elif (
        leg.name == "normal_only_negative_control"
        and profile != "off"
        and parse_float(fp_per_10k) is not None
        and float(parse_float(fp_per_10k) or 0.0) > NORMAL_ONLY_FP_PER_10K_TARGET
    ):
        verdict = "ROW_HOLD_FP"
        verdict_reason = f"normal-only calibrated FP/10k exceeds {NORMAL_ONLY_FP_PER_10K_TARGET:g}"

    return {
        "run_name": leg.name,
        "profile": profile,
        "dataset_path": candidate.path if candidate else relpath(resolve_path(leg.dataset_file), repo_root),
        "dataset_sha256": candidate.sha256 if candidate else None,
        "sample_mode": metrics.get("sample_mode", leg.sample_mode),
        "frame_count": metrics.get("frames_processed"),
        "benign_frames": benign_frames,
        "attack_frames": attack_frames,
        "attack_windows": before.get("attack_window_count"),
        "raw_events": raw.get("event_count"),
        "merged_events": merged.get("event_count"),
        "deduped_events": deduped.get("event_count"),
        "confirmed_events": confirmed_events,
        "calibrated_events": calibrated_events,
        "suppressed_events": suppressed_events,
        "false_positive_events": false_positive_events,
        "raw_false_positive_events": raw.get("false_positives"),
        "fp_per_10k_benign_frames": fp_per_10k,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "attack_window_coverage": coverage,
        "first_detection_latency": latency,
        "crash_hits": crash_hits,
        "runtime_seconds": metrics.get("runtime_seconds", device.get("runtime_seconds", command_result.get("runtime_seconds"))),
        "fps": metrics.get("frames_per_second", device.get("frames_per_second")),
        "selected_device": device.get("selected_device", "unknown"),
        "cuda_available": device.get("cuda_available"),
        "core_touch_result": core_touch_result,
        "raw_visibility_intact": raw_visible,
        "attack_visibility_collapsed": collapsed,
        "run_status": "completed" if metrics else "missing_metrics",
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "run_path": relpath(run_dir, repo_root),
    }


def profile_rows_from_run(
    *,
    leg: guardrails.LegPlan,
    run_dir: Path,
    command_result: Dict[str, Any],
    candidate: Optional[DatasetCandidate],
    core_touch_result: str,
    repo_root: Path = REPO_ROOT,
) -> List[Dict[str, Any]]:
    rows = [
        scale_row_from_run(
            leg=leg,
            profile="off",
            run_dir=run_dir,
            command_result=command_result,
            candidate=candidate,
            core_touch_result=core_touch_result,
            repo_root=repo_root,
        )
    ]
    sweep = read_json(run_dir / "confirmation_profile_sweep.json").get("profiles")
    if not isinstance(sweep, list):
        metrics = read_json(run_dir / "labeled_metrics.json")
        sweep = metrics.get("confirmation_profile_sweep", [])
    for item in sweep:
        if not isinstance(item, dict):
            continue
        profile = str(item.get("profile"))
        if profile not in REQUESTED_PROFILES:
            continue
        rows.append(
            scale_row_from_run(
                leg=leg,
                profile=profile,
                run_dir=run_dir,
                command_result=command_result,
                candidate=candidate,
                core_touch_result=core_touch_result,
                sweep_item=item,
                repo_root=repo_root,
            )
        )
    return rows


def completed_run_receipts_exist(run_dir: Path) -> bool:
    return (run_dir / "run_manifest.json").exists() and (run_dir / "labeled_metrics.json").exists()


def partial_receipts(run_dir: Path, *, repo_root: Path = REPO_ROOT) -> List[str]:
    receipts = [run_dir / name for name in PARTIAL_RECEIPT_NAMES if (run_dir / name).exists()]
    return [relpath(path, repo_root) for path in receipts]


def reused_command_result(run_dir: Path) -> Dict[str, Any]:
    artifacts = guardrails.load_run_artifacts(run_dir)
    manifest = artifacts.get("run_manifest", {})
    metrics = artifacts.get("metrics", {})
    device = manifest.get("device", {}) if isinstance(manifest.get("device"), dict) else {}
    runtime = metrics.get("runtime_seconds", device.get("runtime_seconds"))
    return {
        "command": manifest.get("command", "existing run_manifest.json reused"),
        "resolved_command": manifest.get("command"),
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "runtime_seconds": runtime,
        "timed_out": False,
        "timeout_seconds": None,
        "timeout_reason": None,
        "python_process_exited": True,
        "reused_existing": True,
    }


def write_scale_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SCALE_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in SCALE_COLUMNS})


def value_for_plot(row: Dict[str, Any], key: str) -> float:
    parsed = parse_float(row.get(key))
    return float(parsed) if parsed is not None else 0.0


def write_bar_svg(
    path: Path,
    *,
    title: str,
    labels: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float]]],
    width: int = 1100,
    height: int = 480,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_left = 70
    plot_top = 58
    plot_width = width - 110
    plot_height = height - 125
    max_value = max([0.0] + [float(value) for _, values in series for value in values])
    scale = plot_height / max(max_value, 1.0)
    group_width = plot_width / max(len(labels), 1)
    bar_gap = 4
    bar_width = max(4, (group_width - 12) / max(len(series), 1) - bar_gap)
    colors = ("#245c7a", "#7a9a3d", "#9c5f2f", "#6b5b95", "#4c7c59")
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'.format(w=width, h=height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{title}</text>',
        f'<line x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}" stroke="#333" stroke-width="1"/>',
    ]
    for sidx, (name, values) in enumerate(series):
        color = colors[sidx % len(colors)]
        legend_x = plot_left + sidx * 180
        lines.append(f'<rect x="{legend_x}" y="{height - 35}" width="14" height="14" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 20}" y="{height - 23}" font-family="Arial" font-size="13">{name}</text>')
        for idx, value in enumerate(values):
            x = plot_left + idx * group_width + 8 + sidx * (bar_width + bar_gap)
            bar_height = max(0.0, float(value) * scale)
            y = plot_top + plot_height - bar_height
            lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}"/>')
            if value:
                lines.append(f'<text x="{x + bar_width / 2:.2f}" y="{y - 4:.2f}" text-anchor="middle" font-family="Arial" font-size="10">{fmt(value)}</text>')
    for idx, label in enumerate(labels):
        x = plot_left + idx * group_width + group_width / 2
        safe = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lines.append(f'<text x="{x:.2f}" y="{plot_top + plot_height + 18}" text-anchor="middle" font-family="Arial" font-size="11">{safe}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rows_for_profile(rows: Sequence[Dict[str, Any]], profile: str = "balanced") -> List[Dict[str, Any]]:
    preferred = [row for row in rows if row.get("profile") == profile]
    if preferred:
        return preferred
    return [row for row in rows if row.get("profile") == "off"]


def write_plots(out_dir: Path, rows: Sequence[Dict[str, Any]]) -> List[str]:
    plot_dir = out_dir / "plots"
    selected_rows = rows_for_profile(rows, "balanced")
    labels = [str(row.get("run_name")) for row in selected_rows]
    outputs: List[str] = []
    plots = [
        (
            "event_funnel_by_run.svg",
            "Event Funnel By Run",
            [
                ("raw", [value_for_plot(row, "raw_events") for row in selected_rows]),
                ("merged", [value_for_plot(row, "merged_events") for row in selected_rows]),
                ("deduped", [value_for_plot(row, "deduped_events") for row in selected_rows]),
                ("confirmed", [value_for_plot(row, "confirmed_events") for row in selected_rows]),
                ("calibrated", [value_for_plot(row, "calibrated_events") for row in selected_rows]),
            ],
        ),
        (
            "fp_per_10k_by_run.svg",
            "FP Per 10k By Run",
            [("FP/10k", [value_for_plot(row, "fp_per_10k_benign_frames") for row in selected_rows])],
        ),
        (
            "recall_coverage_by_run.svg",
            "Recall And Attack-Window Coverage By Run",
            [
                ("recall", [value_for_plot(row, "recall") for row in selected_rows]),
                ("coverage", [value_for_plot(row, "attack_window_coverage") for row in selected_rows]),
            ],
        ),
        (
            "latency_by_run.svg",
            "First Detection Latency By Run",
            [("latency", [value_for_plot(row, "first_detection_latency") for row in selected_rows])],
        ),
        (
            "runtime_fps_by_run.svg",
            "Runtime And FPS By Run",
            [
                ("runtime seconds", [value_for_plot(row, "runtime_seconds") for row in selected_rows]),
                ("fps", [value_for_plot(row, "fps") for row in selected_rows]),
            ],
        ),
    ]
    for filename, title, series in plots:
        path = plot_dir / filename
        write_bar_svg(path, title=title, labels=labels, series=series)
        outputs.append(relpath(path))
    return outputs


BEFORE_AFTER_COLUMNS = (
    "run_name",
    "comparison",
    "from_profile",
    "to_profile",
    "from_fp_per_10k",
    "to_fp_per_10k",
    "delta_fp_per_10k",
    "from_precision",
    "to_precision",
    "delta_precision",
    "from_recall",
    "to_recall",
    "delta_recall",
    "from_f1",
    "to_f1",
    "delta_f1",
    "from_attack_window_coverage",
    "to_attack_window_coverage",
    "delta_attack_window_coverage",
    "from_calibrated_events",
    "to_calibrated_events",
    "delta_calibrated_events",
    "from_suppressed_events",
    "to_suppressed_events",
    "delta_suppressed_events",
    "fp_target",
    "fp_stretch_target",
    "target_status",
)


def _first_profile_row(rows: Sequence[Dict[str, Any]], run_name: str, profile: str) -> Optional[Dict[str, Any]]:
    return next((row for row in rows if row.get("run_name") == run_name and row.get("profile") == profile), None)


def _delta(after: Any, before: Any) -> Optional[float]:
    after_value = parse_float(after)
    before_value = parse_float(before)
    if after_value is None or before_value is None:
        return None
    return after_value - before_value


def _target_status(row: Dict[str, Any], *, run_name: str) -> str:
    value = parse_float(row.get("to_fp_per_10k"))
    if run_name != "normal_only_negative_control" or value is None:
        return "not_applicable"
    if value <= NORMAL_ONLY_FP_PER_10K_STRETCH_TARGET:
        return "stretch_pass"
    if value <= NORMAL_ONLY_FP_PER_10K_TARGET:
        return "intermediate_pass"
    return "hold_fp"


def before_after_delta_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    run_names = sorted({str(row.get("run_name")) for row in rows if row.get("run_name")})
    for run_name in run_names:
        off = _first_profile_row(rows, run_name, "off")
        current = _first_profile_row(rows, run_name, CURRENT_CALIBRATED_PROFILE)
        tuned = _first_profile_row(rows, run_name, TUNED_CALIBRATED_PROFILE)
        pairs = [
            ("off_to_current_calibrated", off, current),
            ("current_calibrated_to_tuned_calibrated", current, tuned),
        ]
        for comparison, before, after in pairs:
            if before is None or after is None:
                continue
            row = {
                "run_name": run_name,
                "comparison": comparison,
                "from_profile": before.get("profile"),
                "to_profile": after.get("profile"),
                "from_fp_per_10k": before.get("fp_per_10k_benign_frames"),
                "to_fp_per_10k": after.get("fp_per_10k_benign_frames"),
                "delta_fp_per_10k": _delta(after.get("fp_per_10k_benign_frames"), before.get("fp_per_10k_benign_frames")),
                "from_precision": before.get("precision"),
                "to_precision": after.get("precision"),
                "delta_precision": _delta(after.get("precision"), before.get("precision")),
                "from_recall": before.get("recall"),
                "to_recall": after.get("recall"),
                "delta_recall": _delta(after.get("recall"), before.get("recall")),
                "from_f1": before.get("f1"),
                "to_f1": after.get("f1"),
                "delta_f1": _delta(after.get("f1"), before.get("f1")),
                "from_attack_window_coverage": before.get("attack_window_coverage"),
                "to_attack_window_coverage": after.get("attack_window_coverage"),
                "delta_attack_window_coverage": _delta(after.get("attack_window_coverage"), before.get("attack_window_coverage")),
                "from_calibrated_events": before.get("calibrated_events"),
                "to_calibrated_events": after.get("calibrated_events"),
                "delta_calibrated_events": _delta(after.get("calibrated_events"), before.get("calibrated_events")),
                "from_suppressed_events": before.get("suppressed_events"),
                "to_suppressed_events": after.get("suppressed_events"),
                "delta_suppressed_events": _delta(after.get("suppressed_events"), before.get("suppressed_events")),
                "fp_target": NORMAL_ONLY_FP_PER_10K_TARGET,
                "fp_stretch_target": NORMAL_ONLY_FP_PER_10K_STRETCH_TARGET,
            }
            row["target_status"] = _target_status(row, run_name=run_name)
            output.append(row)
    return output


def write_before_after_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BEFORE_AFTER_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in BEFORE_AFTER_COLUMNS})


def write_before_after_md(path: Path, rows: Sequence[Dict[str, Any]], package: Dict[str, Any]) -> None:
    lines = [
        "# Sentinel Guardrail Scale Matrix Before/After",
        "",
        f"- Final verdict: `{package.get('final_verdict')}`",
        f"- Current calibrated profile: `{CURRENT_CALIBRATED_PROFILE}`",
        f"- Tuned calibrated profile: `{TUNED_CALIBRATED_PROFILE}`",
        f"- Normal-only FP/10k target: `<={NORMAL_ONLY_FP_PER_10K_TARGET:g}`",
        f"- Normal-only FP/10k stretch target: `<={NORMAL_ONLY_FP_PER_10K_STRETCH_TARGET:g}`",
        "",
        "## Delta Table",
        "",
        "| run | comparison | from | to | FP/10k delta | precision delta | recall delta | F1 delta | coverage delta | calibrated event delta | suppressed event delta | target |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {run} | {comparison} | {from_profile} | {to_profile} | {fp} | {precision} | {recall} | {f1} | {coverage} | {events} | {suppressed} | {target} |".format(
                run=row.get("run_name"),
                comparison=row.get("comparison"),
                from_profile=row.get("from_profile"),
                to_profile=row.get("to_profile"),
                fp=fmt(row.get("delta_fp_per_10k")),
                precision=fmt(row.get("delta_precision")),
                recall=fmt(row.get("delta_recall")),
                f1=fmt(row.get("delta_f1")),
                coverage=fmt(row.get("delta_attack_window_coverage")),
                events=fmt(row.get("delta_calibrated_events")),
                suppressed=fmt(row.get("delta_suppressed_events")),
                target=row.get("target_status"),
            )
        )
    lines.extend(
        [
            "",
            "## Proof Logic + Meaning",
            "",
            "The before/after report keeps three views visible: `off` as raw/off evidence, `low_noise` as the current calibrated profile, and `strict` as the tuned proof-side profile. The math compares deltas only between saved receipt rows; it does not infer hidden metrics.",
            "",
            "What remains unproven: these deltas prove only the selected CICIDS/WebAttacks rows and profiles that completed. Missing rows remain `NA`, not zero.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_before_after_plots(out_dir: Path, rows: Sequence[Dict[str, Any]]) -> List[str]:
    plot_dir = out_dir / "plots"
    run_names = sorted({str(row.get("run_name")) for row in rows if row.get("run_name")})
    profiles = ("off", CURRENT_CALIBRATED_PROFILE, TUNED_CALIBRATED_PROFILE)

    def values(profile: str, key: str) -> List[float]:
        return [
            value_for_plot(_first_profile_row(rows, run_name, profile) or {}, key)
            for run_name in run_names
        ]

    outputs: List[str] = []
    plot_specs = [
        (
            "before_after_fp_per_10k.svg",
            "Before/After FP Per 10k",
            [(profile, values(profile, "fp_per_10k_benign_frames")) for profile in profiles],
        ),
        (
            "before_after_recall.svg",
            "Before/After Recall",
            [(profile, values(profile, "recall")) for profile in profiles],
        ),
        (
            "before_after_coverage.svg",
            "Before/After Attack-Window Coverage",
            [(profile, values(profile, "attack_window_coverage")) for profile in profiles],
        ),
        (
            "before_after_event_funnel.svg",
            "Before/After Event Funnel",
            [
                ("raw", values("off", "raw_events")),
                ("merged", values("off", "merged_events")),
                ("deduped", values("off", "deduped_events")),
                (CURRENT_CALIBRATED_PROFILE, values(CURRENT_CALIBRATED_PROFILE, "calibrated_events")),
                (TUNED_CALIBRATED_PROFILE, values(TUNED_CALIBRATED_PROFILE, "calibrated_events")),
            ],
        ),
        (
            "before_after_suppressed_events.svg",
            "Before/After Suppressed Event Counts",
            [
                (CURRENT_CALIBRATED_PROFILE, values(CURRENT_CALIBRATED_PROFILE, "suppressed_events")),
                (TUNED_CALIBRATED_PROFILE, values(TUNED_CALIBRATED_PROFILE, "suppressed_events")),
            ],
        ),
    ]
    for filename, title, series in plot_specs:
        path = plot_dir / filename
        write_bar_svg(path, title=title, labels=run_names, series=series)
        outputs.append(relpath(path))
    return outputs


def write_before_after_artifacts(out_dir: Path, rows: Sequence[Dict[str, Any]], package: Dict[str, Any]) -> Dict[str, Any]:
    delta_rows = before_after_delta_rows(rows)
    payload = {
        "generated_at_utc": utc_now(),
        "current_calibrated_profile": CURRENT_CALIBRATED_PROFILE,
        "tuned_calibrated_profile": TUNED_CALIBRATED_PROFILE,
        "normal_only_fp_per_10k_target": NORMAL_ONLY_FP_PER_10K_TARGET,
        "normal_only_fp_per_10k_stretch_target": NORMAL_ONLY_FP_PER_10K_STRETCH_TARGET,
        "rows": delta_rows,
    }
    write_json(out_dir / "scale_matrix_before_after.json", payload)
    write_before_after_csv(out_dir / "scale_matrix_before_after.csv", delta_rows)
    write_before_after_md(out_dir / "scale_matrix_before_after.md", delta_rows, package)
    plots = write_before_after_plots(out_dir, rows)
    payload["plots"] = plots
    write_json(out_dir / "scale_matrix_before_after.json", payload)
    return payload


def formulas_block() -> List[str]:
    return [
        "```text",
        "FP/10k = false_positive_events / benign_frames * 10000",
        "precision = true_positive_events / max(true_positive_events + false_positive_events, 1)",
        "recall = detected_attack_windows / max(total_attack_windows, 1)",
        "F1 = 2 * precision * recall / max(precision + recall, epsilon)",
        "attack_window_coverage = attack_windows_with_detection / total_attack_windows",
        "```",
    ]


def final_verdict(
    *,
    selected_dataset: Optional[DatasetCandidate],
    rows: Sequence[Dict[str, Any]],
    skipped_legs: Sequence[Dict[str, Any]],
    core_policy: Dict[str, Any],
    branch_pushed: bool,
) -> str:
    _ = branch_pushed
    if selected_dataset is None:
        return "DATASET_MISSING_HOLD"
    if core_policy.get("passed") is not True:
        return "SCALE_HOLD_CRASH_OR_CORE_TOUCH"
    if any(row.get("verdict") in {"ROW_CRASH", "ROW_CORE_TOUCH_FAIL"} for row in rows):
        return "SCALE_HOLD_CRASH_OR_CORE_TOUCH"
    if any(row.get("verdict") == "ROW_HOLD_RECALL" for row in rows if row.get("profile") != "off"):
        return "SCALE_HOLD_RECALL_COLLAPSE"
    if any(
        row.get("verdict") == "ROW_HOLD_FP"
        for row in rows
        if row.get("run_name") == "normal_only_negative_control" and row.get("profile") != "off"
    ):
        return "SCALE_HOLD_FALSE_POSITIVES"
    required = {"balanced_250_cpu", "transition_1k_cpu", "natural_larger_replay_cpu", "normal_only_negative_control"}
    skipped_required = required.intersection({str(item.get("leg")) for item in skipped_legs})
    if skipped_required:
        return "CALIBRATION_ONLY_NEEDS_TUNING"
    completed_required = required.intersection({str(row.get("run_name")) for row in rows if row.get("run_status") == "completed"})
    if completed_required != required:
        return "CALIBRATION_ONLY_NEEDS_TUNING"
    if any(row.get("verdict") != "ROW_PASS" for row in rows if row.get("profile") != "off"):
        return "CALIBRATION_ONLY_NEEDS_TUNING"
    return "MERGE_READY_LARGER_LABELED_GUARDRAILS"


def merge_readiness_reasons(package: Dict[str, Any]) -> List[str]:
    verdict = package.get("final_verdict")
    reasons: List[str] = []
    if verdict == "MERGE_READY_LARGER_LABELED_GUARDRAILS":
        return ["All required larger labeled CPU guardrails completed without FP, recall, crash, or core-touch holds."]
    if verdict == "DATASET_MISSING_HOLD":
        reasons.append("No usable larger labeled CICIDS/WebAttacks CSV was selected.")
    if verdict == "SCALE_HOLD_CRASH_OR_CORE_TOUCH":
        reasons.append("At least one row had crash/raw-visibility/core-touch failure, or core-touch policy did not pass.")
    if verdict == "SCALE_HOLD_RECALL_COLLAPSE":
        reasons.append("At least one calibrated profile reduced attack-window visibility or recall.")
    if verdict == "SCALE_HOLD_FALSE_POSITIVES":
        reasons.append(f"Normal-only calibrated FP/10k remained above {NORMAL_ONLY_FP_PER_10K_TARGET:g}.")
    if verdict == "CALIBRATION_ONLY_NEEDS_TUNING":
        skipped = [str(item.get("leg")) for item in package.get("skipped_legs", [])]
        if skipped:
            reasons.append(f"Required or requested legs were skipped/incomplete: {', '.join(sorted(set(skipped)))}.")
        else:
            reasons.append("One or more required completed rows did not reach ROW_PASS.")
    if not reasons:
        reasons.append("One or more required receipts are missing or inconclusive.")
    return reasons


def write_scale_matrix_md(path: Path, package: Dict[str, Any]) -> None:
    selected = package.get("selected_dataset") or {}
    selected = selected if isinstance(selected, dict) else {}
    raw_dist = selected.get("raw_label_distribution") or {}
    normalized_dist = selected.get("normalized_label_distribution") or {}
    completed_legs = sorted({str(row.get("run_name")) for row in package.get("rows", []) if row.get("run_status") == "completed"})
    readiness_reasons = merge_readiness_reasons(package)
    lines = [
        "# Sentinel Guardrail Scale Matrix",
        "",
        f"- Final verdict: `{package.get('final_verdict')}`",
        f"- Selected dataset: `{selected.get('path') if selected else 'none'}`",
        f"- Dataset rows: `{selected.get('row_count', 'NA')}`",
        f"- Benign / attack rows: `{selected.get('benign_count', 'NA')}` / `{selected.get('attack_count', 'NA')}`",
        f"- Raw label distribution: `{raw_dist}`",
        f"- Normalized label distribution: `{normalized_dist}`",
        f"- Branch pushed before run: `{package.get('branch_preservation', {}).get('branch_pushed')}`",
        f"- Core behavior changed: `{package.get('core_behavior_changed')}`",
        f"- Core-touch policy: `{package.get('core_touch_policy', {}).get('passed')}`",
        f"- Normal-only FP/10k target: `<={NORMAL_ONLY_FP_PER_10K_TARGET:g}`; stretch target: `<={NORMAL_ONLY_FP_PER_10K_STRETCH_TARGET:g}`",
        f"- Completed proof legs: `{', '.join(completed_legs) if completed_legs else 'none'}`",
        f"- Exact merge-readiness reason: `{'; '.join(readiness_reasons)}`",
        "",
        "## Metrics And Formulas",
        "",
        *formulas_block(),
        "",
        "## Scale Matrix",
        "",
        "| run | profile | frames | benign | attack | raw | merged | deduped | confirmed | calibrated | FP/10k | precision | recall | F1 | coverage | latency | crash | verdict |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in package.get("rows", []):
        lines.append(
            "| {run} | {profile} | {frames} | {benign} | {attack} | {raw} | {merged} | {deduped} | {confirmed} | {calibrated} | {fp} | {precision} | {recall} | {f1} | {coverage} | {latency} | {crash} | {verdict} |".format(
                run=row.get("run_name"),
                profile=row.get("profile"),
                frames=fmt(row.get("frame_count")),
                benign=fmt(row.get("benign_frames")),
                attack=fmt(row.get("attack_frames")),
                raw=fmt(row.get("raw_events")),
                merged=fmt(row.get("merged_events")),
                deduped=fmt(row.get("deduped_events")),
                confirmed=fmt(row.get("confirmed_events")),
                calibrated=fmt(row.get("calibrated_events")),
                fp=fmt(row.get("fp_per_10k_benign_frames")),
                precision=fmt(row.get("precision")),
                recall=fmt(row.get("recall")),
                f1=fmt(row.get("f1")),
                coverage=fmt(row.get("attack_window_coverage")),
                latency=fmt(row.get("first_detection_latency")),
                crash=fmt(row.get("crash_hits")),
                verdict=row.get("verdict"),
            )
        )
    lines.extend(
        [
            "",
            "## Skipped Legs",
            "",
        ]
    )
    for item in package.get("skipped_legs", []):
        lines.append(f"- `{item.get('leg')}`: {item.get('reason')}")
    if not package.get("skipped_legs"):
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Proof Logic + Meaning",
            "",
            f"Goal reached: larger labeled CICIDS/WebAttacks availability was turned into a registry and, where data allowed, a scale matrix with raw, merged, deduped, confirmed, and calibrated views side by side. Gate status is `{package.get('final_verdict')}`.",
            "",
            "Specific logic/math used: the proof compares event funnel counts, FP/10k, precision, recall, F1, attack-window coverage, first detection latency, crash-hit count, runtime, FPS, and core-touch policy without changing core Eidos behavior.",
            "",
            "Why this is better than the previous state: the earlier guardrail run was limited to the tiny fixture. This package records whether a larger real labeled source exists, what it contains, and what the proof harness did with it.",
            "",
            "Evidence/artifacts: `scale_matrix.json`, `scale_matrix.csv`, `scale_matrix.md`, `scale_matrix_before_after.json`, `scale_matrix_before_after.csv`, `scale_matrix_before_after.md`, per-run proof receipts, dataset registry, plots, core-touch receipt, and Drive manifest when available.",
            "",
            "What it proves: proof-side reproducibility and larger-data guardrail accounting improved. It proves only the rows and profiles actually run.",
            "",
            "What it does not prove: production readiness, every CICIDS/WebAttacks variant, GPU behavior when CUDA is unavailable, or any core behavior improvement.",
            "",
            "## Remaining Uncertainty",
            "",
            *[f"- {reason}" for reason in readiness_reasons],
            "",
            "How this moves Eidos closer to the ultimate goal: Eidos is not becoming more intelligent because it speaks less. It is becoming more intelligent only if it speaks less while preserving truth, preserving anomaly visibility, and making uncertainty auditable.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def copy_allowlist_to_drive(
    *,
    out_dir: Path,
    run_date: str,
    drive_root: Optional[Path],
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    manifest = {
        "drive_copy_attempted": drive_root is not None,
        "drive_copy_success": False,
        "copy_status": "skipped",
        "drive_root": str(drive_root) if drive_root else "unknown",
        "drive_run_dir": "unknown",
        "files_considered": [],
        "files_copied": [],
        "files_skipped": [],
        "failed": [],
        "timestamp_utc": utc_now(),
        "reason": "",
    }
    if drive_root is None:
        manifest["reason"] = "EIDOS_PROOF_DRIVE_DIR not set and no writable Drive root discovered"
        return manifest
    drive_run_dir = drive_root / "Eidos_Brain_Proof_Phase" / run_date / out_dir.name
    manifest["drive_run_dir"] = str(drive_run_dir)
    top_level_names = {
        "scale_matrix.json",
        "scale_matrix.csv",
        "scale_matrix.md",
        "scale_matrix_before_after.json",
        "scale_matrix_before_after.csv",
        "scale_matrix_before_after.md",
        "core_touch_policy.json",
        "core_touch_policy.md",
        "proof_logic_meaning.md",
        "codex_journal.md",
        "plain_language_test_analysis.md",
    }
    for rel in sorted(top_level_names):
        path = out_dir / rel
        if path.exists():
            manifest["files_considered"].append(rel)
            target = drive_run_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            manifest["files_copied"].append(rel)
    for path in sorted((out_dir / "plots").glob("*.svg")):
        rel = relpath(path, out_dir)
        manifest["files_considered"].append(rel)
        target = drive_run_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest["files_copied"].append(rel)
    for run_dir in sorted((out_dir / "runs").glob("*/*")):
        if not (run_dir / "run_manifest.json").exists():
            continue
        for name in guardrails.PER_RUN_DRIVE_ALLOWLIST:
            path = run_dir / name
            rel = relpath(path, out_dir)
            if path.exists():
                manifest["files_considered"].append(rel)
                target = drive_run_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                manifest["files_copied"].append(rel)
    manifest["drive_copy_success"] = True
    manifest["copy_status"] = "copied"
    manifest["reason"] = "copied allowlisted scale matrix receipts"
    return manifest


def finalize_drive_manifest_copy(out_dir: Path, drive_manifest: Dict[str, Any]) -> Dict[str, Any]:
    if not drive_manifest.get("drive_copy_attempted") or drive_manifest.get("drive_run_dir") in (None, "unknown"):
        return drive_manifest
    drive_run_dir = Path(str(drive_manifest["drive_run_dir"]))
    try:
        drive_run_dir.mkdir(parents=True, exist_ok=True)
        target = drive_run_dir / "drive_manifest.json"
        shutil.copy2(out_dir / "drive_manifest.json", target)
        if "drive_manifest.json" not in drive_manifest.get("files_considered", []):
            drive_manifest.setdefault("files_considered", []).append("drive_manifest.json")
        if "drive_manifest.json" not in drive_manifest.get("files_copied", []):
            drive_manifest.setdefault("files_copied", []).append("drive_manifest.json")
        drive_manifest["drive_copy_success"] = not bool(drive_manifest.get("failed"))
        drive_manifest["copy_status"] = "copied" if drive_manifest["drive_copy_success"] else "partial"
        drive_manifest["reason"] = (
            "copied allowlisted scale matrix receipts"
            if drive_manifest["drive_copy_success"]
            else "copied final drive manifest, but some earlier allowlist entries failed"
        )
    except Exception as exc:
        drive_manifest.setdefault("failed", []).append({"path": "drive_manifest.json", "reason": str(exc)})
        drive_manifest["drive_copy_success"] = False
        drive_manifest["copy_status"] = "partial" if drive_manifest.get("files_copied") else "failed"
        drive_manifest["reason"] = f"failed to copy final drive_manifest.json: {exc}"
    return drive_manifest


def discover_drive_root() -> Optional[Path]:
    return guardrails.discover_drive_root_for_env()


def run_git(args: Sequence[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def branch_preservation_status(branch: str, *, git_root: Path) -> Dict[str, Any]:
    result = run_git(["ls-remote", "--heads", "origin", branch], cwd=git_root)
    remote_line = result.stdout.strip()
    return {
        "branch": branch,
        "branch_pushed": bool(remote_line),
        "remote_ref": remote_line,
        "check_returncode": result.returncode,
        "check_stderr": result.stderr.strip(),
    }


def write_docs_report(repo_root: Path, package: Dict[str, Any]) -> None:
    report_path = repo_root / DOC_REPORT
    write_scale_matrix_md(report_path, package)


def write_proof_run_companions(repo_root: Path, package: Dict[str, Any]) -> None:
    doc_dir = repo_root / DOC_RUN_DIR
    doc_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = package.get("artifact_root")
    final_verdict_text = package.get("final_verdict")
    drive = package.get("drive_copy_status", {})
    skipped = package.get("skipped_legs", [])
    skipped_lines = [f"- `{item.get('leg')}`: {item.get('reason')}" for item in skipped] or ["- None."]
    readiness_lines = [f"- {reason}" for reason in merge_readiness_reasons(package)]
    evidence_lines = [
        f"- `{artifact_root}/scale_matrix.json`",
        f"- `{artifact_root}/scale_matrix.csv`",
        f"- `{artifact_root}/scale_matrix.md`",
        f"- `{artifact_root}/scale_matrix_before_after.json`",
        f"- `{artifact_root}/scale_matrix_before_after.csv`",
        f"- `{artifact_root}/scale_matrix_before_after.md`",
        f"- `{artifact_root}/proof_logic_meaning.md`",
        f"- `{artifact_root}/drive_manifest.json`",
        f"- `{REGISTRY_JSON.as_posix()}`",
        f"- `{REGISTRY_MD.as_posix()}`",
        f"- `{DOC_REPORT.as_posix()}`",
    ]
    command_lines = [
        f"- `{item.get('command', 'unknown command')}` -> returncode `{item.get('returncode')}`, reused `{item.get('reused_existing', False)}`, timed_out `{item.get('timed_out')}`"
        for item in package.get("commands", [])
    ] or ["- No proof commands were run."]
    common_logic = [
        "## Proof Logic + Meaning",
        "",
        f"Goal reached: larger labeled Sentinel guardrail scale packaging is `{final_verdict_text}`. The package discovered and registered a larger CICIDS/WebAttacks CSV, reused completed CPU proof receipts when present, and kept incomplete legs explicit.",
        "",
        "Previous state: the earlier guardrail proof was tiny-fixture-only. It could prove the runner shape, but not whether a larger real labeled CSV existed or how far the proof harness could scale on CPU.",
        "",
        "Technical logic utilized: the builder inspects label distributions, resolves CICIDS leading-space label headers, runs or reuses the existing labeled-domain proof runner, preserves raw/merged/deduped/confirmed/calibrated event views, and evaluates crash/core-touch receipts without changing reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, or hippocampus/familiarity behavior.",
        "",
        "Math / scoring logic:",
        "",
        *formulas_block(),
        "",
        "Philosophical meaning: Sentinel calibration is restraint before alarm, and this matrix is honesty before scale. It records what ran, what timed out, and what remains unproven.",
        "",
        "Why this is better: the proof trail now includes a dataset registry, larger-run receipts, explicit skip reasons, a bounded negative control, and Drive status instead of relying on a tiny fixture alone.",
        "",
        "How this moves Eidos closer to the north-star goal: Eidos Brain is a self-monitoring streaming intelligence codec. This milestone strengthens the reproducibility and incident-explanation side of that claim by making anomaly preservation and uncertainty visible on larger labeled data.",
        "",
        "Evidence:",
        "",
        *evidence_lines,
        "",
        "Remaining uncertainty:",
        "",
        *readiness_lines,
    ]
    journal_lines = [
        f"# Codex Journal - {RUN_DATE}",
        "",
        "## What happened today",
        "",
        "Ran the July 4 larger labeled Sentinel guardrail scale-matrix harness, added a stricter proof-side calibrated profile, and wrote before/after receipts.",
        "",
        "## What was accomplished",
        "",
        "- Added dataset discovery and registry receipts for CICIDS/WebAttacks CSV sources.",
        "- Added a stricter proof-side profile for operator-trust tuning without changing core Eidos behavior.",
        "- Wrote before/after delta receipts for off, current calibrated, and tuned calibrated views.",
        "- Kept skipped, partial, and optional GPU receipts explicit.",
        "",
        "## Tests and commands run",
        "",
        *command_lines,
        "",
        "## Problems encountered",
        "",
        *skipped_lines,
        "",
        "## What changed",
        "",
        "Proof-harness and reporting code changed. Generated artifact folders remain ignored by git.",
        "",
        "## What did not change",
        "",
        "Core model behavior did not change: reservoir dynamics, RLS updates, Sentinel anomaly policy, thresholds, compression behavior, and memory/familiarity behavior were untouched.",
        "",
        *common_logic,
        "",
        "## Artifacts generated",
        "",
        *evidence_lines,
        "",
        "## Google Drive archive status",
        "",
        f"- Copy attempted: `{drive.get('drive_copy_attempted')}`",
        f"- Copy success: `{drive.get('drive_copy_success')}`",
        f"- Drive root: `{drive.get('drive_root')}`",
        f"- Drive run dir: `{drive.get('drive_run_dir')}`",
        f"- Reason/status: `{drive.get('reason', drive.get('copy_status'))}`",
        "",
        "## Thoughts on improvement",
        "",
        "The next proof-sized improvement is to rerun any held or skipped leg with the same receipt shape and keep raw/off evidence beside calibrated reductions.",
        "",
        "## Where to improve next",
        "",
        "If a required leg is held, adjust only proof-side confirmation/calibration controls and rerun the July 4 matrix from a clean artifact folder.",
        "",
        "## Anything that stands out",
        "",
        f"Readiness reason: {'; '.join(merge_readiness_reasons(package))}",
        "",
        "## End-of-task summary",
        "",
        "1. Files changed: proof-harness script, tests, dataset registry/docs, and proof-run reports.",
        "2. Whether core behavior changed: no.",
        "3. Tests added or skipped: scale-matrix tests added; optional GPU leg skipped if CUDA unavailable.",
        "4. Repo-root commands run: see command list above.",
        f"5. Artifacts generated: `{artifact_root}`.",
        "6. Plain-language analysis written: yes.",
        "7. Journal entry written: yes.",
        "8. Google Drive copy status: recorded in drive manifest.",
        f"9. Known limitations: {'; '.join(merge_readiness_reasons(package))}",
        "10. Follow-up tasks not implemented: no core engine changes, no GPU run forced when CUDA is unavailable.",
        "11. Proof Logic + Meaning written: yes.",
        "12. Math/logic explanation included: yes.",
        "13. Philosophical meaning included: yes.",
        "14. Why this is better than previous state: larger dataset registry and bounded scale receipts replace tiny-only evidence.",
        "15. How this moves Eidos closer to the ultimate goal: it strengthens reproducible self-monitoring proof.",
        "16. Evidence files cited: see evidence list above.",
        f"17. Remaining uncertainty / unproven claims: {'; '.join(merge_readiness_reasons(package))}",
    ]
    analysis_lines = [
        f"# Plain-Language Test Analysis - {RUN_DATE}",
        "",
        "## What the task attempted",
        "",
        "This task attempted to reduce normal-only calibrated false positives on a larger labeled CICIDS/WebAttacks proof matrix without hiding raw/off evidence.",
        "",
        "## Why the test matters",
        "",
        "A guardrail that only passes on a tiny fixture is useful for wiring, but it is not enough evidence for larger proof. The scale matrix shows what happens when the same proof runner meets a larger real labeled source.",
        "",
        "## What was tested",
        "",
        "Dataset discovery, label-column resolution, balanced CPU proof, transition CPU proof, natural larger replay handling, normal-only negative control handling, optional GPU skip receipts, and core-touch policy.",
        "",
        "## What passed",
        "",
        "Completed proof rows are listed in the scale matrix with raw, merged, deduped, confirmed, and calibrated views visible side by side.",
        "",
        "## What failed or remained incomplete",
        "",
        *skipped_lines,
        "",
        "## Artifacts generated",
        "",
        *evidence_lines,
        "",
        "## What was saved locally",
        "",
        f"Local artifacts were saved under `{artifact_root}` and docs under `{DOC_RUN_DIR.as_posix()}`.",
        "",
        "## What was saved to Google Drive",
        "",
        f"Drive status: attempted `{drive.get('drive_copy_attempted')}`, success `{drive.get('drive_copy_success')}`, root `{drive.get('drive_root')}`.",
        "",
        "## What remains uncertain",
        "",
        *readiness_lines,
        "",
        "## What should happen next",
        "",
        "Make the natural replay complete through bounded attack-window sampling, checkpoint/resume, or a smaller accepted natural replay leg.",
        "",
        *common_logic,
    ]
    (doc_dir / "codex_journal.md").write_text("\n".join(journal_lines).rstrip() + "\n", encoding="utf-8")
    (doc_dir / "plain_language_test_analysis.md").write_text("\n".join(analysis_lines).rstrip() + "\n", encoding="utf-8")
    out_dir = repo_root / str(artifact_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "codex_journal.md").write_text("\n".join(journal_lines).rstrip() + "\n", encoding="utf-8")
    (out_dir / "plain_language_test_analysis.md").write_text("\n".join(analysis_lines).rstrip() + "\n", encoding="utf-8")


def write_proof_logic(out_dir: Path, package: Dict[str, Any]) -> None:
    selected = package.get("selected_dataset") or {}
    readiness_reasons = merge_readiness_reasons(package)
    lines = [
        "# Proof Logic + Meaning",
        "",
        "## Goal Reached",
        "",
        f"Status: `{package.get('final_verdict')}`. The task created a scale-ready Sentinel guardrail matrix and dataset registry, using `{selected.get('path', 'no larger dataset')}` when available.",
        "",
        "## Specific Logic / Math Used",
        "",
        *formulas_block(),
        "",
        "The runner preserves raw, merged, deduped, confirmed, and calibrated views. Calibration is only an additional proof view, not a replacement for raw evidence.",
        "",
        "## Why This Is Better",
        "",
        "The previous state could fall back to a tiny fixture. This package records dataset availability, exact labels, checksums, row counts, skipped legs, and the per-run evidence chain.",
        "",
        "## Evidence",
        "",
        "- `scale_matrix.json`",
        "- `scale_matrix.csv`",
        "- `scale_matrix.md`",
        "- `scale_matrix_before_after.json`",
        "- `scale_matrix_before_after.csv`",
        "- `scale_matrix_before_after.md`",
        "- `plots/*.svg`",
        "- `docs/proof/dataset_registry.json`",
        "- `docs/proof/dataset_registry.md`",
        "- per-run proof receipts under `runs/`",
        "- `core_touch_policy.json`",
        "- `drive_manifest.json`",
        "",
        "## What It Proves",
        "",
        "It proves the proof harness can find and account for larger labeled CICIDS/WebAttacks data and can preserve raw truth beside confirmation/calibration views for the legs that completed. It also proves the normal-only FP target status from actual saved rows rather than from hidden assumptions.",
        "",
        "## What It Does Not Prove",
        "",
        "It does not prove production readiness, every dataset variant, GPU behavior when CUDA is unavailable, or any change in core Eidos intelligence.",
        "",
        "## Remaining Uncertainty",
        "",
        *[f"- {reason}" for reason in readiness_reasons],
        "",
        "## North-Star Connection",
        "",
        "Eidos is not becoming more intelligent because it speaks less. It is becoming more intelligent only if it speaks less while preserving truth, preserving anomaly visibility, and making uncertainty auditable.",
    ]
    (out_dir / "proof_logic_meaning.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run(args: argparse.Namespace, *, repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    out_dir = resolve_path(args.out, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    search_roots = [Path(item) for item in args.search_root] if args.search_root else default_search_roots(repo_root)
    explicit_dataset = Path(args.dataset_file) if args.dataset_file else None
    if explicit_dataset is not None:
        search_roots = [explicit_dataset, *search_roots]
    candidates = discover_datasets(
        search_roots=search_roots,
        requested_label_column=args.label_column,
        attack_labels=args.attack_labels,
        repo_root=repo_root,
    )
    if explicit_dataset is not None and not any(Path(item.resolved_path).resolve() == resolve_path(explicit_dataset, repo_root).resolve() for item in candidates):
        candidates.append(
            inspect_csv_dataset(
                explicit_dataset,
                requested_label_column=args.label_column,
                attack_labels=args.attack_labels,
                repo_root=repo_root,
            )
        )
    selected = (
        inspect_csv_dataset(
            explicit_dataset,
            requested_label_column=args.label_column,
            attack_labels=args.attack_labels,
            repo_root=repo_root,
        )
        if explicit_dataset is not None
        else choose_larger_dataset(candidates)
    )
    if selected is not None and not selected.usable_for_labeled_proof:
        selected = None

    registry = write_dataset_registry(candidates=candidates, selected=selected, search_roots=search_roots, repo_root=repo_root)
    tiny_candidate = inspect_csv_dataset(
        Path("tests/fixtures/cicids_webattacks_tiny.csv"),
        requested_label_column=args.label_column,
        attack_labels=args.attack_labels,
        repo_root=repo_root,
    )
    normal_candidate: Optional[DatasetCandidate] = None
    if selected is not None:
        normal_candidate = write_normal_only_fixture(
            source=Path(selected.resolved_path),
            target=out_dir / "generated" / "normal_only_negative_control.csv",
            requested_label_column=args.label_column,
            repo_root=repo_root,
        )
        candidates = [*candidates, normal_candidate]

    cuda = cuda_receipt()
    plans = build_scale_leg_plan(
        selected=selected,
        normal_candidate=normal_candidate,
        tiny_candidate=tiny_candidate,
        cuda=cuda,
        natural_frames=args.natural_frames,
        normal_frames=args.normal_frames,
        repo_root=repo_root,
    )

    core_policy_pre = check_core_touch_policy.evaluate("origin/main", cwd=repo_root, include_worktree=True)
    core_touch_result = "passed" if core_policy_pre.get("passed") else "failed"
    write_json(out_dir / "core_touch_policy.json", core_policy_pre)
    check_core_touch_policy.write_report_md(out_dir / "core_touch_policy.md", core_policy_pre)

    rows: List[Dict[str, Any]] = []
    commands: List[Dict[str, Any]] = []
    skipped_legs: List[Dict[str, Any]] = []
    drive_root = discover_drive_root()
    for leg in plans:
        if not leg.should_run:
            skipped_legs.append(
                {
                    "leg": leg.name,
                    "requested_leg": leg.requested_leg,
                    "sample_mode": leg.sample_mode,
                    "frames": leg.frames,
                    "dataset_file": relpath(resolve_path(leg.dataset_file), repo_root),
                    "reason": leg.skip_reason,
                    "verdict": "ROW_SKIPPED",
                }
            )
            continue
        for profile in leg.run_profiles:
            run_dir = out_dir / "runs" / leg.name / profile
            stdout_log = logs_dir / f"{leg.name}_{profile}_stdout.txt"
            stderr_log = logs_dir / f"{leg.name}_{profile}_stderr.txt"
            receipts = partial_receipts(run_dir, repo_root=repo_root)
            if completed_run_receipts_exist(run_dir) and not args.rerun_existing:
                result = reused_command_result(run_dir)
            elif receipts and not args.rerun_existing:
                result = {
                    "command": "existing partial proof receipts reused as incomplete",
                    "resolved_command": None,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "",
                    "runtime_seconds": None,
                    "timed_out": True,
                    "timeout_seconds": args.timeout_seconds,
                    "timeout_reason": "existing partial run lacks run_manifest.json or labeled_metrics.json; rerun skipped because --rerun-existing was not set",
                    "python_process_exited": True,
                    "reused_existing": True,
                    "partial_receipts": receipts,
                }
            else:
                result = guardrails.run_labeled_command(
                    leg=leg,
                    profile=profile,
                    run_dir=run_dir,
                    label_column=args.label_column,
                    attack_labels=args.attack_labels,
                    seed=args.seed,
                    repo_root=repo_root,
                    drive_root=drive_root,
                    timeout_seconds=args.timeout_seconds,
                )
                stdout_log.write_text(result.get("stdout", ""), encoding="utf-8")
                stderr_log.write_text(result.get("stderr", ""), encoding="utf-8")
            command_record = {
                "leg": leg.name,
                "profile": profile,
                "command": result.get("command"),
                "returncode": result.get("returncode"),
                "runtime_seconds": result.get("runtime_seconds"),
                "timed_out": result.get("timed_out"),
                "stdout_log": relpath(stdout_log, repo_root),
                "stderr_log": relpath(stderr_log, repo_root),
                "reused_existing": result.get("reused_existing", False),
                "partial_receipts": result.get("partial_receipts", []),
            }
            commands.append(command_record)
            if result.get("timed_out"):
                skipped_legs.append(
                    {
                        "leg": leg.name,
                        "requested_leg": leg.requested_leg,
                        "sample_mode": leg.sample_mode,
                        "frames": leg.frames,
                        "dataset_file": relpath(resolve_path(leg.dataset_file), repo_root),
                        "reason": result.get("timeout_reason") or "proof command timed out",
                        "partial_receipts": result.get("partial_receipts", []),
                        "verdict": "ROW_SKIPPED",
                    }
                )
                continue
            candidate = candidate_for_leg(leg, candidates)
            rows.extend(
                profile_rows_from_run(
                    leg=leg,
                    run_dir=run_dir,
                    command_result=result,
                    candidate=candidate,
                    core_touch_result=core_touch_result,
                    repo_root=repo_root,
                )
            )

    branch_status = branch_preservation_status(
        "codex/eidos-post-merge-verification-and-sentinel-calibration-guardrails-2026-06-30",
        git_root=repo_root.parent,
    )
    verdict = final_verdict(
        selected_dataset=selected,
        rows=rows,
        skipped_legs=skipped_legs,
        core_policy=core_policy_pre,
        branch_pushed=bool(branch_status.get("branch_pushed")),
    )
    plots = write_plots(out_dir, rows)
    package = {
        "generated_at_utc": utc_now(),
        "repo": "bmparent/eidos",
        "artifact_root": relpath(out_dir, repo_root),
        "final_verdict": verdict,
        "allowed_row_verdicts": [
            "ROW_PASS",
            "ROW_HOLD_FP",
            "ROW_HOLD_RECALL",
            "ROW_SKIPPED",
            "ROW_CRASH",
            "ROW_CORE_TOUCH_FAIL",
        ],
        "allowed_verdicts": [
            "MERGE_READY_LARGER_LABELED_GUARDRAILS",
            "DATASET_MISSING_HOLD",
            "SCALE_HOLD_FALSE_POSITIVES",
            "SCALE_HOLD_RECALL_COLLAPSE",
            "SCALE_HOLD_CRASH_OR_CORE_TOUCH",
            "CALIBRATION_ONLY_NEEDS_TUNING",
        ],
        "normal_only_fp_targets": {
            "intermediate_pass_fp_per_10k": NORMAL_ONLY_FP_PER_10K_TARGET,
            "stretch_fp_per_10k": NORMAL_ONLY_FP_PER_10K_STRETCH_TARGET,
        },
        "selected_dataset": asdict(selected) if selected else None,
        "dataset_registry": registry,
        "tiny_dataset": asdict(tiny_candidate),
        "normal_only_dataset": asdict(normal_candidate) if normal_candidate else None,
        "cuda_receipt": cuda,
        "branch_preservation": branch_status,
        "rows": rows,
        "skipped_legs": skipped_legs,
        "commands": commands,
        "plots": plots,
        "formulas": {
            "FP_per_10k": "false_positive_events / benign_frames * 10000",
            "precision": "true_positive_events / max(true_positive_events + false_positive_events, 1)",
            "recall": "detected_attack_windows / max(total_attack_windows, 1)",
            "F1": "2 * precision * recall / max(precision + recall, epsilon)",
            "attack_window_coverage": "attack_windows_with_detection / total_attack_windows",
        },
        "core_touch_policy": core_policy_pre,
        "core_behavior_changed": False,
        "core_behavior_boundaries": {
            "reservoir_dynamics_changed": False,
            "rls_updates_changed": False,
            "raw_sentinel_anomaly_policy_changed": False,
            "compression_codec_behavior_changed": False,
            "hippocampus_memory_behavior_changed": False,
            "domain_adapter_math_changed": False,
        },
        "drive_copy_status": {
            "copy_status": "pending",
            "drive_root": str(drive_root) if drive_root else "unknown",
        },
    }
    before_after = write_before_after_artifacts(out_dir, rows, package)
    package["before_after_delta"] = {
        "json": "scale_matrix_before_after.json",
        "csv": "scale_matrix_before_after.csv",
        "md": "scale_matrix_before_after.md",
        "row_count": len(before_after.get("rows", [])),
        "plots": before_after.get("plots", []),
    }
    package["plots"] = [*plots, *before_after.get("plots", [])]
    write_json(out_dir / "scale_matrix.json", package)
    write_scale_csv(out_dir / "scale_matrix.csv", rows)
    write_scale_matrix_md(out_dir / "scale_matrix.md", package)
    write_proof_logic(out_dir, package)
    write_docs_report(repo_root, package)
    write_proof_run_companions(repo_root, package)

    drive_manifest = copy_allowlist_to_drive(out_dir=out_dir, run_date=args.run_date, drive_root=drive_root, repo_root=repo_root)
    write_json(out_dir / "drive_manifest.json", drive_manifest)
    drive_manifest = finalize_drive_manifest_copy(out_dir, drive_manifest)
    write_json(out_dir / "drive_manifest.json", drive_manifest)
    drive_manifest = finalize_drive_manifest_copy(out_dir, drive_manifest)
    write_json(out_dir / "drive_manifest.json", drive_manifest)
    package["drive_copy_status"] = {
        "drive_copy_attempted": drive_manifest.get("drive_copy_attempted"),
        "drive_copy_success": drive_manifest.get("drive_copy_success"),
        "copy_status": drive_manifest.get("copy_status"),
        "drive_root": drive_manifest.get("drive_root"),
        "drive_run_dir": drive_manifest.get("drive_run_dir"),
        "reason": drive_manifest.get("reason"),
    }
    write_json(out_dir / "scale_matrix.json", package)
    write_scale_matrix_md(out_dir / "scale_matrix.md", package)
    write_proof_logic(out_dir, package)
    write_docs_report(repo_root, package)
    write_proof_run_companions(repo_root, package)
    return package


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--dataset-file", type=Path, default=None)
    parser.add_argument("--search-root", action="append", default=[])
    parser.add_argument("--label-column", default="Label")
    parser.add_argument("--attack-labels", nargs="+", default=list(DEFAULT_ATTACK_LABELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--natural-frames", type=int, default=13000)
    parser.add_argument("--normal-frames", type=int, default=1000)
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    parser.add_argument("--run-date", default=RUN_DATE)
    parser.add_argument("--rerun-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    package = run(parse_args(argv))
    print(f"wrote Sentinel guardrail scale matrix: {package['artifact_root']}")
    print(f"final verdict: {package['final_verdict']}")
    return 1 if package["final_verdict"] == "SCALE_HOLD_CRASH_OR_CORE_TOUCH" else 0


if __name__ == "__main__":
    raise SystemExit(main())
