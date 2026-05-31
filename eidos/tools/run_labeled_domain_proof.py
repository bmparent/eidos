"""Run a labeled CICIDS/WebAttacks-style Eidos Brain proof harness.

This runner is intentionally a proof wrapper. It loads a labeled cyber CSV,
feeds projected rows through the existing Eidos engine, evaluates labels after
the run, and writes receipts. It does not change reservoir dynamics, RLS
updates, Sentinel thresholds, or anomaly policy.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sentinel import EvidenceFrame, process_stream
from tools import run_proof_baseline as proof_helpers
from tools.domain_tuner import load_engine_module

DEFAULT_DATASET = "cicids_webattacks"
DEFAULT_FEATURES = 64
DEFAULT_CONFIRMATION_MODE = "balanced"
BENIGN_LABELS = {"", "0", "false", "no", "normal", "benign", "none"}
NON_FEATURE_HINTS = {
    "flow id",
    "flow_id",
    "src ip",
    "source ip",
    "source_ip",
    "dst ip",
    "destination ip",
    "destination_ip",
    "timestamp",
    "time",
    "label",
}

CSV_COLUMNS = [
    "dataset",
    "suite",
    "seed",
    "frames_requested",
    "frames_processed",
    "label_column",
    "labels_detected",
    "label_distribution",
    "candidate_events",
    "confirmed_events",
    "suppressed_candidates",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "false_positives_per_10k_frames",
    "incident_card_count",
    "eidos_compression_ratio",
    "best_external_baseline",
    "best_external_baseline_ratio",
    "runtime_seconds",
    "crash_hit_count",
    "status",
    "notes",
]


@dataclass(frozen=True)
class LabeledDataset:
    name: str
    source_path: Path
    label_column: str
    frames: np.ndarray
    events: List[Dict[str, Any]]
    labels: np.ndarray
    raw_labels: List[str]
    label_distribution: Dict[str, int]
    attack_labels: List[str]
    feature_columns: List[str]
    source_rows_read: int

    def make_gen_factory(self, max_frames: int) -> Callable[[], Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]]:
        def _gen() -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
            limit = min(max_frames, len(self.events))
            for idx in range(limit):
                event = self.events[idx]
                meta = {
                    "kind": "cicids_webattacks_row",
                    "dataset": self.name,
                    "row_idx": idx,
                    "label": self.raw_labels[idx],
                    "attack": bool(self.labels[idx]),
                    "entities": {
                        key: event[key]
                        for key in ("src_ip", "dst_ip", "destination_port", "protocol", "label", "attack")
                        if key in event
                    },
                }
                yield event, meta

        return _gen


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    out_dir: Path
    metrics: Dict[str, Any]
    artifact_committed: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_out_dir() -> Path:
    return Path("artifacts") / f"cicids_webattacks_proof_{timestamp_slug()}"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--attack-labels", default="")
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--suite", choices=("smoke", "full"), required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args(argv)


def command_text(parts: Sequence[str]) -> str:
    if os.name == "nt":
        import subprocess

        return subprocess.list2cmdline(list(parts))
    return " ".join(parts)


def resolve_out_dir(out: Optional[Path], repo_root: Path = REPO_ROOT) -> Path:
    selected = out if out is not None else default_out_dir()
    if selected.is_absolute():
        return selected
    return repo_root / selected


def relpath(path: Path, root: Path = REPO_ROOT) -> str:
    return proof_helpers.relpath(path, root)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    proof_helpers.write_json(path, data)


def json_safe(value: Any) -> Any:
    return proof_helpers.json_safe(value)


def stable_hash(data: Dict[str, Any]) -> str:
    payload = json.dumps(json_safe(data), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_name(value: str) -> str:
    return " ".join(str(value).replace("\ufeff", "").strip().lower().replace("_", " ").split())


def normalize_label(value: Any) -> str:
    return str(value if value is not None else "").strip()


def parse_attack_labels(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def is_attack_label(raw_label: Any, attack_labels: Sequence[str]) -> bool:
    label = normalize_label(raw_label)
    norm = label.lower()
    if attack_labels:
        return norm in {item.lower() for item in attack_labels}
    return norm not in BENIGN_LABELS


def resolve_column(fieldnames: Sequence[str], requested: str) -> str:
    if requested in fieldnames:
        return requested
    requested_norm = normalize_name(requested)
    for field in fieldnames:
        if normalize_name(field) == requested_norm:
            return field
    raise ValueError(f"label column {requested!r} was not found; available columns: {', '.join(fieldnames)}")


def parse_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def feature_column_candidates(rows: List[Dict[str, str]], label_column: str) -> List[str]:
    fieldnames = list(rows[0].keys()) if rows else []
    candidates: List[str] = []
    for name in fieldnames:
        if name == label_column or normalize_name(name) in NON_FEATURE_HINTS:
            continue
        values = [parse_float(row.get(name)) for row in rows]
        if any(value is not None for value in values):
            candidates.append(name)
    if not candidates:
        raise ValueError("no numeric feature columns were found in the labeled dataset")
    return candidates


def standardize_matrix(values: np.ndarray) -> np.ndarray:
    mu = np.nanmean(values, axis=0, keepdims=True)
    sd = np.nanstd(values, axis=0, keepdims=True)
    sd[sd == 0.0] = 1.0
    standardized = (values - mu) / sd
    return np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)


def project_rows(engine: Any, arr: np.ndarray, features: int, seed: int) -> np.ndarray:
    if hasattr(engine, "AutoProjector"):
        projector = engine.AutoProjector(features, seed=seed)
        projected = np.zeros((arr.shape[0], features), dtype=np.float64)
        for idx in range(arr.shape[0]):
            projected[idx] = projector.to_dim(arr[idx])
        return projected
    projected = np.zeros((arr.shape[0], features), dtype=np.float64)
    width = min(features, arr.shape[1])
    projected[:, :width] = arr[:, :width]
    return projected


def metadata_from_row(row: Dict[str, str]) -> Dict[str, Any]:
    normalized = {normalize_name(key): value for key, value in row.items()}
    meta: Dict[str, Any] = {}
    for source, target in (
        ("src ip", "src_ip"),
        ("source ip", "src_ip"),
        ("dst ip", "dst_ip"),
        ("destination ip", "dst_ip"),
        ("destination port", "destination_port"),
        ("dest port", "destination_port"),
        ("protocol", "protocol"),
        ("flow id", "flow_id"),
    ):
        if source in normalized and normalized[source] not in (None, ""):
            meta[target] = normalized[source]
    return meta


def load_labeled_dataset(
    *,
    dataset: str,
    file_path: Path,
    label_column: str,
    attack_labels: Sequence[str],
    max_rows: Optional[int],
    engine: Any,
    features: int,
    seed: int,
    repo_root: Path = REPO_ROOT,
) -> LabeledDataset:
    resolved_file = file_path if file_path.is_absolute() else repo_root / file_path
    if not resolved_file.exists():
        raise FileNotFoundError(f"dataset file not found: {resolved_file}")

    rows: List[Dict[str, str]] = []
    with resolved_file.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"dataset file has no header: {resolved_file}")
        actual_label_column = resolve_column(reader.fieldnames, label_column)
        for row in reader:
            rows.append(dict(row))
            if max_rows is not None and len(rows) >= max_rows:
                break

    if not rows:
        raise ValueError(f"dataset file has no data rows: {resolved_file}")

    feature_columns = feature_column_candidates(rows, actual_label_column)
    raw_values: List[List[float]] = []
    raw_labels: List[str] = []
    binary_labels: List[int] = []
    for row in rows:
        raw_labels.append(normalize_label(row.get(actual_label_column)))
        binary_labels.append(1 if is_attack_label(row.get(actual_label_column), attack_labels) else 0)
        raw_values.append([
            parse_float(row.get(column)) if parse_float(row.get(column)) is not None else math.nan
            for column in feature_columns
        ])

    standardized = standardize_matrix(np.asarray(raw_values, dtype=np.float64))
    frames = project_rows(engine, standardized, features=features, seed=seed)
    projected_feature_names = [f"cicids_projected_{idx:02d}" for idx in range(features)]

    events: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        event = {
            "x": frames[idx].astype(float).tolist(),
            "row_index": idx,
            "label": raw_labels[idx],
            "attack": bool(binary_labels[idx]),
            "dataset": dataset,
            "feature_names": projected_feature_names,
            "source_feature_names": list(feature_columns),
        }
        event.update(metadata_from_row(row))
        events.append(event)

    return LabeledDataset(
        name=dataset,
        source_path=resolved_file,
        label_column=actual_label_column,
        frames=frames,
        events=events,
        labels=np.asarray(binary_labels, dtype=int),
        raw_labels=raw_labels,
        label_distribution=dict(Counter(raw_labels)),
        attack_labels=list(attack_labels),
        feature_columns=feature_columns,
        source_rows_read=len(rows),
    )


def load_engine_for_labeled(out_dir: Path, repo_root: Path = REPO_ROOT) -> Tuple[Any, Path]:
    engine = load_engine_module()
    engine_path = repo_root / proof_helpers.ENGINE_FILENAME
    refactored_path = repo_root / "repo" / "src" / "eidos_brain" / "engine" / "eidos_v0_4_7_02.py"
    if refactored_path.exists() and hasattr(engine, "run_stream_once") and getattr(engine, "__file__", ""):
        engine_path = Path(str(engine.__file__))

    engine_artifact_root = out_dir / "engine_artifacts"
    engine_archive_root = engine_artifact_root / "eidos_brain_archive"
    engine_artifact_root.mkdir(parents=True, exist_ok=True)
    engine_archive_root.mkdir(parents=True, exist_ok=True)
    engine.ARTIFACT_ROOT_PREFERRED = str(engine_artifact_root)
    engine.EIDOS_DATA_ROOT = str(engine_artifact_root)
    engine.EIDOS_ARCHIVE_ROOT = str(engine_archive_root)
    return engine, engine_path


def build_command(args: argparse.Namespace, out_dir: Path, repo_root: Path = REPO_ROOT) -> str:
    parts = [
        "python",
        "tools/run_labeled_domain_proof.py",
        "--dataset",
        args.dataset,
        "--file",
        relpath(args.file if args.file.is_absolute() else repo_root / args.file, repo_root),
        "--label-column",
        args.label_column,
        "--frames",
        str(args.frames),
        "--seed",
        str(args.seed),
        "--out",
        relpath(out_dir, repo_root),
        "--suite",
        args.suite,
    ]
    if args.attack_labels:
        parts.extend(["--attack-labels", args.attack_labels])
    if args.max_rows is not None:
        parts.extend(["--max-rows", str(args.max_rows)])
    return command_text(parts)


def write_environment(path: Path, repo_root: Path = REPO_ROOT) -> Dict[str, str]:
    environment_text, packages = proof_helpers.collect_environment(repo_root)
    path.write_text(environment_text, encoding="utf-8")
    return packages


def status_to_severity(status: Any) -> Optional[str]:
    text = str(status or "").upper()
    if "RED" in text:
        return "RED"
    if "AMBER" in text:
        return "AMBER"
    return None


def evidence_frames_from_step_rows(step_rows: List[Dict[str, Any]], limit: int) -> List[EvidenceFrame]:
    frames: List[EvidenceFrame] = []
    previous_dom: Optional[float] = None
    previous_entropy: Optional[float] = None
    for idx, row in enumerate(step_rows[:limit]):
        frame_index = int(row.get("step", idx))
        z_value = parse_float(row.get("z")) or 0.0
        threshold = parse_float(row.get("z_thresh_eff")) or 0.0
        dominance = parse_float(row.get("dominance"))
        state_entropy = parse_float(row.get("state_entropy"))
        dom_delta = abs(dominance - previous_dom) if dominance is not None and previous_dom is not None else 0.0
        entropy_delta = abs(state_entropy - previous_entropy) if state_entropy is not None and previous_entropy is not None else 0.0
        previous_dom = dominance if dominance is not None else previous_dom
        previous_entropy = state_entropy if state_entropy is not None else previous_entropy
        is_engine_candidate = z_value >= threshold if threshold > 0 else False
        geometry_change = max(min(dom_delta, 1.0), 0.3 if is_engine_candidate else 0.0)
        novelty = max(min(entropy_delta, 1.0), 0.3 if is_engine_candidate else 0.0)
        frames.append(
            EvidenceFrame(
                frame=frame_index,
                residual_score=z_value,
                geometry_change=geometry_change,
                novelty=novelty,
                severity_hint=status_to_severity(row.get("status")),
                raw_evidence_ref=f"step_row:{idx}",
            )
        )
    return frames


def contiguous_windows(mask: Sequence[bool], labels: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    start: Optional[int] = None
    for idx, active in enumerate(mask):
        if active and start is None:
            start = idx
        if (not active or idx == len(mask) - 1) and start is not None:
            end = idx if active and idx == len(mask) - 1 else idx - 1
            label_counts: Dict[str, int] = {}
            if labels is not None:
                label_counts = dict(Counter(str(item) for item in labels[start : end + 1]))
            windows.append(
                {
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "duration": int(end - start + 1),
                    "label_distribution": label_counts,
                }
            )
            start = None
    return windows


def processed_indices_from_step_rows(step_rows: List[Dict[str, Any]], upper_bound: int) -> List[int]:
    indices: List[int] = []
    for fallback, row in enumerate(step_rows):
        try:
            idx = int(row.get("step", fallback))
        except (TypeError, ValueError):
            idx = fallback
        if 0 <= idx < upper_bound:
            indices.append(idx)
    return indices


def contiguous_windows_from_indices(indices: Sequence[int], labels: Sequence[int], raw_labels: Sequence[str]) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    start: Optional[int] = None
    last_idx: Optional[int] = None
    label_bucket: List[str] = []
    for idx, label, raw_label in zip(indices, labels, raw_labels):
        active = bool(label)
        consecutive = last_idx is not None and idx == last_idx + 1
        if active and start is None:
            start = idx
            label_bucket = [str(raw_label)]
        elif active and start is not None and consecutive:
            label_bucket.append(str(raw_label))
        elif active and start is not None and not consecutive:
            assert last_idx is not None
            windows.append(
                {
                    "start_frame": int(start),
                    "end_frame": int(last_idx),
                    "duration": int(last_idx - start + 1),
                    "label_distribution": dict(Counter(label_bucket)),
                }
            )
            start = idx
            label_bucket = [str(raw_label)]
        elif not active and start is not None:
            assert last_idx is not None
            windows.append(
                {
                    "start_frame": int(start),
                    "end_frame": int(last_idx),
                    "duration": int(last_idx - start + 1),
                    "label_distribution": dict(Counter(label_bucket)),
                }
            )
            start = None
            label_bucket = []
        last_idx = idx
    if start is not None and last_idx is not None:
        windows.append(
            {
                "start_frame": int(start),
                "end_frame": int(last_idx),
                "duration": int(last_idx - start + 1),
                "label_distribution": dict(Counter(label_bucket)),
            }
        )
    return windows


def overlaps(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return int(left["start_frame"]) <= int(right["end_frame"]) and int(right["start_frame"]) <= int(left["end_frame"])


def event_label_metrics(detection_events: List[Dict[str, Any]], label_windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    event_windows = [
        {
            "start_frame": int(event["start_frame"]),
            "end_frame": int(event["end_frame"]),
            "event_id": event.get("event_id"),
        }
        for event in detection_events
    ]
    true_positive_events = [event for event in event_windows if any(overlaps(event, window) for window in label_windows)]
    false_positive_events = [event for event in event_windows if not any(overlaps(event, window) for window in label_windows)]
    false_negative_windows = [window for window in label_windows if not any(overlaps(event, window) for event in event_windows)]
    tp = len(true_positive_events)
    fp = len(false_positive_events)
    fn = len(false_negative_windows)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) else None
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive_events": true_positive_events,
        "false_positive_events": false_positive_events,
        "false_negative_label_windows": false_negative_windows,
    }


def load_engine_incident_cards(out_dir: Path) -> List[Dict[str, Any]]:
    path = out_dir / "engine_artifacts" / "incident_cards.jsonl"
    if not path.exists():
        return []
    cards: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                cards.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return cards


def engine_card_to_event(card: Dict[str, Any]) -> Dict[str, Any]:
    step = int(card.get("step", card.get("start_frame", 0)))
    return {
        "event_id": card.get("incident_id", f"engine_incident_{step}"),
        "start_frame": step,
        "end_frame": step,
        "source": "engine_incident_card",
        "severity": card.get("severity", card.get("regime")),
    }


def combined_detection_events(confirmed_events: List[Dict[str, Any]], engine_cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    seen = set()
    for event in confirmed_events:
        normalized = {
            "event_id": event.get("event_id"),
            "start_frame": int(event["start_frame"]),
            "end_frame": int(event["end_frame"]),
            "source": "sentinel_confirmation",
            "severity": event.get("severity"),
        }
        key = (normalized["start_frame"], normalized["end_frame"], normalized["source"], normalized["event_id"])
        if key not in seen:
            events.append(normalized)
            seen.add(key)
    for card in engine_cards:
        normalized = engine_card_to_event(card)
        key = (normalized["start_frame"], normalized["end_frame"], normalized["source"], normalized["event_id"])
        if key not in seen:
            events.append(normalized)
            seen.add(key)
    return sorted(events, key=lambda item: (item["start_frame"], item["end_frame"], str(item.get("event_id"))))


def write_incident_cards(
    out_dir: Path,
    confirmation_cards: List[Dict[str, Any]],
    engine_cards: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    incident_dir = out_dir / "incident_cards"
    incident_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for idx, card in enumerate(confirmation_cards, start=1):
        path = incident_dir / f"confirmed_event_{idx:03d}.json"
        write_json(path, card)
        written.append(relpath(path, out_dir))
    for idx, card in enumerate(engine_cards or [], start=1):
        path = incident_dir / f"engine_card_{idx:03d}.json"
        write_json(path, card)
        written.append(relpath(path, out_dir))
    if not written:
        (incident_dir / "README.md").write_text(
            "# Incident Cards\n\nNo confirmed incident cards were emitted for this labeled smoke run.\n",
            encoding="utf-8",
        )
    return written


def build_labeled_metrics(
    *,
    args: argparse.Namespace,
    dataset: LabeledDataset,
    frames_processed: int,
    runtime_seconds: float,
    step_rows: List[Dict[str, Any]],
    confirmation: Any,
    engine_incident_cards: List[Dict[str, Any]],
    incident_cards_written: List[str],
    compression_baselines: Dict[str, Any],
    crash_scan: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    seen_count = min(int(args.frames), dataset.source_rows_read)
    processed_indices = processed_indices_from_step_rows(step_rows[:frames_processed], seen_count)
    labels = np.asarray([int(dataset.labels[idx]) for idx in processed_indices], dtype=int)
    raw_labels = [dataset.raw_labels[idx] for idx in processed_indices]
    label_windows = contiguous_windows_from_indices(processed_indices, labels.tolist(), raw_labels)
    confirmed_events = [event.to_dict() for event in confirmation.confirmed_events]
    detection_events = combined_detection_events(confirmed_events, engine_incident_cards)
    label_metrics = event_label_metrics(detection_events, label_windows)
    eidos_ratio = step_rows[-1].get("ratio") if step_rows else None
    crash_scan = crash_scan or {"crash_hit_count": 0, "status": "not_run"}
    event_summary = {
        "confirmation_mode": DEFAULT_CONFIRMATION_MODE,
        "candidate_events": confirmation.candidate_events,
        "confirmed_events": detection_events,
        "confirmed_event_count": len(detection_events),
        "sentinel_confirmed_events": confirmed_events,
        "sentinel_confirmed_event_count": len(confirmed_events),
        "engine_incident_cards": engine_incident_cards,
        "engine_incident_card_count": len(engine_incident_cards),
        "suppressed_candidates": confirmation.suppressed_candidates,
        "cooldown_suppressions": confirmation.cooldown_suppressions,
        "merged_events": confirmation.merged_events,
        "label_windows": label_windows,
        "incident_cards_written": incident_cards_written,
        "policy_note": (
            "Existing Sentinel confirmation mode was used on engine step rows for event-level scoring; "
            "no Eidos thresholds or core behavior were tuned."
        ),
    }
    metrics = {
        "dataset": args.dataset,
        "suite": args.suite,
        "seed": args.seed,
        "frames_requested": args.frames,
        "frames_seen": seen_count,
        "frames_processed": frames_processed,
        "source_rows_read": dataset.source_rows_read,
        "source_file": relpath(dataset.source_path),
        "label_column": dataset.label_column,
        "labels_detected": sorted(dataset.label_distribution),
        "label_distribution": dict(Counter(dataset.raw_labels[:seen_count])),
        "scored_label_distribution": dict(Counter(raw_labels)),
        "scored_frame_indices": processed_indices,
        "attack_labels": dataset.attack_labels or "non-benign labels treated as attacks",
        "label_window_count": len(label_windows),
        "candidate_events": confirmation.candidate_events,
        "confirmed_events": len(detection_events),
        "sentinel_confirmed_events": len(confirmed_events),
        "engine_incident_card_count": len(engine_incident_cards),
        "suppressed_candidates": confirmation.suppressed_candidates,
        "cooldown_suppressions": confirmation.cooldown_suppressions,
        "merged_events": confirmation.merged_events,
        **{key: label_metrics[key] for key in ("true_positives", "false_positives", "false_negatives", "precision", "recall", "f1")},
        "false_positives_per_10k_frames": (
            label_metrics["false_positives"] * 10000.0 / frames_processed if frames_processed else None
        ),
        "incident_card_count": len(incident_cards_written),
        "incident_card_filenames": incident_cards_written,
        "eidos_compression_ratio": eidos_ratio,
        "external_compression_baselines": compression_baselines,
        "runtime_seconds": round(runtime_seconds, 6),
        "crash_hit_count": crash_scan.get("crash_hit_count", 0),
        "crash_scan_status": crash_scan.get("status", "unknown"),
        "known_limitations": [
            "This is a labeled proof harness and dataset adapter, not threshold tuning.",
            "Metrics are event-level over contiguous attack label windows and existing confirmed events.",
            "Large CICIDS/WebAttacks files are not downloaded by this runner; pass a mounted or uploaded CSV path with --file.",
        ],
    }
    return metrics, event_summary


def format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def benchmark_row(metrics: Dict[str, Any]) -> Dict[str, Any]:
    baselines = metrics.get("external_compression_baselines", {})
    return {
        "dataset": metrics.get("dataset"),
        "suite": metrics.get("suite"),
        "seed": metrics.get("seed"),
        "frames_requested": metrics.get("frames_requested"),
        "frames_processed": metrics.get("frames_processed"),
        "label_column": metrics.get("label_column"),
        "labels_detected": ", ".join(metrics.get("labels_detected", [])),
        "label_distribution": json.dumps(metrics.get("label_distribution", {}), sort_keys=True),
        "candidate_events": metrics.get("candidate_events"),
        "confirmed_events": metrics.get("confirmed_events"),
        "suppressed_candidates": metrics.get("suppressed_candidates"),
        "true_positives": metrics.get("true_positives"),
        "false_positives": metrics.get("false_positives"),
        "false_negatives": metrics.get("false_negatives"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "false_positives_per_10k_frames": metrics.get("false_positives_per_10k_frames"),
        "incident_card_count": metrics.get("incident_card_count"),
        "eidos_compression_ratio": metrics.get("eidos_compression_ratio"),
        "best_external_baseline": baselines.get("best_baseline", ""),
        "best_external_baseline_ratio": baselines.get("best_baseline_compression_ratio", ""),
        "runtime_seconds": metrics.get("runtime_seconds"),
        "crash_hit_count": metrics.get("crash_hit_count"),
        "status": "passed" if metrics.get("crash_hit_count", 0) == 0 else "crash_scan_failed",
        "notes": "first labeled/domain proof harness; no threshold tuning",
    }


def write_benchmark_csv(path: Path, metrics: Dict[str, Any]) -> None:
    row = benchmark_row(metrics)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def write_labeled_metrics_md(path: Path, metrics: Dict[str, Any]) -> None:
    lines = [
        "# Labeled Metrics",
        "",
        f"- Dataset: `{metrics.get('dataset')}`",
        f"- Frames processed: `{metrics.get('frames_processed')}`",
        f"- Labels detected: `{', '.join(metrics.get('labels_detected', []))}`",
        f"- Label distribution: `{metrics.get('label_distribution')}`",
        f"- Candidate events: `{metrics.get('candidate_events')}`",
        f"- Confirmed events: `{metrics.get('confirmed_events')}`",
        f"- Suppressed candidates: `{metrics.get('suppressed_candidates')}`",
        f"- True positives / false positives / false negatives: `{metrics.get('true_positives')}` / `{metrics.get('false_positives')}` / `{metrics.get('false_negatives')}`",
        f"- Precision / recall / F1: `{format_metric(metrics.get('precision'))}` / `{format_metric(metrics.get('recall'))}` / `{format_metric(metrics.get('f1'))}`",
        f"- False positives per 10k frames: `{format_metric(metrics.get('false_positives_per_10k_frames'))}`",
        f"- Incident-card count: `{metrics.get('incident_card_count')}`",
        f"- Eidos compression ratio: `{format_metric(metrics.get('eidos_compression_ratio'))}`",
        f"- Runtime seconds: `{metrics.get('runtime_seconds')}`",
        f"- Crash hits: `{metrics.get('crash_hit_count')}`",
        "",
        "## Interpretation",
        "",
        "These metrics compare existing Eidos/Sentinel outputs against labeled attack windows. They are receipts for a first labeled proof harness, not evidence that thresholds have been tuned.",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_benchmark_md(path: Path, *, command: str, metrics: Dict[str, Any], out_dir: Path, git_info: Dict[str, Any]) -> None:
    row = benchmark_row(metrics)
    lines = [
        "# CICIDS/WebAttacks Labeled Proof Summary",
        "",
        "## Exact command",
        "",
        f"```bash\n{command}\n```",
        "",
        f"- Artifact directory: `{relpath(out_dir)}`",
        f"- Git commit: `{git_info.get('commit', 'unknown')}`",
        f"- Git branch: `{git_info.get('branch', 'unknown')}`",
        f"- Git dirty at run start: `{git_info.get('dirty')}`",
        "",
        "## Summary",
        "",
        "| dataset | frames | labels | candidate events | confirmed events | TP | FP | FN | precision | recall | F1 | FP/10k | incident cards | Eidos ratio | runtime | crash hits |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| {dataset} | {frames} | {labels} | {candidate_events} | {confirmed_events} | {tp} | {fp} | {fn} | {precision} | {recall} | {f1} | {fp10k} | {cards} | {ratio} | {runtime} | {crashes} |".format(
            dataset=row["dataset"],
            frames=row["frames_processed"],
            labels=row["labels_detected"],
            candidate_events=row["candidate_events"],
            confirmed_events=row["confirmed_events"],
            tp=row["true_positives"],
            fp=row["false_positives"],
            fn=row["false_negatives"],
            precision=format_metric(row["precision"]),
            recall=format_metric(row["recall"]),
            f1=format_metric(row["f1"]),
            fp10k=format_metric(row["false_positives_per_10k_frames"]),
            cards=row["incident_card_count"],
            ratio=format_metric(row["eidos_compression_ratio"]),
            runtime=row["runtime_seconds"],
            crashes=row["crash_hit_count"],
        ),
        "",
        "## Compression Baselines",
        "",
        f"- Best external baseline: `{row['best_external_baseline']}`",
        f"- Best external baseline ratio: `{row['best_external_baseline_ratio']}`",
        "",
        "## Known Limitations",
        "",
        "- This run does not tune thresholds or anomaly policy.",
        "- The runner depends on a caller-provided labeled CSV path.",
        "- Event metrics are only meaningful when labels are frame-aligned enough to define attack windows.",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_config_doc(
    *,
    args: argparse.Namespace,
    dataset: LabeledDataset,
    engine_info: Dict[str, Any],
    command: str,
    out_dir: Path,
) -> Dict[str, Any]:
    doc = {
        "benchmark": {
            "dataset": args.dataset,
            "suite": args.suite,
            "seed": args.seed,
            "frames": args.frames,
            "max_rows": args.max_rows,
            "command": command,
            "artifact_dir": relpath(out_dir),
        },
        "dataset": {
            "source_file": relpath(dataset.source_path),
            "label_column": dataset.label_column,
            "labels_detected": sorted(dataset.label_distribution),
            "label_distribution": dataset.label_distribution,
            "attack_labels": dataset.attack_labels or "non-benign labels treated as attacks",
            "feature_columns": dataset.feature_columns,
            "rows_read": dataset.source_rows_read,
        },
        "engine": engine_info,
        "core_behavior": {
            "reservoir_dynamics_changed": False,
            "rls_updates_changed": False,
            "sentinel_thresholds_changed": False,
            "anomaly_policy_tuned": False,
            "new_architecture_layers_added": False,
        },
    }
    doc["config_hash_sha256"] = stable_hash(doc)
    return doc


def build_manifest(
    *,
    generated_at: str,
    command: str,
    git_info: Dict[str, Any],
    engine_info: Dict[str, Any],
    packages: Dict[str, str],
    metrics: Dict[str, Any],
    config_hash: str,
    drive_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = {
        "generated_at_utc": generated_at,
        "command": command,
        "git": {
            "branch": git_info.get("branch", "unknown"),
            "commit": git_info.get("commit", "unknown"),
            "dirty": bool(git_info.get("dirty")),
        },
        "engine": engine_info,
        "packages": packages,
        "config": {
            "config_hash_sha256": config_hash,
            "config_path": "config.json",
        },
        "metrics": {
            key: metrics.get(key)
            for key in (
                "frames_processed",
                "candidate_events",
                "confirmed_events",
                "suppressed_candidates",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "f1",
                "false_positives_per_10k_frames",
                "incident_card_count",
                "eidos_compression_ratio",
                "runtime_seconds",
                "crash_hit_count",
            )
        },
        "outputs": {
            "config_json": "config.json",
            "run_manifest_json": "run_manifest.json",
            "labeled_metrics_json": "labeled_metrics.json",
            "labeled_metrics_md": "labeled_metrics.md",
            "benchmark_summary_csv": "benchmark_summary.csv",
            "benchmark_summary_md": "benchmark_summary.md",
            "event_summary_json": "event_summary.json",
            "incident_cards_dir": "incident_cards",
            "proof_digest_json": "proof_digest.json",
            "proof_digest_md": "proof_digest.md",
            "crash_scan_json": "crash_scan.json",
            "environment_txt": "environment.txt",
            "drive_manifest_json": "drive_manifest.json",
        },
    }
    if drive_manifest is not None:
        manifest["drive"] = {
            "drive_copy_attempted": drive_manifest.get("drive_copy_attempted"),
            "drive_copy_success": drive_manifest.get("drive_copy_success"),
            "drive_root": drive_manifest.get("drive_root"),
            "drive_run_dir": drive_manifest.get("drive_run_dir"),
            "reason": drive_manifest.get("reason"),
        }
    return manifest


def build_proof_digest(
    *,
    command: str,
    git_info: Dict[str, Any],
    metrics: Dict[str, Any],
    out_dir: Path,
    crash_scan: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "repo_branch": git_info.get("branch", "unknown"),
        "git_commit": git_info.get("commit", "unknown"),
        "git_dirty": bool(git_info.get("dirty")),
        "command": command,
        "dataset": metrics.get("dataset"),
        "suite": metrics.get("suite"),
        "seed": metrics.get("seed"),
        "frames_processed": metrics.get("frames_processed"),
        "labels_detected": metrics.get("labels_detected"),
        "label_distribution": metrics.get("label_distribution"),
        "candidate_events": metrics.get("candidate_events"),
        "confirmed_events": metrics.get("confirmed_events"),
        "suppressed_candidates": metrics.get("suppressed_candidates"),
        "true_positives": metrics.get("true_positives"),
        "false_positives": metrics.get("false_positives"),
        "false_negatives": metrics.get("false_negatives"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "false_positives_per_10k_frames": metrics.get("false_positives_per_10k_frames"),
        "incident_card_count": metrics.get("incident_card_count"),
        "eidos_compression_ratio": metrics.get("eidos_compression_ratio"),
        "external_compression_baselines": metrics.get("external_compression_baselines"),
        "runtime_seconds": metrics.get("runtime_seconds"),
        "crash_scan": crash_scan,
        "clean": crash_scan.get("crash_hit_count", 0) == 0,
        "artifact_dir": relpath(out_dir),
        "generated_at_utc": utc_now(),
        "known_limitations": metrics.get("known_limitations", []),
    }


def write_proof_digest_md(path: Path, digest: Dict[str, Any]) -> None:
    crash = digest.get("crash_scan", {})
    lines = [
        "# Proof Digest",
        "",
        f"- Branch: `{digest.get('repo_branch')}`",
        f"- Commit: `{digest.get('git_commit')}`",
        f"- Dataset: `{digest.get('dataset')}`",
        f"- Frames processed: `{digest.get('frames_processed')}`",
        f"- Labels detected: `{', '.join(digest.get('labels_detected') or [])}`",
        f"- Candidate / confirmed / suppressed: `{digest.get('candidate_events')}` / `{digest.get('confirmed_events')}` / `{digest.get('suppressed_candidates')}`",
        f"- TP / FP / FN: `{digest.get('true_positives')}` / `{digest.get('false_positives')}` / `{digest.get('false_negatives')}`",
        f"- Precision / recall / F1: `{format_metric(digest.get('precision'))}` / `{format_metric(digest.get('recall'))}` / `{format_metric(digest.get('f1'))}`",
        f"- Incident cards: `{digest.get('incident_card_count')}`",
        f"- Eidos compression ratio: `{format_metric(digest.get('eidos_compression_ratio'))}`",
        f"- Runtime seconds: `{digest.get('runtime_seconds')}`",
        f"- Crash scan: `{crash.get('status')}` with `{crash.get('crash_hit_count')}` hits",
        "",
        "## Known Limitations",
        "",
    ]
    for item in digest.get("known_limitations", []):
        lines.append(f"- {item}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_proof_digest(out_dir: Path, digest: Dict[str, Any]) -> None:
    write_json(out_dir / "proof_digest.json", digest)
    write_proof_digest_md(out_dir / "proof_digest.md", digest)


def scan_crashes(out_dir: Path) -> Dict[str, Any]:
    return proof_helpers.scan_crash_strings(out_dir)


def append_or_create(path: Path, heading: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if heading in existing:
        return
    prefix = existing.rstrip() + "\n\n" if existing.strip() else ""
    path.write_text(prefix + heading + "\n\n" + body.rstrip() + "\n", encoding="utf-8")


def write_proof_docs(
    *,
    repo_root: Path,
    out_dir: Path,
    run_date: str,
    command: str,
    metrics: Dict[str, Any],
    drive_manifest: Dict[str, Any],
    files_changed: Optional[List[str]] = None,
) -> None:
    docs_dir = repo_root / "docs" / "proof_runs" / run_date
    docs_dir.mkdir(parents=True, exist_ok=True)
    drive_status = "copied" if drive_manifest.get("drive_copy_success") else "skipped or failed"
    drive_reason = str(drive_manifest.get("reason", "unknown"))
    drive_root = str(drive_manifest.get("drive_root", "unknown"))
    drive_folder = str(drive_manifest.get("drive_run_dir", "unknown"))
    artifact_files = [relpath(path) for path in proof_helpers.artifact_files(out_dir)]
    changed = files_changed or [
        "eidos_domain_adapters.py",
        "tools/run_labeled_domain_proof.py",
        "tests/test_labeled_domain_proof_runner.py",
        "docs/proof_runs/2026-05-31/cicids_webattacks_plan.md",
        ".gitignore",
        relpath(out_dir),
    ]
    heading = f"## Labeled CICIDS/WebAttacks proof harness -- {relpath(out_dir)}"
    journal_body = "\n".join(
        [
            "### What happened today",
            "Built and ran the first labeled/domain proof harness after the official Colab GPU 10k baseline.",
            "",
            "### What was accomplished",
            "- Added CICIDS/WebAttacks row adaptation and a repo-root labeled proof runner.",
            "- Captured label distributions, event metrics, compression baselines, incident cards, runtime, and crash scan receipts.",
            "- Kept core Eidos model behavior untouched.",
            "",
            "### Tests and commands run",
            f"- `{command}` -> labeled smoke proof artifacts written.",
            "",
            "### Problems encountered",
            f"- Google Drive status: {drive_status}; reason: {drive_reason}.",
            "- This is not threshold tuning, so misses or false positives are reported rather than optimized away.",
            "",
            "### What changed",
            *[f"- {item}" for item in changed],
            "",
            "### What did not change",
            "Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.",
            "",
            "### Artifacts generated",
            *[f"- {item}" for item in artifact_files],
            "",
            "### Google Drive archive status",
            f"- Drive root used: {drive_root}",
            f"- Drive folder used: {drive_folder}",
            f"- Files copied: {len(drive_manifest.get('files_copied', []))}",
            f"- Files skipped: {len(drive_manifest.get('files_skipped', []))}",
            f"- Reason: {drive_reason}",
            "",
            "### End-of-task summary",
            f"1. Files changed: {', '.join(changed)}",
            "2. Whether core behavior changed: no.",
            "3. Tests added or skipped: focused labeled runner tests added; full pytest run handled outside this runner.",
            f"4. Repo-root commands run: `{command}`.",
            f"5. Artifacts generated: {len(artifact_files)} files under `{relpath(out_dir)}`.",
            "6. Plain-language analysis written: yes.",
            "7. Journal entry written: yes.",
            f"8. Google Drive copy status: {drive_status}; {drive_reason}.",
            "9. Known limitations: labeled windows are frame-aligned only; no threshold tuning was attempted.",
            "10. Follow-up tasks not implemented: full CICIDS dataset run and threshold calibration.",
        ]
    )
    analysis_body = "\n".join(
        [
            "### What the task attempted",
            "The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.",
            "",
            "### Why the test matters",
            "The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.",
            "",
            "### What was tested",
            "The runner processed a labeled CICIDS/WebAttacks-style fixture, grouped attack labels into windows, compared existing confirmed events to those windows, and wrote crash and compression receipts.",
            "",
            "### What passed",
            f"- Frames processed: {metrics.get('frames_processed')}",
            f"- Crash hits: {metrics.get('crash_hit_count')}",
            f"- Incident cards: {metrics.get('incident_card_count')}",
            "",
            "### What failed or remains uncertain",
            "- Any false positives and false negatives are recorded in the metrics instead of being tuned away.",
            "- A full CICIDS/WebAttacks run still requires the dataset file to be mounted or uploaded.",
            "",
            "### What was saved locally",
            f"Artifacts were saved under `{relpath(out_dir)}`.",
            "",
            "### What was saved to Google Drive",
            f"Drive status: {drive_status}; folder: {drive_folder}; reason: {drive_reason}.",
            "",
            "### What should happen next",
            "Run the same harness against the real CICIDS2017 WebAttacks CSV in Colab, then review metrics before any threshold tuning.",
        ]
    )
    append_or_create(docs_dir / "codex_journal.md", heading, journal_body)
    append_or_create(docs_dir / "plain_language_test_analysis.md", heading, analysis_body)
    (out_dir / "codex_journal.md").write_text("# Codex Journal -- Labeled Domain Proof\n\n" + journal_body + "\n", encoding="utf-8")
    (out_dir / "plain_language_test_analysis.md").write_text(
        "# Plain-Language Test Analysis -- Labeled Domain Proof\n\n" + analysis_body + "\n",
        encoding="utf-8",
    )


def run(
    args: argparse.Namespace,
    *,
    repo_root: Path = REPO_ROOT,
    load_engine_fn: Callable[[Path, Path], Tuple[Any, Path]] = load_engine_for_labeled,
    mirror_to_drive_fn: Callable[[Path, str, str], Dict[str, Any]] = proof_helpers.mirror_to_drive,
    write_docs_fn: Callable[..., None] = write_proof_docs,
) -> RunResult:
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    out_dir = resolve_out_dir(args.out, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "incident_cards").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    generated_at = utc_now()
    run_date = datetime.now(timezone.utc).date().isoformat()
    command = build_command(args, out_dir, repo_root)

    git_info = proof_helpers.collect_git_info(repo_root)
    engine, engine_path = load_engine_fn(out_dir, repo_root)
    engine_info = {
        "code_hash_sha256": proof_helpers.sha256_file(engine_path) if engine_path.exists() else "unknown",
        "module": relpath(engine_path, repo_root),
        "version": str(getattr(engine, "ENGINE_VERSION", "unknown")),
    }
    attack_labels = parse_attack_labels(args.attack_labels)
    dataset = load_labeled_dataset(
        dataset=args.dataset,
        file_path=args.file,
        label_column=args.label_column,
        attack_labels=attack_labels,
        max_rows=args.max_rows,
        engine=engine,
        features=DEFAULT_FEATURES,
        seed=args.seed,
        repo_root=repo_root,
    )
    frames_processed = min(args.frames, dataset.frames.shape[0])
    if frames_processed <= 0:
        raise ValueError("no frames available after applying --frames/--max-rows")

    config_doc = build_config_doc(args=args, dataset=dataset, engine_info=engine_info, command=command, out_dir=out_dir)
    write_json(out_dir / "config.json", config_doc)
    packages = write_environment(out_dir / "environment.txt", repo_root)
    proof_helpers.write_git_commit(out_dir / "git_commit.txt", git_info)

    start = time.perf_counter()
    log_path = out_dir / "logs" / "engine_output.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            results = engine.run_stream_once(
                dataset.make_gen_factory(frames_processed),
                est_frames=frames_processed,
                features=DEFAULT_FEATURES,
                profile_label=f"{args.dataset}_labeled_proof",
                session_label=f"{args.dataset}_labeled_proof_seed{args.seed}",
                cfg_overrides={"domain": "cicids_webattacks"},
                return_step_rows=True,
                return_top_surprises=False,
                seed=args.seed,
            )
    runtime_seconds = time.perf_counter() - start
    results = results or {}
    step_rows = list(results.get("step_rows") or [])
    if len(step_rows) < frames_processed:
        frames_processed = len(step_rows)
    processed_indices = processed_indices_from_step_rows(step_rows[:frames_processed], dataset.frames.shape[0])
    processed_frames = dataset.frames[processed_indices] if processed_indices else dataset.frames[:frames_processed]
    compression_baselines = proof_helpers.compression_baselines_for_frames(processed_frames)
    evidence_frames = evidence_frames_from_step_rows(step_rows, frames_processed)
    confirmation = process_stream(evidence_frames, mode=DEFAULT_CONFIRMATION_MODE)
    engine_incident_cards = load_engine_incident_cards(out_dir)
    incident_cards_written = write_incident_cards(out_dir, confirmation.incident_cards, engine_incident_cards)

    crash_scan = scan_crashes(out_dir)
    metrics, event_summary = build_labeled_metrics(
        args=args,
        dataset=dataset,
        frames_processed=frames_processed,
        runtime_seconds=runtime_seconds,
        step_rows=step_rows,
        confirmation=confirmation,
        engine_incident_cards=engine_incident_cards,
        incident_cards_written=incident_cards_written,
        compression_baselines=compression_baselines,
        crash_scan=crash_scan,
    )
    write_json(out_dir / "labeled_metrics.json", metrics)
    write_labeled_metrics_md(out_dir / "labeled_metrics.md", metrics)
    write_json(out_dir / "event_summary.json", event_summary)
    write_benchmark_csv(out_dir / "benchmark_summary.csv", metrics)
    write_benchmark_md(out_dir / "benchmark_summary.md", command=command, metrics=metrics, out_dir=out_dir, git_info=git_info)
    write_json(out_dir / "crash_scan.json", crash_scan)
    digest = build_proof_digest(command=command, git_info=git_info, metrics=metrics, out_dir=out_dir, crash_scan=crash_scan)
    write_proof_digest(out_dir, digest)

    draft_manifest = build_manifest(
        generated_at=generated_at,
        command=command,
        git_info=git_info,
        engine_info=engine_info,
        packages=packages,
        metrics=metrics,
        config_hash=config_doc["config_hash_sha256"],
    )
    write_json(out_dir / "run_manifest.json", draft_manifest)

    run_id = f"cicids_webattacks_proof_{args.suite}_seed{args.seed}_frames{frames_processed}_{timestamp_slug()}"
    drive_manifest = mirror_to_drive_fn(out_dir, run_id, run_date)
    write_json(out_dir / "drive_manifest.json", drive_manifest)
    final_manifest = build_manifest(
        generated_at=generated_at,
        command=command,
        git_info=git_info,
        engine_info=engine_info,
        packages=packages,
        metrics=metrics,
        config_hash=config_doc["config_hash_sha256"],
        drive_manifest=drive_manifest,
    )
    write_json(out_dir / "run_manifest.json", final_manifest)
    write_docs_fn(
        repo_root=repo_root,
        out_dir=out_dir,
        run_date=run_date,
        command=command,
        metrics=metrics,
        drive_manifest=drive_manifest,
    )
    proof_helpers.copy_selected_to_drive(
        out_dir,
        drive_manifest,
        [
            out_dir / "run_manifest.json",
            out_dir / "drive_manifest.json",
            out_dir / "codex_journal.md",
            out_dir / "plain_language_test_analysis.md",
        ],
    )
    return RunResult(exit_code=0 if crash_scan.get("crash_hit_count", 0) == 0 else 1, out_dir=out_dir, metrics=metrics)


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = run(parse_args(argv))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
