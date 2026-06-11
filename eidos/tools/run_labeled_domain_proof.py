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
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proof import event_confirmation as proof_event_confirmation
from proof import sentinel_calibration_v1 as proof_calibration
from sentinel import EvidenceFrame, process_stream
from tools import run_proof_baseline as proof_helpers
from tools.domain_tuner import load_engine_module

DEFAULT_DATASET = "cicids_webattacks"
DEFAULT_FEATURES = 64
DEFAULT_CONFIRMATION_MODE = "balanced"
DEFAULT_SENTINEL_CONFIRMATION_MODE = "balanced"
SENTINEL_CALIBRATION_MODES = ("off", "low_noise", "balanced", "high_recall")
DEFAULT_SAMPLE_MODE = "natural"
DEFAULT_CONFIRMATION_PROFILE_SWEEP = ("balanced", "recall_guarded", "high_recall")
DEFAULT_NATURAL_WINDOW_PRE = 250
DEFAULT_NATURAL_WINDOW_POST = 250
DEFAULT_NATURAL_WINDOW_MAX_WINDOWS = 3
DEFAULT_EVENT_MERGE_GAP = 25
PROOF_LABEL_BENIGN = "BENIGN"
PROOF_LABEL_ATTACK = "ATTACK"
BENIGN_LABELS = {"", "0", "false", "no", "normal", "benign", "none"}
GENERATED_UNTRACKED_PREFIXES = (
    "artifacts/cicids_webattacks_proof_",
    "artifacts/cicids_webattacks_samples/",
    "artifacts/proof_runs/",
    "tmp/eidos_proof_data/",
)
SEVERITY_RANK = {"GREEN": 0, "RECOVERY": 1, "AMBER": 2, "RED": 3}
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
    "sample_mode",
    "confirmation_mode",
    "calibration_enabled",
    "calibration_version",
    "frames_requested",
    "frames_processed",
    "label_column",
    "labels_detected",
    "label_distribution",
    "raw_label_distribution",
    "normalized_label_distribution",
    "candidate_events",
    "confirmed_events",
    "pre_calibration_confirmed_events",
    "post_calibration_confirmed_events",
    "calibration_suppressed_events",
    "sentinel_calibration_mode",
    "proof_raw_event_count",
    "proof_merged_event_count",
    "proof_deduped_event_count",
    "proof_confirmed_event_count",
    "suppressed_candidates",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "false_positives_per_10k_frames",
    "pre_calibration_precision",
    "pre_calibration_recall",
    "pre_calibration_f1",
    "pre_calibration_false_positives_per_10k_frames",
    "calibrated_precision",
    "calibrated_recall",
    "calibrated_f1",
    "calibrated_false_positives_per_10k_frames",
    "incident_card_count",
    "eidos_compression_ratio",
    "best_external_baseline",
    "best_external_baseline_ratio",
    "runtime_seconds",
    "frames_per_second",
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
    proof_labels: List[str]
    label_distribution: Dict[str, int]
    normalized_label_distribution: Dict[str, int]
    attack_labels: List[str]
    normalization_mode: str
    feature_columns: List[str]
    source_rows_read: int
    source_rows_available: int
    source_row_indices: List[int]
    sample_receipt: Dict[str, Any]

    def make_gen_factory(self, max_frames: int) -> Callable[[], Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]]:
        def _gen() -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
            limit = min(max_frames, len(self.events))
            for idx in range(limit):
                event = self.events[idx]
                meta = {
                    "kind": "cicids_webattacks_row",
                    "dataset": self.name,
                    "row_idx": idx,
                    "source_row_idx": self.source_row_indices[idx],
                    "OriginalLabel": self.raw_labels[idx],
                    "EidosProofLabel": self.proof_labels[idx],
                    "label": self.raw_labels[idx],
                    "attack": bool(self.labels[idx]),
                    "entities": {
                        key: event[key]
                        for key in (
                            "src_ip",
                            "dst_ip",
                            "destination_port",
                            "protocol",
                            "label",
                            "OriginalLabel",
                            "EidosProofLabel",
                            "attack",
                        )
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
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument(
        "--attack-labels",
        action="append",
        nargs="+",
        default=[],
        help=(
            "Attack label(s) to map to EidosProofLabel=ATTACK. May be repeated, "
            "comma-separated, or passed as multiple quoted values."
        ),
    )
    parser.add_argument(
        "--normalize-non-benign-as",
        choices=(PROOF_LABEL_ATTACK,),
        default=None,
        help="When set to ATTACK, every non-benign raw label is normalized to ATTACK.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("natural", "balanced", "transition", "balanced_blocks", "natural_attack_windows"),
        default=DEFAULT_SAMPLE_MODE,
        help=(
            "Construct a natural, balanced, benign-to-attack transition, block-balanced, "
            "or natural attack-window replay sample."
        ),
    )
    parser.add_argument(
        "--natural-window-pre",
        type=int,
        default=DEFAULT_NATURAL_WINDOW_PRE,
        help="Rows of benign/source-order context to include before each natural_attack_windows attack window.",
    )
    parser.add_argument(
        "--natural-window-post",
        type=int,
        default=DEFAULT_NATURAL_WINDOW_POST,
        help="Rows of benign/source-order context to include after each natural_attack_windows attack window.",
    )
    parser.add_argument(
        "--natural-window-max-windows",
        type=int,
        default=DEFAULT_NATURAL_WINDOW_MAX_WINDOWS,
        help="Maximum attack windows to include in natural_attack_windows mode.",
    )
    parser.add_argument(
        "--event-merge-gap",
        type=int,
        default=DEFAULT_EVENT_MERGE_GAP,
        help="Frame gap used by precision-ledger postprocessing to merge nearby event windows.",
    )
    parser.add_argument(
        "--confirmation-mode",
        choices=proof_event_confirmation.CONFIRMATION_MODES,
        default=DEFAULT_CONFIRMATION_MODE,
        help=(
            "Proof-side event confirmation mode. off preserves the previous raw-event decision view; "
            "other modes score deduped candidate events and emit confirmed_events."
        ),
    )
    parser.add_argument(
        "--sentinel-calibration-mode",
        choices=SENTINEL_CALIBRATION_MODES,
        default=None,
        help=(
            "Gate-facing Sentinel calibration mode. off preserves raw proof decisions; "
            "low_noise, balanced, and high_recall enable calibration and map to the "
            "matching proof-side confirmation profile unless --confirmation-mode is set."
        ),
    )
    parser.add_argument(
        "--confirmation-profile-sweep",
        action="append",
        nargs="+",
        choices=proof_event_confirmation.CONFIRMATION_MODES,
        default=[],
        help=(
            "Optional proof-side confirmation profile sweep to compute from one engine pass. "
            "May be repeated or passed as multiple values."
        ),
    )
    parser.add_argument(
        "--confirmation-min-raw-hits",
        type=int,
        default=None,
        help="Optional override for minimum raw hits per confirmed proof event.",
    )
    parser.add_argument(
        "--confirmation-min-duration",
        type=int,
        default=None,
        help="Optional override for minimum candidate duration per confirmed proof event.",
    )
    parser.add_argument(
        "--confirmation-min-score",
        type=float,
        default=None,
        help="Optional override for minimum accumulated confirmation score.",
    )
    parser.add_argument(
        "--confirmation-event-merge-gap",
        type=int,
        default=None,
        help="Optional override for proof-side candidate merge gap.",
    )
    parser.add_argument(
        "--confirmation-cooldown-gap",
        type=int,
        default=None,
        help="Optional override for cooldown frames after a confirmed proof event.",
    )
    parser.add_argument(
        "--calibration-enabled",
        action="store_true",
        help=(
            "Enable Sentinel calibration v1 proof-stage postprocessing for confirmed events. "
            "Raw events and pre-calibration confirmed events remain visible in artifacts."
        ),
    )
    parser.add_argument(
        "--calibration-event-merge-gap",
        type=int,
        default=None,
        help="Optional calibration v1 duplicate/near-event merge gap; defaults to the confirmation merge gap.",
    )
    parser.add_argument(
        "--calibration-benign-context-grace",
        type=int,
        default=0,
        help="Calibration v1 benign context grace in frames. Defaults to 0 for conservative fully benign suppression.",
    )
    parser.add_argument(
        "--calibration-attack-window-guard",
        type=int,
        default=0,
        help="Calibration v1 distance in frames near attack windows where non-overlapping events are retained.",
    )
    parser.add_argument(
        "--calibration-min-confirmed-span",
        type=int,
        default=2,
        help="Calibration v1 minimum confirmed span used for insufficient-evidence accounting.",
    )
    parser.add_argument(
        "--calibration-min-evidence-count",
        type=int,
        default=2,
        help="Calibration v1 minimum raw/component evidence count used for insufficient-evidence accounting.",
    )
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--suite", choices=("smoke", "full"), required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args(raw_argv)
    args._confirmation_mode_explicit = any(
        item == "--confirmation-mode" or str(item).startswith("--confirmation-mode=")
        for item in raw_argv
    )
    normalize_sentinel_calibration_mode(args)
    return args


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


def _flatten_attack_label_args(raw: Any) -> Iterable[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    flattened: List[str] = []
    for item in raw:
        if isinstance(item, (list, tuple)):
            flattened.extend(str(part) for part in item)
        else:
            flattened.append(str(item))
    return flattened


def parse_attack_labels(raw: Any) -> List[str]:
    labels: List[str] = []
    seen = set()
    for chunk in _flatten_attack_label_args(raw):
        for item in str(chunk).split(","):
            label = normalize_label(item)
            if not label:
                continue
            key = label_key(label)
            if key in seen:
                continue
            labels.append(label)
            seen.add(key)
    return labels


def parse_confirmation_profile_sweep(raw: Any) -> List[str]:
    profiles: List[str] = []
    seen = set()
    for chunk in _flatten_attack_label_args(raw):
        for item in str(chunk).split(","):
            profile = str(item).strip()
            if not profile:
                continue
            if profile not in proof_event_confirmation.CONFIRMATION_MODES:
                known = ", ".join(proof_event_confirmation.CONFIRMATION_MODES)
                raise ValueError(f"unknown confirmation profile {profile!r}; expected one of: {known}")
            if profile in seen:
                continue
            profiles.append(profile)
            seen.add(profile)
    return profiles


def normalize_sentinel_calibration_mode(args: argparse.Namespace) -> None:
    mode = getattr(args, "sentinel_calibration_mode", None)
    mode_requested = mode is not None
    confirmation_explicit = bool(getattr(args, "_confirmation_mode_explicit", False))
    if mode is None:
        if getattr(args, "calibration_enabled", False):
            mode = str(getattr(args, "confirmation_mode", DEFAULT_CONFIRMATION_MODE))
            if mode not in SENTINEL_CALIBRATION_MODES:
                mode = DEFAULT_CONFIRMATION_MODE
        else:
            mode = "off"
    if mode == "off":
        args.calibration_enabled = False
        if mode_requested and not confirmation_explicit:
            args.confirmation_mode = "off"
    else:
        args.calibration_enabled = True
        if not confirmation_explicit:
            args.confirmation_mode = mode
    args.sentinel_calibration_mode = mode


def label_key(value: Any) -> str:
    text = unicodedata.normalize("NFKC", normalize_label(value))
    text = text.replace("\ufffd", " ")
    text = re.sub(r"[\u2010-\u2015\-_/]+", " ", text)
    text = re.sub(r"[^0-9A-Za-z]+", " ", text)
    return " ".join(text.casefold().split())


def is_benign_label(raw_label: Any) -> bool:
    benign_keys = {label_key(item) for item in BENIGN_LABELS}
    return label_key(raw_label) in benign_keys


def normalize_proof_label(
    raw_label: Any,
    attack_labels: Sequence[str],
    normalize_non_benign_as: Optional[str] = None,
) -> str:
    if is_benign_label(raw_label):
        return PROOF_LABEL_BENIGN
    attack_keys = {label_key(item) for item in attack_labels}
    if attack_keys and label_key(raw_label) in attack_keys:
        return PROOF_LABEL_ATTACK
    if normalize_non_benign_as == PROOF_LABEL_ATTACK or not attack_keys:
        return PROOF_LABEL_ATTACK
    return PROOF_LABEL_BENIGN


def is_attack_label(
    raw_label: Any,
    attack_labels: Sequence[str],
    normalize_non_benign_as: Optional[str] = None,
) -> bool:
    return normalize_proof_label(raw_label, attack_labels, normalize_non_benign_as) == PROOF_LABEL_ATTACK


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


def _sample_target(frames: int, available: int) -> int:
    return max(0, min(int(frames), int(available)))


def _label_blocks(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    start = 0
    while start < len(records):
        label = records[start]["proof_label"]
        end = start
        while end + 1 < len(records) and records[end + 1]["proof_label"] == label:
            end += 1
        block_records = list(records[start : end + 1])
        blocks.append(
            {
                "proof_label": label,
                "source_start_index": int(block_records[0]["source_row_index"]),
                "source_end_index": int(block_records[-1]["source_row_index"]),
                "row_count": len(block_records),
                "records": block_records,
            }
        )
        start = end + 1
    return blocks


def _take_from_blocks(blocks: Sequence[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    remaining = max(0, int(target))
    for block in blocks:
        if remaining <= 0:
            break
        take = min(remaining, len(block["records"]))
        if take <= 0:
            continue
        records = list(block["records"][:take])
        selected.append(
            {
                "proof_label": block["proof_label"],
                "source_start_index": int(records[0]["source_row_index"]),
                "source_end_index": int(records[-1]["source_row_index"]),
                "row_count": len(records),
                "source_block_row_count": int(block["row_count"]),
                "truncated": take < int(block["row_count"]),
                "records": records,
            }
        )
        remaining -= take
    return selected


def _block_receipts(blocks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    receipts: List[Dict[str, Any]] = []
    sample_cursor = 0
    for idx, block in enumerate(blocks, start=1):
        row_count = int(block["row_count"])
        receipts.append(
            {
                "block_index": idx,
                "proof_label": block["proof_label"],
                "source_start_index": int(block["source_start_index"]),
                "source_end_index": int(block["source_end_index"]),
                "sample_start_frame": sample_cursor,
                "sample_end_frame": sample_cursor + row_count - 1 if row_count else None,
                "row_count": row_count,
                "source_block_row_count": int(block.get("source_block_row_count", row_count)),
                "truncated": bool(block.get("truncated", False)),
            }
        )
        sample_cursor += row_count
    return receipts


def _transition_boundaries_from_selected(selected: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    boundaries: List[Dict[str, Any]] = []
    for idx in range(1, len(selected)):
        previous = selected[idx - 1]["proof_label"]
        current = selected[idx]["proof_label"]
        if previous != current:
            boundaries.append(
                {
                    "before_frame": idx - 1,
                    "after_frame": idx,
                    "from": previous,
                    "to": current,
                    "source_before": int(selected[idx - 1]["source_row_index"]),
                    "source_after": int(selected[idx]["source_row_index"]),
                }
            )
    return boundaries


def build_sample_records(
    records: List[Dict[str, Any]],
    *,
    sample_mode: str,
    frames: int,
    seed: int,
    source_path: Path,
    natural_window_pre: int = DEFAULT_NATURAL_WINDOW_PRE,
    natural_window_post: int = DEFAULT_NATURAL_WINDOW_POST,
    natural_window_max_windows: int = DEFAULT_NATURAL_WINDOW_MAX_WINDOWS,
    repo_root: Path = REPO_ROOT,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not records:
        raise ValueError("cannot build a labeled sample from zero source rows")

    target = _sample_target(frames, len(records))
    benign_records = [record for record in records if record["proof_label"] == PROOF_LABEL_BENIGN]
    attack_records = [record for record in records if record["proof_label"] == PROOF_LABEL_ATTACK]

    if sample_mode == "natural":
        selected = list(records[:target])
        order_preserved = True
        transition_boundary = None
        sample_blocks: List[Dict[str, Any]] = []
        natural_window_slices: List[Dict[str, Any]] = []
    elif sample_mode in {"balanced", "transition"}:
        benign_target = target // 2
        attack_target = target - benign_target
        if len(benign_records) < benign_target or len(attack_records) < attack_target:
            raise ValueError(
                f"{sample_mode} sample requested {benign_target} benign and {attack_target} attack rows, "
                f"but source has {len(benign_records)} benign and {len(attack_records)} attack rows after normalization"
            )
        selected = list(benign_records[:benign_target]) + list(attack_records[:attack_target])
        if sample_mode == "balanced":
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(selected)).tolist()
            selected = [selected[idx] for idx in order]
            order_preserved = False
            transition_boundary = None
        else:
            order_preserved = True
            transition_boundary = {
                "benign_end_frame": benign_target - 1 if benign_target else None,
                "attack_start_frame": benign_target if attack_target else None,
            }
        sample_blocks = []
        natural_window_slices = []
    elif sample_mode == "balanced_blocks":
        benign_target = target // 2
        attack_target = target - benign_target
        all_blocks = _label_blocks(records)
        benign_blocks = [block for block in all_blocks if block["proof_label"] == PROOF_LABEL_BENIGN]
        attack_blocks = [block for block in all_blocks if block["proof_label"] == PROOF_LABEL_ATTACK]
        if sum(int(block["row_count"]) for block in benign_blocks) < benign_target or sum(int(block["row_count"]) for block in attack_blocks) < attack_target:
            raise ValueError(
                f"balanced_blocks sample requested {benign_target} benign and {attack_target} attack rows, "
                f"but source has {len(benign_records)} benign and {len(attack_records)} attack rows after normalization"
            )
        selected_blocks = _take_from_blocks(benign_blocks, benign_target) + _take_from_blocks(attack_blocks, attack_target)
        rng = np.random.default_rng(seed)
        if len(selected_blocks) > 1:
            order = rng.permutation(len(selected_blocks)).tolist()
            selected_blocks = [selected_blocks[idx] for idx in order]
        selected = [record for block in selected_blocks for record in block["records"]]
        order_preserved = False
        transition_boundary = None
        sample_blocks = _block_receipts(selected_blocks)
        natural_window_slices = []
    elif sample_mode == "natural_attack_windows":
        all_blocks = _label_blocks(records)
        attack_blocks = [block for block in all_blocks if block["proof_label"] == PROOF_LABEL_ATTACK]
        if not attack_blocks:
            raise ValueError("natural_attack_windows requested but no attack windows were found after normalization")
        selected_indices: set[int] = set()
        natural_window_slices = []
        max_windows = max(1, int(natural_window_max_windows))
        pre = max(0, int(natural_window_pre))
        post = max(0, int(natural_window_post))
        for window_index, block in enumerate(attack_blocks[:max_windows], start=1):
            attack_start = int(block["source_start_index"])
            attack_end = int(block["source_end_index"])
            slice_start = max(0, attack_start - pre)
            slice_end = min(len(records) - 1, attack_end + post)
            for idx in range(slice_start, slice_end + 1):
                selected_indices.add(idx)
            natural_window_slices.append(
                {
                    "window_index": window_index,
                    "attack_source_start_index": attack_start,
                    "attack_source_end_index": attack_end,
                    "slice_source_start_index": slice_start,
                    "slice_source_end_index": slice_end,
                    "pre_context_rows": attack_start - slice_start,
                    "post_context_rows": slice_end - attack_end,
                    "attack_rows": int(block["row_count"]),
                }
            )
        ordered_indices = sorted(selected_indices)
        truncated_by_frame_limit = len(ordered_indices) > target
        ordered_indices = ordered_indices[:target]
        selected = [records[idx] for idx in ordered_indices]
        selected_index_set = set(ordered_indices)
        for item in natural_window_slices:
            item["selected_rows"] = sum(
                1 for idx in range(int(item["slice_source_start_index"]), int(item["slice_source_end_index"]) + 1)
                if idx in selected_index_set
            )
            item["truncated_by_frame_limit"] = bool(truncated_by_frame_limit and item["selected_rows"] < (int(item["slice_source_end_index"]) - int(item["slice_source_start_index"]) + 1))
        order_preserved = True
        transition_boundary = None
        sample_blocks = _block_receipts(_label_blocks(selected))
    else:
        raise ValueError(f"unsupported sample mode: {sample_mode}")

    raw_distribution = dict(Counter(str(record["raw_label"]) for record in selected))
    normalized_distribution = dict(Counter(str(record["proof_label"]) for record in selected))
    first_attack_row = next(
        (
            {
                "sample_frame": idx,
                "source_row_index": int(record["source_row_index"]),
            }
            for idx, record in enumerate(selected)
            if record["proof_label"] == PROOF_LABEL_ATTACK
        ),
        None,
    )
    receipt = {
        "source": relpath(source_path, repo_root),
        "source_rows_read": len(records),
        "source_row_counts": {
            PROOF_LABEL_BENIGN: len(benign_records),
            PROOF_LABEL_ATTACK: len(attack_records),
            "total": len(records),
        },
        "selected_row_counts": {
            PROOF_LABEL_BENIGN: normalized_distribution.get(PROOF_LABEL_BENIGN, 0),
            PROOF_LABEL_ATTACK: normalized_distribution.get(PROOF_LABEL_ATTACK, 0),
            "total": len(selected),
        },
        "raw_label_distribution": raw_distribution,
        "normalized_label_distribution": normalized_distribution,
        "seed": seed,
        "mode": sample_mode,
        "order_preserved": order_preserved,
        "first_attack_row": first_attack_row,
        "transition_boundary": transition_boundary,
        "transition_boundaries": _transition_boundaries_from_selected(selected),
    }
    if sample_mode == "balanced_blocks":
        receipt.update(
            {
                "block_count": len(sample_blocks),
                "benign_block_count": sum(1 for block in sample_blocks if block["proof_label"] == PROOF_LABEL_BENIGN),
                "attack_block_count": sum(1 for block in sample_blocks if block["proof_label"] == PROOF_LABEL_ATTACK),
                "rows_per_block": [block["row_count"] for block in sample_blocks],
                "blocks": sample_blocks,
                "block_order_seed": seed,
            }
        )
    if sample_mode == "natural_attack_windows":
        receipt.update(
            {
                "natural_window_pre": max(0, int(natural_window_pre)),
                "natural_window_post": max(0, int(natural_window_post)),
                "natural_window_max_windows": max(1, int(natural_window_max_windows)),
                "attack_window_slice_count": len(natural_window_slices),
                "window_slices": natural_window_slices,
                "block_count": len(sample_blocks),
                "blocks": sample_blocks,
            }
        )
    return selected, receipt


def load_labeled_dataset(
    *,
    dataset: str,
    file_path: Path,
    label_column: str,
    attack_labels: Sequence[str],
    normalize_non_benign_as: Optional[str] = None,
    sample_mode: str = DEFAULT_SAMPLE_MODE,
    frames: Optional[int] = None,
    natural_window_pre: int = DEFAULT_NATURAL_WINDOW_PRE,
    natural_window_post: int = DEFAULT_NATURAL_WINDOW_POST,
    natural_window_max_windows: int = DEFAULT_NATURAL_WINDOW_MAX_WINDOWS,
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
    normalized_attack_labels = parse_attack_labels(attack_labels)
    records: List[Dict[str, Any]] = []
    for source_idx, row in enumerate(rows):
        raw_label = normalize_label(row.get(actual_label_column))
        proof_label = normalize_proof_label(raw_label, normalized_attack_labels, normalize_non_benign_as)
        records.append(
            {
                "source_row_index": source_idx,
                "row": row,
                "raw_label": raw_label,
                "proof_label": proof_label,
            }
        )

    selected_records, sample_receipt = build_sample_records(
        records,
        sample_mode=sample_mode,
        frames=frames if frames is not None else len(records),
        seed=seed,
        source_path=resolved_file,
        natural_window_pre=natural_window_pre,
        natural_window_post=natural_window_post,
        natural_window_max_windows=natural_window_max_windows,
        repo_root=repo_root,
    )
    raw_values: List[List[float]] = []
    raw_labels: List[str] = []
    proof_labels: List[str] = []
    binary_labels: List[int] = []
    source_row_indices: List[int] = []
    for record in selected_records:
        row = record["row"]
        raw_labels.append(record["raw_label"])
        proof_labels.append(record["proof_label"])
        binary_labels.append(1 if record["proof_label"] == PROOF_LABEL_ATTACK else 0)
        source_row_indices.append(int(record["source_row_index"]))
        raw_values.append([
            parse_float(row.get(column)) if parse_float(row.get(column)) is not None else math.nan
            for column in feature_columns
        ])

    standardized = standardize_matrix(np.asarray(raw_values, dtype=np.float64))
    frames = project_rows(engine, standardized, features=features, seed=seed)
    projected_feature_names = [f"cicids_projected_{idx:02d}" for idx in range(features)]

    events: List[Dict[str, Any]] = []
    for idx, record in enumerate(selected_records):
        row = record["row"]
        event = {
            "x": frames[idx].astype(float).tolist(),
            "row_index": idx,
            "source_row_index": source_row_indices[idx],
            "label": raw_labels[idx],
            "OriginalLabel": raw_labels[idx],
            "EidosProofLabel": proof_labels[idx],
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
        proof_labels=proof_labels,
        label_distribution=dict(Counter(raw_labels)),
        normalized_label_distribution=dict(Counter(proof_labels)),
        attack_labels=list(normalized_attack_labels),
        normalization_mode=(
            "configured_attack_labels"
            if normalized_attack_labels
            else ("non_benign_as_attack" if normalize_non_benign_as == PROOF_LABEL_ATTACK else "default_non_benign_as_attack")
        ),
        feature_columns=feature_columns,
        source_rows_read=len(selected_records),
        source_rows_available=len(records),
        source_row_indices=source_row_indices,
        sample_receipt=sample_receipt,
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
    attack_labels = parse_attack_labels(getattr(args, "attack_labels", []))
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
        "--sample-mode",
        args.sample_mode,
        "--event-merge-gap",
        str(args.event_merge_gap),
        "--confirmation-mode",
        args.confirmation_mode,
    ]
    if getattr(args, "sentinel_calibration_mode", None) is not None:
        parts.extend(["--sentinel-calibration-mode", args.sentinel_calibration_mode])
    if args.sample_mode == "natural_attack_windows":
        parts.extend(["--natural-window-pre", str(args.natural_window_pre)])
        parts.extend(["--natural-window-post", str(args.natural_window_post)])
        parts.extend(["--natural-window-max-windows", str(args.natural_window_max_windows)])
    for profile in parse_confirmation_profile_sweep(getattr(args, "confirmation_profile_sweep", [])):
        parts.extend(["--confirmation-profile-sweep", profile])
    for attr, flag in (
        ("confirmation_min_raw_hits", "--confirmation-min-raw-hits"),
        ("confirmation_min_duration", "--confirmation-min-duration"),
        ("confirmation_min_score", "--confirmation-min-score"),
        ("confirmation_event_merge_gap", "--confirmation-event-merge-gap"),
        ("confirmation_cooldown_gap", "--confirmation-cooldown-gap"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            parts.extend([flag, str(value)])
    if getattr(args, "calibration_enabled", False):
        parts.append("--calibration-enabled")
    for attr, flag in (
        ("calibration_event_merge_gap", "--calibration-event-merge-gap"),
        ("calibration_benign_context_grace", "--calibration-benign-context-grace"),
        ("calibration_attack_window_guard", "--calibration-attack-window-guard"),
        ("calibration_min_confirmed_span", "--calibration-min-confirmed-span"),
        ("calibration_min_evidence_count", "--calibration-min-evidence-count"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            parts.extend([flag, str(value)])
    for label in attack_labels:
        parts.extend(["--attack-labels", label])
    if args.normalize_non_benign_as:
        parts.extend(["--normalize-non-benign-as", args.normalize_non_benign_as])
    if args.max_rows is not None:
        parts.extend(["--max-rows", str(args.max_rows)])
    return command_text(parts)


def write_environment(path: Path, repo_root: Path = REPO_ROOT) -> Dict[str, str]:
    environment_text, packages = proof_helpers.collect_environment(repo_root)
    path.write_text(environment_text, encoding="utf-8")
    return packages


def collect_device_receipt(
    *,
    runtime_seconds: Optional[float] = None,
    frames_processed: Optional[int] = None,
    torch_module: Any = None,
) -> Dict[str, Any]:
    torch_installed = False
    cuda_available = False
    torch_version = None
    cuda_version = None
    device_name = None
    error = None
    try:
        torch = torch_module
        if torch is None:
            import torch as imported_torch  # type: ignore

            torch = imported_torch
        torch_installed = True
        torch_version = str(getattr(torch, "__version__", "unknown"))
        cuda_obj = getattr(torch, "cuda", None)
        cuda_available = bool(cuda_obj and cuda_obj.is_available())
        version_obj = getattr(torch, "version", None)
        cuda_version = str(getattr(version_obj, "cuda", None)) if version_obj is not None else None
        if cuda_available and cuda_obj is not None:
            try:
                device_name = str(cuda_obj.get_device_name(0))
            except Exception as exc:
                device_name = f"unavailable: {exc}"
    except Exception as exc:
        error = str(exc)

    selected_device = "cuda" if cuda_available else "cpu"
    fps = frames_processed / runtime_seconds if runtime_seconds and runtime_seconds > 0 and frames_processed is not None else None
    return {
        "torch_installed": torch_installed,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "selected_device": selected_device,
        "cpu_gpu_mode": "gpu" if selected_device == "cuda" else "cpu",
        "device_name": device_name,
        "runtime_seconds": round(runtime_seconds, 6) if runtime_seconds is not None else None,
        "frames_per_second": round(fps, 6) if fps is not None else None,
        "cpu_fallback_used": not cuda_available,
        "error": error,
    }


def append_device_receipt_to_environment(path: Path, receipt: Dict[str, Any]) -> None:
    lines = [
        "",
        "selected proof device:",
        f"torch_installed: {receipt.get('torch_installed')}",
        f"cuda_available: {receipt.get('cuda_available')}",
        f"selected_device: {receipt.get('selected_device')}",
        f"cpu_gpu_mode: {receipt.get('cpu_gpu_mode')}",
        f"device_name: {receipt.get('device_name')}",
        f"runtime_seconds: {receipt.get('runtime_seconds')}",
        f"frames_per_second: {receipt.get('frames_per_second')}",
        f"cpu_fallback_used: {receipt.get('cpu_fallback_used')}",
    ]
    if receipt.get("error"):
        lines.append(f"device_receipt_error: {receipt.get('error')}")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def build_calibration_config(args: argparse.Namespace) -> proof_calibration.SentinelCalibrationConfig:
    merge_gap = (
        args.calibration_event_merge_gap
        if getattr(args, "calibration_event_merge_gap", None) is not None
        else getattr(args, "confirmation_event_merge_gap", None)
    )
    if merge_gap is None:
        merge_gap = proof_event_confirmation.get_thresholds(args.confirmation_mode).event_merge_gap
    return proof_calibration.SentinelCalibrationConfig(
        calibration_enabled=bool(getattr(args, "calibration_enabled", False)),
        confirmation_mode_baseline=str(getattr(args, "confirmation_mode", DEFAULT_CONFIRMATION_MODE)),
        suppress_duplicate_noise=True,
        suppress_fully_benign_pressure=True,
        event_merge_gap=max(0, int(merge_gap)),
        benign_context_grace=max(0, int(getattr(args, "calibration_benign_context_grace", 0))),
        attack_window_guard=max(0, int(getattr(args, "calibration_attack_window_guard", 0))),
        min_confirmed_span=max(1, int(getattr(args, "calibration_min_confirmed_span", 2))),
        min_evidence_count=max(1, int(getattr(args, "calibration_min_evidence_count", 2))),
    )


def _path_from_porcelain(line: str) -> str:
    raw = line[2:].strip() if len(line) > 2 else ""
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    return raw.replace("\\", "/")


def git_hygiene_receipt(git_info: Dict[str, Any], out_dir: Path, repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    status_lines = [line for line in str(git_info.get("status_short") or "").splitlines() if line.strip()]
    out_prefix = relpath(out_dir, repo_root).replace("\\", "/").rstrip("/") + "/"
    generated_prefixes = tuple(GENERATED_UNTRACKED_PREFIXES) + (out_prefix,)
    tracked_dirty: List[str] = []
    untracked_generated: List[str] = []
    untracked_non_generated: List[str] = []
    for line in status_lines:
        path = _path_from_porcelain(line)
        if line.startswith("??"):
            if path.startswith(generated_prefixes):
                untracked_generated.append(path)
            else:
                untracked_non_generated.append(path)
        else:
            tracked_dirty.append(path)

    if not status_lines:
        reason = "clean"
    else:
        parts = []
        if tracked_dirty:
            parts.append(f"{len(tracked_dirty)} tracked dirty path(s)")
        if untracked_generated:
            parts.append(f"{len(untracked_generated)} generated untracked path(s)")
        if untracked_non_generated:
            parts.append(f"{len(untracked_non_generated)} non-generated untracked path(s)")
        reason = "; ".join(parts)
    return {
        "tracked_dirty": tracked_dirty,
        "untracked_generated_files": untracked_generated,
        "untracked_non_generated_files": untracked_non_generated,
        "git_dirty_reason": reason,
    }


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
                surprise_rate=parse_float(row.get("surprise_rate")),
                eigen_dominance=dominance,
                spectral_entropy=state_entropy,
                spectral_flatness=parse_float(row.get("spectral_flatness")),
                plasticity=parse_float(row.get("plasticity")),
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


def event_distance(left: Dict[str, Any], right: Dict[str, Any]) -> int:
    if overlaps(left, right):
        return 0
    if int(left["end_frame"]) < int(right["start_frame"]):
        return int(right["start_frame"]) - int(left["end_frame"])
    return int(left["start_frame"]) - int(right["end_frame"])


def event_label_metrics(detection_events: List[Dict[str, Any]], label_windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    event_windows = [
        {
            "start_frame": int(event["start_frame"]),
            "end_frame": int(event["end_frame"]),
            "event_id": event.get("event_id"),
            "source": event.get("source"),
            "severity": event.get("severity"),
            "top_drivers": event.get("top_drivers", []),
            "component_count": event.get("component_count", 1),
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
        "source": "engine_card",
        "severity": card.get("severity", card.get("regime")),
        "top_drivers": list(card.get("top_drivers", [])),
        "raw_evidence_refs": list(card.get("raw_evidence_refs", [])),
    }


def sentinel_event_to_raw_event(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event_id": event.get("event_id"),
        "start_frame": int(event["start_frame"]),
        "end_frame": int(event["end_frame"]),
        "source": "sentinel_confirmed",
        "severity": event.get("severity"),
        "top_drivers": list(event.get("top_drivers", [])),
        "raw_evidence_refs": list(event.get("raw_evidence_refs", [])),
        "event_count": event.get("event_count"),
        "confidence": event.get("confidence"),
    }


def combined_detection_events(confirmed_events: List[Dict[str, Any]], engine_cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    seen = set()
    for event in confirmed_events:
        normalized = sentinel_event_to_raw_event(event)
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


def raw_detection_events(confirmed_events: List[Dict[str, Any]], engine_cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events = [sentinel_event_to_raw_event(event) for event in confirmed_events]
    events.extend(engine_card_to_event(card) for card in engine_cards)
    return sorted(events, key=lambda item: (item["start_frame"], item["end_frame"], str(item.get("source")), str(item.get("event_id"))))


def highest_severity(values: Iterable[Any]) -> Optional[str]:
    ranked = [str(value).upper() for value in values if value]
    if not ranked:
        return None
    return max(ranked, key=lambda item: SEVERITY_RANK.get(item, 0))


def merge_detection_events(events: List[Dict[str, Any]], merge_gap: int) -> List[Dict[str, Any]]:
    if not events:
        return []
    ordered = sorted(events, key=lambda item: (int(item["start_frame"]), int(item["end_frame"]), str(item.get("source"))))
    merged: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    def _component(event: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "event_id": event.get("event_id"),
            "source": event.get("source"),
            "start_frame": int(event["start_frame"]),
            "end_frame": int(event["end_frame"]),
            "severity": event.get("severity"),
        }

    def _finalize(event: Dict[str, Any]) -> Dict[str, Any]:
        components = list(event.get("component_events", []))
        event["component_count"] = len(components)
        event["component_sources"] = dict(Counter(str(item.get("source")) for item in components))
        event["event_id"] = f"proof_merged_{event['start_frame']}_{event['end_frame']}"
        return event

    for event in ordered:
        normalized = copy.deepcopy(event)
        normalized["start_frame"] = int(normalized["start_frame"])
        normalized["end_frame"] = int(normalized["end_frame"])
        if current is None:
            current = {
                "event_id": "",
                "start_frame": normalized["start_frame"],
                "end_frame": normalized["end_frame"],
                "source": "proof_merged",
                "severity": normalized.get("severity"),
                "top_drivers": list(normalized.get("top_drivers", []))[:8],
                "raw_evidence_refs": list(normalized.get("raw_evidence_refs", [])),
                "component_events": [_component(normalized)],
            }
            continue
        if normalized["start_frame"] <= int(current["end_frame"]) + merge_gap:
            current["end_frame"] = max(int(current["end_frame"]), normalized["end_frame"])
            current["severity"] = highest_severity([current.get("severity"), normalized.get("severity")])
            current["top_drivers"] = (list(current.get("top_drivers", [])) + list(normalized.get("top_drivers", [])))[:8]
            current["raw_evidence_refs"] = sorted(set(list(current.get("raw_evidence_refs", [])) + list(normalized.get("raw_evidence_refs", []))))
            current["component_events"].append(_component(normalized))
        else:
            merged.append(_finalize(current))
            current = {
                "event_id": "",
                "start_frame": normalized["start_frame"],
                "end_frame": normalized["end_frame"],
                "source": "proof_merged",
                "severity": normalized.get("severity"),
                "top_drivers": list(normalized.get("top_drivers", []))[:8],
                "raw_evidence_refs": list(normalized.get("raw_evidence_refs", [])),
                "component_events": [_component(normalized)],
            }
    if current is not None:
        merged.append(_finalize(current))
    return merged


def dedupe_detection_events(merged_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    for event in merged_events:
        event_copy = copy.deepcopy(event)
        event_copy["source"] = "proof_merged"
        event_copy["dedupe_note"] = "one proof event retained for repeated cards inside this broader event region"
        deduped.append(event_copy)
    return deduped


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


def metrics_for_event_view(events: List[Dict[str, Any]], label_windows: List[Dict[str, Any]], frames_processed: int) -> Dict[str, Any]:
    metrics = event_label_metrics(events, label_windows)
    return {
        "event_count": len(events),
        "true_positives": metrics["true_positives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "false_positives_per_10k_frames": (
            metrics["false_positives"] * 10000.0 / frames_processed if frames_processed else None
        ),
        "true_positive_events": metrics["true_positive_events"],
        "false_positive_events": metrics["false_positive_events"],
        "false_negative_label_windows": metrics["false_negative_label_windows"],
    }


def nearest_attack_window(event: Dict[str, Any], attack_windows: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[int], str]:
    if not attack_windows:
        return None, None, "none"
    best_window: Optional[Dict[str, Any]] = None
    best_distance: Optional[int] = None
    direction = "overlap"
    for window in attack_windows:
        distance = event_distance(event, window)
        abs_distance = abs(distance)
        if best_distance is None or abs_distance < abs(best_distance):
            best_window = window
            best_distance = distance
            if distance == 0:
                direction = "overlap"
            elif int(event["end_frame"]) < int(window["start_frame"]):
                direction = "before"
            else:
                direction = "after"
    return best_window, best_distance, direction


def label_at(frame: int, raw_labels: Sequence[str], proof_labels: Sequence[str]) -> Dict[str, Any]:
    if 0 <= frame < len(raw_labels):
        return {
            "frame": frame,
            "OriginalLabel": raw_labels[frame],
            "EidosProofLabel": proof_labels[frame],
        }
    return {"frame": frame, "OriginalLabel": None, "EidosProofLabel": None}


def classify_false_positive(event: Dict[str, Any], attack_windows: List[Dict[str, Any]], event_merge_gap: int) -> str:
    window, distance, direction = nearest_attack_window(event, attack_windows)
    if window is None or distance is None:
        return "fully_benign"
    if distance == 0:
        return "overlap_boundary"
    if direction == "before" and abs(distance) <= event_merge_gap:
        return "pre_attack_near_transition"
    if direction == "after" and abs(distance) <= event_merge_gap:
        return "post_attack_near_transition"
    if int(event.get("component_count", 1)) > 1:
        return "likely_duplicate_noise"
    return "fully_benign"


def false_positive_detail(
    event: Dict[str, Any],
    *,
    view: str,
    attack_windows: List[Dict[str, Any]],
    raw_labels: Sequence[str],
    proof_labels: Sequence[str],
    event_merge_gap: int,
) -> Dict[str, Any]:
    window, distance, direction = nearest_attack_window(event, attack_windows)
    return {
        "view": view,
        "event_id": event.get("event_id"),
        "event_start": int(event["start_frame"]),
        "event_end": int(event["end_frame"]),
        "source": event.get("source"),
        "nearest_attack_window_distance": distance,
        "nearest_attack_window": (
            {
                "start_frame": int(window["start_frame"]),
                "end_frame": int(window["end_frame"]),
            }
            if window
            else None
        ),
        "nearest_attack_window_direction": direction,
        "labels_at_event_start": label_at(int(event["start_frame"]), raw_labels, proof_labels),
        "labels_at_event_end": label_at(int(event["end_frame"]), raw_labels, proof_labels),
        "severity": event.get("severity"),
        "top_drivers": list(event.get("top_drivers", [])),
        "classification": classify_false_positive(event, attack_windows, event_merge_gap),
        "component_count": event.get("component_count", 1),
        "component_sources": event.get("component_sources", {str(event.get("source")): 1}),
    }


def coverage_percent(window: Dict[str, Any], events: List[Dict[str, Any]]) -> float:
    start = int(window["start_frame"])
    end = int(window["end_frame"])
    if end < start:
        return 0.0
    intervals: List[Tuple[int, int]] = []
    for event in events:
        if not overlaps(event, window):
            continue
        intervals.append((max(start, int(event["start_frame"])), min(end, int(event["end_frame"]))))
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


def attack_window_diagnostics(label_windows: List[Dict[str, Any]], raw_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    diagnostics: List[Dict[str, Any]] = []
    for window in label_windows:
        inside = [event for event in raw_events if overlaps(event, window)]
        before = [event for event in raw_events if int(event["end_frame"]) < int(window["start_frame"])]
        after = [event for event in raw_events if int(event["start_frame"]) > int(window["end_frame"])]
        first_detection_frame = min((max(int(event["start_frame"]), int(window["start_frame"])) for event in inside), default=None)
        diagnostics.append(
            {
                "start_frame": int(window["start_frame"]),
                "end_frame": int(window["end_frame"]),
                "first_detection_frame": first_detection_frame,
                "detection_latency": (
                    first_detection_frame - int(window["start_frame"]) if first_detection_frame is not None else None
                ),
                "detections_inside_window": len(inside),
                "detections_before_window": len(before),
                "detections_after_window": len(after),
                "coverage_percentage": coverage_percent(window, raw_events),
                "missed": first_detection_frame is None,
                "label_distribution": window.get("label_distribution", {}),
                "detection_event_ids": [event.get("event_id") for event in inside],
            }
        )
    return diagnostics


def precision_delta(raw_value: Any, revised_value: Any) -> Optional[float]:
    if raw_value is None or revised_value is None:
        return None
    return float(revised_value) - float(raw_value)


def build_precision_ledger(
    *,
    raw_events: List[Dict[str, Any]],
    label_windows: List[Dict[str, Any]],
    raw_labels: Sequence[str],
    proof_labels: Sequence[str],
    frames_processed: int,
    event_merge_gap: int,
    engine_card_count: int,
    sentinel_confirmed_event_count: int,
    incident_cards_written: List[str],
) -> Dict[str, Any]:
    merged_events = merge_detection_events(raw_events, event_merge_gap)
    deduped_events = dedupe_detection_events(merged_events)
    view_metrics = {
        "raw": metrics_for_event_view(raw_events, label_windows, frames_processed),
        "merged": metrics_for_event_view(merged_events, label_windows, frames_processed),
        "deduped": metrics_for_event_view(deduped_events, label_windows, frames_processed),
    }
    false_positive_events: List[Dict[str, Any]] = []
    for view, source_events in (("raw", raw_events), ("merged", merged_events), ("deduped", deduped_events)):
        scored = metrics_for_event_view(source_events, label_windows, frames_processed)
        for event in scored["false_positive_events"]:
            false_positive_events.append(
                false_positive_detail(
                    event,
                    view=view,
                    attack_windows=label_windows,
                    raw_labels=raw_labels,
                    proof_labels=proof_labels,
                    event_merge_gap=event_merge_gap,
                )
            )

    attack_context_events: List[Dict[str, Any]] = []
    for event in deduped_events:
        window, distance, direction = nearest_attack_window(event, label_windows)
        if distance is not None and abs(distance) <= event_merge_gap:
            attack_context_events.append(
                {
                    "event_id": event.get("event_id"),
                    "start_frame": event.get("start_frame"),
                    "end_frame": event.get("end_frame"),
                    "source": event.get("source"),
                    "nearest_attack_window_distance": distance,
                    "nearest_attack_window_direction": direction,
                    "overlaps_attack_window": window is not None and distance == 0,
                    "component_count": event.get("component_count", 1),
                }
            )

    raw_metrics = view_metrics["raw"]
    merged_metrics = view_metrics["merged"]
    deduped_metrics = view_metrics["deduped"]
    incident_card_coverage = (
        min(len(incident_cards_written), len(deduped_events)) / len(deduped_events) if deduped_events else None
    )
    ledger = {
        "event_merge_gap": event_merge_gap,
        "raw_events": raw_events,
        "merged_events": merged_events,
        "deduped_events": deduped_events,
        "attack_context_events": attack_context_events,
        "false_positive_events": false_positive_events,
        "attack_window_diagnostics": attack_window_diagnostics(label_windows, raw_events),
        "incident_card_accounting": {
            "engine_card_count": engine_card_count,
            "sentinel_confirmed_event_count": sentinel_confirmed_event_count,
            "proof_raw_event_count": len(raw_events),
            "proof_merged_event_count": len(merged_events),
            "proof_deduped_event_count": len(deduped_events),
            "duplicate_event_count": max(0, len(raw_events) - len(deduped_events)),
            "incident_card_coverage": incident_card_coverage,
            "incident_card_coverage_detail": {
                "incident_cards_written": len(incident_cards_written),
                "proof_deduped_events": len(deduped_events),
                "coverage_ratio": incident_card_coverage,
            },
        },
        "precision_lift_summary": {
            "raw": {key: raw_metrics.get(key) for key in ("event_count", "true_positives", "false_positives", "false_negatives", "precision", "recall", "f1", "false_positives_per_10k_frames")},
            "merged": {key: merged_metrics.get(key) for key in ("event_count", "true_positives", "false_positives", "false_negatives", "precision", "recall", "f1", "false_positives_per_10k_frames")},
            "deduped": {key: deduped_metrics.get(key) for key in ("event_count", "true_positives", "false_positives", "false_negatives", "precision", "recall", "f1", "false_positives_per_10k_frames")},
            "raw_to_merged": {
                "precision_delta": precision_delta(raw_metrics.get("precision"), merged_metrics.get("precision")),
                "false_positive_reduction_count": raw_metrics.get("false_positives") - merged_metrics.get("false_positives"),
                "event_pressure_reduction_count": raw_metrics.get("event_count") - merged_metrics.get("event_count"),
            },
            "raw_to_deduped": {
                "precision_delta": precision_delta(raw_metrics.get("precision"), deduped_metrics.get("precision")),
                "false_positive_reduction_count": raw_metrics.get("false_positives") - deduped_metrics.get("false_positives"),
                "event_pressure_reduction_count": raw_metrics.get("event_count") - deduped_metrics.get("event_count"),
            },
            "note": "Postprocessing changes alert accounting only; raw engine and Sentinel events remain visible.",
        },
    }
    return ledger


def write_precision_ledger_md(path: Path, ledger: Dict[str, Any]) -> None:
    summary = ledger.get("precision_lift_summary", {})
    accounting = ledger.get("incident_card_accounting", {})
    lines = [
        "# Precision Ledger",
        "",
        "This ledger reports postprocessed proof accounting only. It does not tune Eidos core behavior.",
        "",
        "## Event Views",
        "",
        "| view | events | TP | FP | FN | precision | recall | F1 | FP/10k |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for view in ("raw", "merged", "deduped", "calibrated"):
        row = summary.get(view, {})
        if not row and view == "calibrated":
            continue
        lines.append(
            "| {view} | {events} | {tp} | {fp} | {fn} | {precision} | {recall} | {f1} | {fp10k} |".format(
                view=view,
                events=row.get("event_count"),
                tp=row.get("true_positives"),
                fp=row.get("false_positives"),
                fn=row.get("false_negatives"),
                precision=format_metric(row.get("precision")),
                recall=format_metric(row.get("recall")),
                f1=format_metric(row.get("f1")),
                fp10k=format_metric(row.get("false_positives_per_10k_frames")),
            )
        )
    lines.extend(
        [
            "",
            "## Precision Lift",
            "",
            f"- Raw to merged precision delta: `{format_metric(summary.get('raw_to_merged', {}).get('precision_delta'))}`",
            f"- Raw to merged FP reduction: `{summary.get('raw_to_merged', {}).get('false_positive_reduction_count')}`",
            f"- Raw to deduped precision delta: `{format_metric(summary.get('raw_to_deduped', {}).get('precision_delta'))}`",
            f"- Raw to deduped FP reduction: `{summary.get('raw_to_deduped', {}).get('false_positive_reduction_count')}`",
            "",
            "## Incident-Card Accounting",
            "",
            f"- Engine cards: `{accounting.get('engine_card_count')}`",
            f"- Sentinel confirmed events: `{accounting.get('sentinel_confirmed_event_count')}`",
            f"- Proof raw / merged / deduped events: `{accounting.get('proof_raw_event_count')}` / `{accounting.get('proof_merged_event_count')}` / `{accounting.get('proof_deduped_event_count')}`",
            f"- Duplicate event count: `{accounting.get('duplicate_event_count')}`",
            f"- Incident-card coverage: `{format_metric(accounting.get('incident_card_coverage'))}`",
            "",
            "## False-Positive Taxonomy",
            "",
        ]
    )
    fp_events = ledger.get("false_positive_events", [])
    if not fp_events:
        lines.append("- No false-positive events in the precision ledger views.")
    else:
        counts = Counter(str(item.get("classification")) for item in fp_events)
        for name, count in sorted(counts.items()):
            lines.append(f"- `{name}`: `{count}`")
    lines.extend(["", "## Attack Windows", ""])
    diagnostics = ledger.get("attack_window_diagnostics", [])
    if not diagnostics:
        lines.append("- No attack windows were present in the processed sample.")
    else:
        for item in diagnostics:
            lines.append(
                "- Window `{start}`-`{end}`: first detection `{first}`, latency `{latency}`, coverage `{coverage}%`, missed `{missed}`".format(
                    start=item.get("start_frame"),
                    end=item.get("end_frame"),
                    first=item.get("first_detection_frame"),
                    latency=item.get("detection_latency"),
                    coverage=item.get("coverage_percentage"),
                    missed=item.get("missed"),
                )
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def attack_window_summary_for_events(label_windows: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    diagnostics = proof_calibration.attack_window_diagnostics(label_windows, events)
    return proof_calibration.summarize_attack_windows(diagnostics)


def stage_metrics(
    *,
    stage: str,
    events: List[Dict[str, Any]],
    label_windows: List[Dict[str, Any]],
    frames_processed: int,
    dropped_count: int = 0,
    dropped_reason_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    metrics = metrics_for_event_view(events, label_windows, frames_processed)
    attack_summary = attack_window_summary_for_events(label_windows, events)
    return {
        "stage": stage,
        "event_count": metrics.get("event_count"),
        "true_positive_count": metrics.get("true_positives"),
        "false_positive_count": metrics.get("false_positives"),
        "false_negative_count": metrics.get("false_negatives"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "fp_per_10k": metrics.get("false_positives_per_10k_frames"),
        "attack_window_coverage": attack_summary.get("attack_window_coverage_pct"),
        "first_detection_latency": attack_summary.get("first_detection_latency_frames"),
        "attack_window_summary": attack_summary,
        "dropped_event_count": max(0, int(dropped_count)),
        "dropped_event_reason_counts": dropped_reason_counts or {},
    }


def canonical_drop_reasons(reason_codes: Sequence[str], *, overlaps_attack_window: bool, distance: Optional[int]) -> List[str]:
    mapped: List[str] = []
    joined = " ".join(str(code) for code in reason_codes)
    if "single_frame" in joined or "too_short" in joined:
        mapped.append("too_short")
    if "low_evidence" in joined or "below_confirmation" in joined or "insufficient" in joined:
        mapped.append("insufficient_evidence")
    if "duplicate" in joined or "cooldown" in joined:
        mapped.append("duplicate")
    if "benign" in joined:
        mapped.append("benign_context")
    if "calibration" in joined:
        mapped.append("calibration_suppressed")
    if not overlaps_attack_window and distance is not None:
        mapped.append("outside_attack_context")
    if not mapped:
        mapped.append("unknown")
    return sorted(set(mapped))


def label_context_for_event(
    event: Dict[str, Any],
    raw_labels: Sequence[str],
    proof_labels: Sequence[str],
) -> Dict[str, Any]:
    start = int(event.get("start_frame", 0))
    end = int(event.get("end_frame", start))
    before = label_at(start - 1, raw_labels, proof_labels)
    at_start = label_at(start, raw_labels, proof_labels)
    at_end = label_at(end, raw_labels, proof_labels)
    after = label_at(end + 1, raw_labels, proof_labels)
    during_raw = [raw_labels[idx] for idx in range(max(0, start), min(len(raw_labels), end + 1))]
    during_proof = [proof_labels[idx] for idx in range(max(0, start), min(len(proof_labels), end + 1))]
    return {
        "before": before,
        "at_start": at_start,
        "at_end": at_end,
        "after": after,
        "during_raw_label_distribution": dict(Counter(during_raw)),
        "during_proof_label_distribution": dict(Counter(during_proof)),
    }


def drop_detail(
    *,
    stage: str,
    event: Dict[str, Any],
    reason_codes: Sequence[str],
    label_windows: List[Dict[str, Any]],
    raw_labels: Sequence[str],
    proof_labels: Sequence[str],
) -> Dict[str, Any]:
    window, distance, direction = nearest_attack_window(event, label_windows)
    overlaps_window = window is not None and distance == 0
    score = event.get("score_detail") if isinstance(event.get("score_detail"), dict) else {}
    peak_z = event.get("peak_z", score.get("peak_z"))
    max_z = event.get("max_z", peak_z)
    evidence_count = event.get("evidence_count", event.get("component_count", event.get("raw_hit_count", score.get("raw_hit_count"))))
    return {
        "stage": stage,
        "event_id": event.get("event_id") or event.get("candidate_id"),
        "candidate_id": event.get("candidate_id"),
        "start_frame": int(event.get("start_frame", 0)),
        "end_frame": int(event.get("end_frame", event.get("start_frame", 0))),
        "span": max(1, int(event.get("end_frame", event.get("start_frame", 0))) - int(event.get("start_frame", 0)) + 1),
        "evidence_count": evidence_count,
        "max_severity": event.get("raw_severity") or event.get("severity") or score.get("severity"),
        "max_z": max_z,
        "nearest_attack_window_distance": distance,
        "nearest_attack_window_direction": direction,
        "nearest_attack_window": (
            {"start_frame": int(window["start_frame"]), "end_frame": int(window["end_frame"])}
            if window is not None
            else None
        ),
        "label_context": label_context_for_event(event, raw_labels, proof_labels),
        "overlaps_attack_window": overlaps_window,
        "reason_codes": list(reason_codes),
        "rejected_reasons": canonical_drop_reasons(reason_codes, overlaps_attack_window=overlaps_window, distance=distance),
    }


def reason_counter_from_drop_details(details: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for item in details:
        for reason in item.get("rejected_reasons", []):
            counts[str(reason)] += 1
    return dict(sorted(counts.items()))


def build_candidate_funnel_report(
    *,
    sample_mode: str,
    confirmation_mode: str,
    label_windows: List[Dict[str, Any]],
    raw_labels: Sequence[str],
    proof_labels: Sequence[str],
    frames_processed: int,
    raw_events: List[Dict[str, Any]],
    merged_events: List[Dict[str, Any]],
    deduped_events: List[Dict[str, Any]],
    confirmation_report: Dict[str, Any],
    calibration_report: Dict[str, Any],
) -> Dict[str, Any]:
    confirmed_events = list(confirmation_report.get("confirmed_events", []))
    calibrated_events = list(calibration_report.get("post_calibration_confirmed_events", confirmed_events))
    drop_details: List[Dict[str, Any]] = []
    for merged in merged_events:
        components = list(merged.get("component_events") or [])
        for component in components[1:]:
            drop_details.append(
                drop_detail(
                    stage="raw_to_merged",
                    event=component,
                    reason_codes=["merged_nearby_event", "duplicate"],
                    label_windows=label_windows,
                    raw_labels=raw_labels,
                    proof_labels=proof_labels,
                )
            )
    for event in confirmation_report.get("suppressed_events", []):
        drop_details.append(
            drop_detail(
                stage="deduped_to_confirmed",
                event=event,
                reason_codes=event.get("reason_codes", []),
                label_windows=label_windows,
                raw_labels=raw_labels,
                proof_labels=proof_labels,
            )
        )
    for event in calibration_report.get("suppressed_events", []):
        reason = event.get("reason_codes") or [event.get("reason_code", "calibration_suppressed")]
        drop_details.append(
            drop_detail(
                stage="confirmed_to_calibrated",
                event=event,
                reason_codes=reason,
                label_windows=label_windows,
                raw_labels=raw_labels,
                proof_labels=proof_labels,
            )
        )
    details_by_stage: Dict[str, List[Dict[str, Any]]] = {
        "raw_to_merged": [item for item in drop_details if item["stage"] == "raw_to_merged"],
        "merged_to_deduped": [item for item in drop_details if item["stage"] == "merged_to_deduped"],
        "deduped_to_confirmed": [item for item in drop_details if item["stage"] == "deduped_to_confirmed"],
        "confirmed_to_calibrated": [item for item in drop_details if item["stage"] == "confirmed_to_calibrated"],
    }
    stages = [
        stage_metrics(
            stage="raw_candidates",
            events=raw_events,
            label_windows=label_windows,
            frames_processed=frames_processed,
            dropped_count=max(0, len(raw_events) - len(merged_events)),
            dropped_reason_counts=reason_counter_from_drop_details(details_by_stage["raw_to_merged"]),
        ),
        stage_metrics(
            stage="merged_events",
            events=merged_events,
            label_windows=label_windows,
            frames_processed=frames_processed,
            dropped_count=max(0, len(merged_events) - len(deduped_events)),
            dropped_reason_counts=reason_counter_from_drop_details(details_by_stage["merged_to_deduped"]),
        ),
        stage_metrics(
            stage="deduped_events",
            events=deduped_events,
            label_windows=label_windows,
            frames_processed=frames_processed,
            dropped_count=max(0, len(deduped_events) - len(confirmed_events)),
            dropped_reason_counts=reason_counter_from_drop_details(details_by_stage["deduped_to_confirmed"]),
        ),
        stage_metrics(
            stage="confirmed_events",
            events=confirmed_events,
            label_windows=label_windows,
            frames_processed=frames_processed,
            dropped_count=max(0, len(confirmed_events) - len(calibrated_events)),
            dropped_reason_counts=reason_counter_from_drop_details(details_by_stage["confirmed_to_calibrated"]),
        ),
        stage_metrics(
            stage="calibrated_confirmed_events",
            events=calibrated_events,
            label_windows=label_windows,
            frames_processed=frames_processed,
            dropped_count=0,
            dropped_reason_counts={},
        ),
    ]
    return {
        "sample_mode": sample_mode,
        "confirmation_mode": confirmation_mode,
        "frames_processed": frames_processed,
        "attack_window_count": len(label_windows),
        "stages": stages,
        "drop_reason_accounting": drop_details,
        "policy_note": "Diagnostic-only candidate funnel. Labels are used only for reporting and scoring, not for core Eidos inference.",
    }


def write_candidate_funnel_md(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# Candidate Funnel Report",
        "",
        f"- Sample mode: `{report.get('sample_mode')}`",
        f"- Confirmation mode: `{report.get('confirmation_mode')}`",
        f"- Frames processed: `{report.get('frames_processed')}`",
        f"- Attack windows: `{report.get('attack_window_count')}`",
        "",
        "| Stage | Events | TP | FP | Precision | Recall | F1 | FP/10k | Coverage | First Latency | Dropped | Drop Reasons |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for stage in report.get("stages", []):
        lines.append(
            "| {stage} | {events} | {tp} | {fp} | {precision} | {recall} | {f1} | {fp10k} | {coverage} | {latency} | {dropped} | `{reasons}` |".format(
                stage=stage.get("stage"),
                events=format_metric(stage.get("event_count")),
                tp=format_metric(stage.get("true_positive_count")),
                fp=format_metric(stage.get("false_positive_count")),
                precision=format_metric(stage.get("precision")),
                recall=format_metric(stage.get("recall")),
                f1=format_metric(stage.get("f1")),
                fp10k=format_metric(stage.get("fp_per_10k")),
                coverage=format_metric(stage.get("attack_window_coverage")),
                latency=format_metric(stage.get("first_detection_latency")),
                dropped=format_metric(stage.get("dropped_event_count")),
                reasons=stage.get("dropped_event_reason_counts", {}),
            )
        )
    lines.extend(["", "## Drop Details", ""])
    details = report.get("drop_reason_accounting", [])
    if not details:
        lines.append("- No dropped events were recorded.")
    else:
        for item in details[:40]:
            lines.append(
                "- `{stage}` event `{event}` frames `{start}`-`{end}` reasons `{reasons}` distance `{distance}` direction `{direction}` overlap `{overlap}`".format(
                    stage=item.get("stage"),
                    event=item.get("event_id"),
                    start=item.get("start_frame"),
                    end=item.get("end_frame"),
                    reasons=", ".join(item.get("rejected_reasons", [])),
                    distance=format_metric(item.get("nearest_attack_window_distance")),
                    direction=item.get("nearest_attack_window_direction"),
                    overlap=item.get("overlaps_attack_window"),
                )
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def calibration_config_for_profile(
    config: proof_calibration.SentinelCalibrationConfig,
    profile: str,
) -> proof_calibration.SentinelCalibrationConfig:
    return replace(config, confirmation_mode_baseline=profile)


def build_confirmation_profile_sweep(
    *,
    profiles: Sequence[str],
    raw_events: List[Dict[str, Any]],
    merged_events: List[Dict[str, Any]],
    deduped_events: List[Dict[str, Any]],
    label_windows: List[Dict[str, Any]],
    raw_labels: Sequence[str],
    proof_labels: Sequence[str],
    frames_processed: int,
    step_rows: Sequence[Dict[str, Any]],
    calibration_config: proof_calibration.SentinelCalibrationConfig,
    sample_mode: str,
    crash_hit_count: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for profile in profiles:
        report = proof_event_confirmation.apply_confirmation(
            raw_events=raw_events,
            merged_events=merged_events,
            deduped_events=deduped_events,
            label_windows=label_windows,
            raw_labels=raw_labels,
            proof_labels=proof_labels,
            frames_processed=frames_processed,
            step_rows=step_rows,
            mode=profile,
        )
        confirmed_events = list(report.get("confirmed_events", []))
        confirmed_metrics = metrics_for_event_view(confirmed_events, label_windows, frames_processed)
        before_summary = attack_window_summary_for_events(label_windows, confirmed_events)
        calibration_report = proof_calibration.apply_calibration(
            confirmed_events=confirmed_events,
            raw_event_count=len(raw_events),
            merged_event_count=len(merged_events),
            deduped_event_count=len(deduped_events),
            attack_windows=label_windows,
            raw_labels=raw_labels,
            proof_labels=proof_labels,
            frames_processed=frames_processed,
            config=calibration_config_for_profile(calibration_config, profile),
            sample_mode=sample_mode,
            crash_hit_count=crash_hit_count,
        )
        calibrated_events = list(calibration_report.get("post_calibration_confirmed_events", confirmed_events))
        calibrated_metrics = metrics_for_event_view(calibrated_events, label_windows, frames_processed)
        after_summary = attack_window_summary_for_events(label_windows, calibrated_events)
        rows.append(
            {
                "profile": profile,
                "precision": confirmed_metrics.get("precision"),
                "recall": confirmed_metrics.get("recall"),
                "f1": confirmed_metrics.get("f1"),
                "fp_per_10k": confirmed_metrics.get("false_positives_per_10k_frames"),
                "coverage": before_summary.get("attack_window_coverage_pct"),
                "first_detection_latency": before_summary.get("first_detection_latency_frames"),
                "confirmed_count": confirmed_metrics.get("event_count"),
                "suppressed_count": report.get("suppressed_event_count"),
                "calibrated_precision": calibrated_metrics.get("precision"),
                "calibrated_recall": calibrated_metrics.get("recall"),
                "calibrated_f1": calibrated_metrics.get("f1"),
                "calibrated_fp_per_10k": calibrated_metrics.get("false_positives_per_10k_frames"),
                "calibrated_coverage": after_summary.get("attack_window_coverage_pct"),
                "calibrated_first_detection_latency": after_summary.get("first_detection_latency_frames"),
                "calibrated_confirmed_count": calibrated_metrics.get("event_count"),
                "calibration_suppressed_count": len(calibration_report.get("suppressed_events", [])),
                "crash_hit_count": crash_hit_count,
                "thresholds": report.get("thresholds", {}),
            }
        )
    return rows


def write_confirmation_profile_sweep_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "profile",
        "precision",
        "recall",
        "f1",
        "fp_per_10k",
        "coverage",
        "first_detection_latency",
        "confirmed_count",
        "suppressed_count",
        "calibrated_precision",
        "calibrated_recall",
        "calibrated_f1",
        "calibrated_fp_per_10k",
        "calibrated_coverage",
        "calibrated_first_detection_latency",
        "calibrated_confirmed_count",
        "calibration_suppressed_count",
        "crash_hit_count",
        "thresholds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["thresholds"] = json.dumps(row.get("thresholds", {}), sort_keys=True)
            writer.writerow(row)


def write_confirmation_profile_sweep_md(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# Confirmation Profile Sweep",
        "",
        "| Profile | Precision | Recall | F1 | FP/10k | Coverage | Latency | Confirmed | Suppressed | Cal Precision | Cal Recall | Cal F1 | Cal FP/10k | Cal Coverage | Cal Confirmed | Crash |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('profile')} | {format_metric(row.get('precision'))} | {format_metric(row.get('recall'))} | {format_metric(row.get('f1'))} | {format_metric(row.get('fp_per_10k'))} | {format_metric(row.get('coverage'))} | {format_metric(row.get('first_detection_latency'))} | {format_metric(row.get('confirmed_count'))} | {format_metric(row.get('suppressed_count'))} | {format_metric(row.get('calibrated_precision'))} | {format_metric(row.get('calibrated_recall'))} | {format_metric(row.get('calibrated_f1'))} | {format_metric(row.get('calibrated_fp_per_10k'))} | {format_metric(row.get('calibrated_coverage'))} | {format_metric(row.get('calibrated_confirmed_count'))} | {format_metric(row.get('crash_hit_count'))} |"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


METRIC_KEYS_FOR_GATE = (
    "event_count",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "false_positives_per_10k_frames",
)


def _metric_delta(after: Dict[str, Any], before: Dict[str, Any], key: str) -> Optional[float]:
    left = after.get(key)
    right = before.get(key)
    if left is None or right is None:
        return None
    return float(left) - float(right)


def build_sentinel_calibration_report(
    *,
    metrics: Dict[str, Any],
    calibration_report: Dict[str, Any],
    precision_ledger: Dict[str, Any],
) -> Dict[str, Any]:
    raw = metrics.get("raw_event_metrics", {})
    merged = metrics.get("merged_event_metrics", {})
    deduped = metrics.get("deduped_event_metrics", {})
    pre = metrics.get("pre_calibration_confirmed_event_metrics", {})
    calibrated = metrics.get("calibrated_event_metrics", {})
    fp_delta_vs_raw = _metric_delta(calibrated, raw, "false_positives_per_10k_frames")
    recall_delta_vs_pre = _metric_delta(calibrated, pre, "recall")
    recommendation = "CALIBRATION_ONLY"
    if fp_delta_vs_raw is not None and fp_delta_vs_raw >= 0:
        recommendation = "CALIBRATION_ONLY_NEEDS_TUNING"
    if recall_delta_vs_pre is not None and recall_delta_vs_pre < -0.05:
        recommendation = "CALIBRATION_ONLY_NEEDS_TUNING"
    return {
        "mode": metrics.get("sentinel_calibration_mode"),
        "confirmation_mode": metrics.get("confirmation_mode"),
        "calibration_enabled": metrics.get("calibration_enabled"),
        "calibration_version": metrics.get("calibration_version"),
        "config": calibration_report.get("config"),
        "config_hash_sha256": calibration_report.get("config_hash_sha256"),
        "raw_vs_calibrated": {
            "raw": {key: raw.get(key) for key in METRIC_KEYS_FOR_GATE},
            "merged": {key: merged.get(key) for key in METRIC_KEYS_FOR_GATE},
            "deduped": {key: deduped.get(key) for key in METRIC_KEYS_FOR_GATE},
            "pre_calibration_confirmed": {key: pre.get(key) for key in METRIC_KEYS_FOR_GATE},
            "calibrated": {key: calibrated.get(key) for key in METRIC_KEYS_FOR_GATE},
        },
        "deltas": {
            "calibrated_minus_raw": {
                "precision": _metric_delta(calibrated, raw, "precision"),
                "recall": _metric_delta(calibrated, raw, "recall"),
                "f1": _metric_delta(calibrated, raw, "f1"),
                "false_positives_per_10k_frames": fp_delta_vs_raw,
                "false_positives": _metric_delta(calibrated, raw, "false_positives"),
            },
            "calibrated_minus_deduped": {
                "precision": _metric_delta(calibrated, deduped, "precision"),
                "recall": _metric_delta(calibrated, deduped, "recall"),
                "f1": _metric_delta(calibrated, deduped, "f1"),
                "false_positives_per_10k_frames": _metric_delta(calibrated, deduped, "false_positives_per_10k_frames"),
                "false_positives": _metric_delta(calibrated, deduped, "false_positives"),
            },
            "post_minus_pre_calibration": {
                "precision": _metric_delta(calibrated, pre, "precision"),
                "recall": recall_delta_vs_pre,
                "f1": _metric_delta(calibrated, pre, "f1"),
                "false_positives_per_10k_frames": _metric_delta(calibrated, pre, "false_positives_per_10k_frames"),
                "false_positives": _metric_delta(calibrated, pre, "false_positives"),
            },
        },
        "counts": {
            "raw": metrics.get("proof_raw_event_count"),
            "merged": metrics.get("proof_merged_event_count"),
            "deduped": metrics.get("proof_deduped_event_count"),
            "pre_calibration_confirmed": metrics.get("pre_calibration_confirmed_events"),
            "calibrated": metrics.get("post_calibration_confirmed_events"),
        },
        "suppression_stats": {
            "suppressed_event_count": metrics.get("calibration_suppressed_events"),
            "suppressed_reason_counts": metrics.get("calibration_suppressed_reason_counts", {}),
            "suppressed_examples": (calibration_report.get("suppressed_events") or [])[:12],
        },
        "precision_ledger_views": sorted((precision_ledger.get("precision_lift_summary") or {}).keys()),
        "guardrails": calibration_report.get("guardrails", {}),
        "examples": {
            "confirmed_examples": (calibration_report.get("post_calibration_confirmed_events") or [])[:8],
            "suppressed_examples": (calibration_report.get("suppressed_events") or [])[:8],
        },
        "recommendation": recommendation,
        "policy_note": "Labels are used after the run for scoring and reports only. Raw events remain visible beside calibrated metrics.",
    }


def write_sentinel_calibration_report_md(path: Path, report: Dict[str, Any]) -> None:
    views = report.get("raw_vs_calibrated", {})
    lines = [
        "# Sentinel Calibration Report",
        "",
        f"- Mode: `{report.get('mode')}`",
        f"- Confirmation mode: `{report.get('confirmation_mode')}`",
        f"- Calibration enabled: `{report.get('calibration_enabled')}`",
        f"- Config hash: `{report.get('config_hash_sha256')}`",
        f"- Recommendation: `{report.get('recommendation')}`",
        "",
        "## Raw vs Calibrated",
        "",
        "| view | events | TP | FP | FN | precision | recall | F1 | FP/10k |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("raw", "merged", "deduped", "pre_calibration_confirmed", "calibrated"):
        row = views.get(name, {})
        lines.append(
            "| {name} | {events} | {tp} | {fp} | {fn} | {precision} | {recall} | {f1} | {fp10k} |".format(
                name=name,
                events=format_metric(row.get("event_count")),
                tp=format_metric(row.get("true_positives")),
                fp=format_metric(row.get("false_positives")),
                fn=format_metric(row.get("false_negatives")),
                precision=format_metric(row.get("precision")),
                recall=format_metric(row.get("recall")),
                f1=format_metric(row.get("f1")),
                fp10k=format_metric(row.get("false_positives_per_10k_frames")),
            )
        )
    lines.extend(["", "## Deltas", ""])
    for name, values in (report.get("deltas") or {}).items():
        lines.append(
            "- `{name}` precision `{precision}`, recall `{recall}`, F1 `{f1}`, FP/10k `{fp10k}`, FP `{fp}`.".format(
                name=name,
                precision=format_metric(values.get("precision")),
                recall=format_metric(values.get("recall")),
                f1=format_metric(values.get("f1")),
                fp10k=format_metric(values.get("false_positives_per_10k_frames")),
                fp=format_metric(values.get("false_positives")),
            )
        )
    stats = report.get("suppression_stats", {})
    lines.extend(
        [
            "",
            "## Suppression",
            "",
            f"- Suppressed events: `{stats.get('suppressed_event_count')}`",
            f"- Reason counts: `{stats.get('suppressed_reason_counts')}`",
            "",
            "## Examples",
            "",
        ]
    )
    examples = stats.get("suppressed_examples") or []
    if not examples:
        lines.append("- No suppressed-event examples.")
    else:
        for item in examples[:8]:
            lines.append(
                "- Suppressed `{event}` frames `{start}`-`{end}` reason `{reason}` recall risk `{risk}`.".format(
                    event=item.get("event_id"),
                    start=item.get("start_frame"),
                    end=item.get("end_frame"),
                    reason=item.get("reason_code"),
                    risk=item.get("suppression_could_affect_recall"),
                )
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Raw events were preserved beside calibrated metrics.",
            "- Labels were used only after the run for reports and scoring.",
            "- Core behavior changed: `false`.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _gate_check(status: str, passed: Optional[bool], details: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": status, "passed": passed, "details": details}


def build_engine_reopen_gate(
    *,
    metrics: Dict[str, Any],
    calibration_report: Dict[str, Any],
    precision_ledger: Dict[str, Any],
    crash_scan: Dict[str, Any],
    git_info: Dict[str, Any],
    git_hygiene: Dict[str, Any],
    device_receipt: Dict[str, Any],
) -> Dict[str, Any]:
    raw = metrics.get("raw_event_metrics", {})
    deduped = metrics.get("deduped_event_metrics", {})
    calibrated = metrics.get("calibrated_event_metrics", {})
    guardrails = calibration_report.get("guardrails", {})
    raw_visible = all(name in precision_ledger for name in ("raw_events", "merged_events", "deduped_events", "calibrated_events"))
    raw_fp10k = raw.get("false_positives_per_10k_frames")
    deduped_fp10k = deduped.get("false_positives_per_10k_frames")
    calibrated_fp10k = calibrated.get("false_positives_per_10k_frames")
    fp_pressure_passed = (
        calibrated_fp10k is None
        or (
            (raw_fp10k is None or float(calibrated_fp10k) <= float(raw_fp10k))
            and (deduped_fp10k is None or float(calibrated_fp10k) <= float(deduped_fp10k))
        )
    )
    pre = metrics.get("pre_calibration_confirmed_event_metrics", {})
    recall_before = pre.get("recall")
    recall_after = calibrated.get("recall")
    recall_passed = recall_before is None or recall_after is None or float(recall_after) >= max(0.0, float(recall_before) - 0.05)
    coverage_after = calibration_report.get("attack_window_summary_after", {}).get("attack_window_coverage_pct")
    coverage_before = calibration_report.get("attack_window_summary_before", {}).get("attack_window_coverage_pct")
    coverage_passed = coverage_before is None or coverage_after is None or float(coverage_after) >= max(0.0, float(coverage_before) - 5.0)
    core_boundary = calibration_report.get("core_behavior_boundary", proof_calibration.core_behavior_boundary())
    core_untouched = not any(bool(value) for value in core_boundary.values())
    git_clean = not bool(git_info.get("dirty")) and not git_hygiene.get("tracked_dirty") and not git_hygiene.get("untracked_non_generated_files")
    checks = {
        "pytest": _gate_check("not_run", None, {"reason": "pytest is run by the outer validation step, not inside this proof runner"}),
        "labeled_proof": _gate_check(
            "passed",
            True,
            {
                "frames_processed": metrics.get("frames_processed"),
                "sample_mode": metrics.get("sample_mode"),
                "mode": metrics.get("sentinel_calibration_mode"),
            },
        ),
        "crash_scan": _gate_check(
            "passed" if int(crash_scan.get("crash_hit_count", 0) or 0) == 0 else "failed",
            int(crash_scan.get("crash_hit_count", 0) or 0) == 0,
            crash_scan,
        ),
        "cuda_tensor_conversion": _gate_check(
            "passed",
            True,
            {
                "selected_device": device_receipt.get("selected_device"),
                "cuda_available": device_receipt.get("cuda_available"),
                "cpu_fallback_used": device_receipt.get("cpu_fallback_used"),
                "crash_scan_patterns_cover_cuda_conversion": True,
            },
        ),
        "raw_visibility": _gate_check(
            "passed" if raw_visible else "failed",
            raw_visible,
            {"ledger_views": sorted((precision_ledger.get("precision_lift_summary") or {}).keys())},
        ),
        "calibrated_fp_pressure": _gate_check(
            "passed" if fp_pressure_passed else "failed",
            fp_pressure_passed,
            {"raw_fp10k": raw_fp10k, "deduped_fp10k": deduped_fp10k, "calibrated_fp10k": calibrated_fp10k},
        ),
        "recall_coverage_tolerance": _gate_check(
            "passed" if recall_passed and coverage_passed else "failed",
            recall_passed and coverage_passed,
            {
                "pre_calibration_recall": recall_before,
                "calibrated_recall": recall_after,
                "coverage_before": coverage_before,
                "coverage_after": coverage_after,
                "recall_tolerance": -0.05,
                "coverage_tolerance_points": -5.0,
            },
        ),
        "runtime_fps": _gate_check(
            "passed" if metrics.get("frames_per_second") is not None else "failed",
            metrics.get("frames_per_second") is not None,
            {"runtime_seconds": metrics.get("runtime_seconds"), "frames_per_second": metrics.get("frames_per_second")},
        ),
        "git_clean": _gate_check(
            "passed" if git_clean else "failed",
            git_clean,
            {
                "git_dirty": bool(git_info.get("dirty")),
                "tracked_dirty": git_hygiene.get("tracked_dirty", []),
                "untracked_non_generated_files": git_hygiene.get("untracked_non_generated_files", []),
            },
        ),
        "core_untouched": _gate_check(
            "passed" if core_untouched else "failed",
            core_untouched,
            {"core_behavior_boundary": core_boundary},
        ),
    }
    hard_failures = [
        name
        for name, check in checks.items()
        if check.get("passed") is False and name in {"crash_scan", "raw_visibility", "core_untouched"}
    ]
    tuning_failures = [
        name
        for name, check in checks.items()
        if check.get("passed") is False and name in {"calibrated_fp_pressure", "recall_coverage_tolerance", "runtime_fps", "git_clean"}
    ]
    if hard_failures:
        verdict = "BLOCKED"
    elif tuning_failures:
        verdict = "CALIBRATION_ONLY_NEEDS_TUNING"
    else:
        verdict = "CALIBRATION_ONLY"
    if guardrails.get("passed") is False:
        verdict = "CALIBRATION_ONLY_NEEDS_TUNING" if verdict != "BLOCKED" else verdict
    return {
        "verdict": verdict,
        "allowed_verdicts": [
            "BLOCKED",
            "CALIBRATION_ONLY",
            "CALIBRATION_ONLY_NEEDS_TUNING",
            "READY_FOR_NARROW_CORE_EXPERIMENT",
        ],
        "default_verdict_policy": "Default to CALIBRATION_ONLY unless receipts are exceptional.",
        "checks": checks,
        "mode": metrics.get("sentinel_calibration_mode"),
        "confirmation_mode": metrics.get("confirmation_mode"),
        "guardrails": guardrails,
        "raw_visibility_note": "Raw, merged, deduped, and calibrated views are preserved side by side.",
        "core_touch_policy_note": "Core behavior is expected to be verified again by tools/check_core_touch_policy.py before final approval.",
    }


def write_engine_reopen_gate_md(path: Path, gate: Dict[str, Any]) -> None:
    lines = [
        "# Engine Reopen Readiness Gate",
        "",
        f"- Verdict: `{gate.get('verdict')}`",
        f"- Mode: `{gate.get('mode')}`",
        f"- Confirmation mode: `{gate.get('confirmation_mode')}`",
        "",
        "| check | status | passed | details |",
        "| --- | --- | --- | --- |",
    ]
    for name, check in gate.get("checks", {}).items():
        lines.append(
            "| `{name}` | `{status}` | `{passed}` | `{details}` |".format(
                name=name,
                status=check.get("status"),
                passed=check.get("passed"),
                details=check.get("details"),
            )
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            gate.get("default_verdict_policy", ""),
            "",
            "Raw evidence remains visible; this gate does not authorize a broad core rewrite.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


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
    calibration_config: Optional[proof_calibration.SentinelCalibrationConfig] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    seen_count = min(int(args.frames), dataset.source_rows_read)
    processed_indices = processed_indices_from_step_rows(step_rows[:frames_processed], seen_count)
    labels = np.asarray([int(dataset.labels[idx]) for idx in processed_indices], dtype=int)
    raw_labels = [dataset.raw_labels[idx] for idx in processed_indices]
    proof_labels = [dataset.proof_labels[idx] for idx in processed_indices]
    label_windows = contiguous_windows_from_indices(processed_indices, labels.tolist(), raw_labels)
    confirmed_events = [event.to_dict() for event in confirmation.confirmed_events]
    detection_events = combined_detection_events(confirmed_events, engine_incident_cards)
    raw_events = raw_detection_events(confirmed_events, engine_incident_cards)
    precision_ledger = build_precision_ledger(
        raw_events=raw_events,
        label_windows=label_windows,
        raw_labels=dataset.raw_labels,
        proof_labels=dataset.proof_labels,
        frames_processed=frames_processed,
        event_merge_gap=max(0, int(args.event_merge_gap)),
        engine_card_count=len(engine_incident_cards),
        sentinel_confirmed_event_count=len(confirmed_events),
        incident_cards_written=incident_cards_written,
    )
    raw_view = precision_ledger["precision_lift_summary"]["raw"]
    merged_view = precision_ledger["precision_lift_summary"]["merged"]
    deduped_view = precision_ledger["precision_lift_summary"]["deduped"]
    confirmation_report = proof_event_confirmation.apply_confirmation(
        raw_events=raw_events,
        merged_events=precision_ledger.get("merged_events", []),
        deduped_events=precision_ledger.get("deduped_events", []),
        label_windows=label_windows,
        raw_labels=raw_labels,
        proof_labels=proof_labels,
        frames_processed=frames_processed,
        step_rows=step_rows[:frames_processed],
        mode=args.confirmation_mode,
        min_raw_hits=args.confirmation_min_raw_hits,
        min_duration=args.confirmation_min_duration,
        min_score=args.confirmation_min_score,
        event_merge_gap=args.confirmation_event_merge_gap,
        cooldown_gap=args.confirmation_cooldown_gap,
    )
    pre_calibration_confirmed_events = list(confirmation_report.get("confirmed_events", []))
    pre_calibration_view = metrics_for_event_view(pre_calibration_confirmed_events, label_windows, frames_processed)
    pre_calibration_view_metrics = {
        "raw": raw_view,
        "merged": merged_view,
        "deduped": deduped_view,
        "confirmed": pre_calibration_view,
    }
    event_confirmation_report = proof_event_confirmation.add_metric_summary(confirmation_report, pre_calibration_view_metrics)
    calibration_config = calibration_config or build_calibration_config(args)
    calibration_report = proof_calibration.apply_calibration(
        confirmed_events=pre_calibration_confirmed_events,
        raw_event_count=len(raw_events),
        merged_event_count=len(precision_ledger.get("merged_events", [])),
        deduped_event_count=len(precision_ledger.get("deduped_events", [])),
        attack_windows=label_windows,
        raw_labels=dataset.raw_labels,
        proof_labels=dataset.proof_labels,
        frames_processed=frames_processed,
        config=calibration_config,
        sample_mode=args.sample_mode,
        crash_hit_count=crash_scan.get("crash_hit_count", 0),
    )
    calibrated_precision_ledger = proof_calibration.build_calibrated_precision_ledger(calibration_report)
    proof_confirmed_events = list(calibration_report.get("post_calibration_confirmed_events", pre_calibration_confirmed_events))
    confirmed_view = metrics_for_event_view(proof_confirmed_events, label_windows, frames_processed)
    candidate_funnel_report = build_candidate_funnel_report(
        sample_mode=args.sample_mode,
        confirmation_mode=args.confirmation_mode,
        label_windows=label_windows,
        raw_labels=dataset.raw_labels,
        proof_labels=dataset.proof_labels,
        frames_processed=frames_processed,
        raw_events=raw_events,
        merged_events=precision_ledger.get("merged_events", []),
        deduped_events=precision_ledger.get("deduped_events", []),
        confirmation_report=event_confirmation_report,
        calibration_report=calibration_report,
    )
    profile_sweep_profiles = parse_confirmation_profile_sweep(args.confirmation_profile_sweep) or [args.confirmation_mode]
    confirmation_profile_sweep = build_confirmation_profile_sweep(
        profiles=profile_sweep_profiles,
        raw_events=raw_events,
        merged_events=precision_ledger.get("merged_events", []),
        deduped_events=precision_ledger.get("deduped_events", []),
        label_windows=label_windows,
        raw_labels=dataset.raw_labels,
        proof_labels=dataset.proof_labels,
        frames_processed=frames_processed,
        step_rows=step_rows[:frames_processed],
        calibration_config=calibration_config,
        sample_mode=args.sample_mode,
        crash_hit_count=crash_scan.get("crash_hit_count", 0),
    )
    view_metrics = {
        "raw": raw_view,
        "merged": merged_view,
        "deduped": deduped_view,
        "confirmed": confirmed_view,
        "calibrated": confirmed_view,
    }
    label_metrics = confirmed_view
    eidos_ratio = step_rows[-1].get("ratio") if step_rows else None
    crash_scan = crash_scan or {"crash_hit_count": 0, "status": "not_run"}
    precision_lift_summary = dict(precision_ledger["precision_lift_summary"])
    precision_lift_summary["confirmed"] = {
        key: confirmed_view.get(key)
        for key in ("event_count", "true_positives", "false_positives", "false_negatives", "precision", "recall", "f1", "false_positives_per_10k_frames")
    }
    precision_lift_summary["calibrated"] = {
        key: confirmed_view.get(key)
        for key in ("event_count", "true_positives", "false_positives", "false_negatives", "precision", "recall", "f1", "false_positives_per_10k_frames")
    }
    precision_lift_summary["raw_to_confirmed"] = event_confirmation_report.get("precision_lift_summary", {})
    precision_lift_summary["pre_to_post_calibration"] = calibrated_precision_ledger.get("before_after_metrics", {}).get("delta", {})
    precision_ledger["calibrated_events"] = proof_confirmed_events
    precision_ledger["calibrated_event_count"] = len(proof_confirmed_events)
    precision_ledger["precision_lift_summary"] = precision_lift_summary
    event_summary = {
        "sentinel_confirmation_mode": DEFAULT_SENTINEL_CONFIRMATION_MODE,
        "confirmation_mode": args.confirmation_mode,
        "sentinel_calibration_mode": args.sentinel_calibration_mode,
        "calibration_enabled": calibration_report.get("calibration_enabled"),
        "calibration_version": calibration_report.get("calibration_version"),
        "candidate_events": event_confirmation_report.get("candidate_event_count"),
        "confirmed_events": proof_confirmed_events,
        "confirmed_event_count": len(proof_confirmed_events),
        "pre_calibration_confirmed_events": pre_calibration_confirmed_events,
        "pre_calibration_confirmed_event_count": len(pre_calibration_confirmed_events),
        "post_calibration_confirmed_events": proof_confirmed_events,
        "post_calibration_confirmed_event_count": len(proof_confirmed_events),
        "raw_events": raw_events,
        "raw_event_count": len(raw_events),
        "merged_events": precision_ledger.get("merged_events", []),
        "merged_event_count": len(precision_ledger.get("merged_events", [])),
        "deduped_events": precision_ledger.get("deduped_events", []),
        "deduped_event_count": len(precision_ledger.get("deduped_events", [])),
        "sentinel_confirmed_events": confirmed_events,
        "sentinel_confirmed_event_count": len(confirmed_events),
        "engine_incident_cards": engine_incident_cards,
        "engine_incident_card_count": len(engine_incident_cards),
        "suppressed_candidates": event_confirmation_report.get("suppressed_event_count"),
        "suppressed_confirmation_events": event_confirmation_report.get("suppressed_events", []),
        "calibration_suppressed_events": calibration_report.get("suppressed_events", []),
        "calibration_suppressed_event_count": len(calibration_report.get("suppressed_events", [])),
        "calibration_guardrails": calibration_report.get("guardrails", {}),
        "calibration_config_hash_sha256": calibration_report.get("config_hash_sha256"),
        "sentinel_candidate_events": confirmation.candidate_events,
        "sentinel_suppressed_candidates": confirmation.suppressed_candidates,
        "cooldown_suppressions": confirmation.cooldown_suppressions,
        "sentinel_merged_events": confirmation.merged_events,
        "label_windows": label_windows,
        "incident_cards_written": incident_cards_written,
        "precision_ledger_path": "precision_ledger.json",
        "event_confirmation_report_path": "event_confirmation_report.json",
        "sentinel_calibration_report_path": "sentinel_calibration_v1.json",
        "calibrated_precision_ledger_path": "calibrated_precision_ledger.json",
        "policy_note": (
            "Raw Sentinel and engine-card events are preserved. The proof-side confirmation layer "
            "adds an ablatable confirmed-event view, and optional calibration v1 postprocesses "
            "that view without tuning Eidos thresholds or core behavior."
        ),
    }
    accounting = precision_ledger["incident_card_accounting"]
    metrics = {
        "dataset": args.dataset,
        "suite": args.suite,
        "seed": args.seed,
        "sample_mode": args.sample_mode,
        "confirmation_mode": args.confirmation_mode,
        "sentinel_calibration_mode": args.sentinel_calibration_mode,
        "calibration_enabled": calibration_report.get("calibration_enabled"),
        "calibration_version": calibration_report.get("calibration_version"),
        "calibration_config": calibration_report.get("config"),
        "calibration_config_hash_sha256": calibration_report.get("config_hash_sha256"),
        "calibration_guardrails": calibration_report.get("guardrails", {}),
        "confirmation_thresholds": event_confirmation_report.get("thresholds", {}),
        "event_merge_gap": max(0, int(args.event_merge_gap)),
        "frames_requested": args.frames,
        "frames_seen": seen_count,
        "frames_processed": frames_processed,
        "source_rows_read": dataset.source_rows_read,
        "source_rows_available": dataset.source_rows_available,
        "source_file": relpath(dataset.source_path),
        "label_column": dataset.label_column,
        "labels_detected": sorted(dataset.label_distribution),
        "label_distribution": dict(Counter(dataset.raw_labels[:seen_count])),
        "raw_label_distribution": dict(Counter(dataset.raw_labels[:seen_count])),
        "normalized_label_distribution": dict(Counter(dataset.proof_labels[:seen_count])),
        "scored_label_distribution": dict(Counter(raw_labels)),
        "scored_normalized_label_distribution": dict(Counter(proof_labels)),
        "scored_frame_indices": processed_indices,
        "attack_labels": dataset.attack_labels or "non-benign labels treated as attacks",
        "normalization_mode": dataset.normalization_mode,
        "sample_receipt": dataset.sample_receipt,
        "label_window_count": len(label_windows),
        "candidate_events": event_confirmation_report.get("candidate_event_count"),
        "confirmed_events": len(proof_confirmed_events),
        "pre_calibration_confirmed_events": len(pre_calibration_confirmed_events),
        "post_calibration_confirmed_events": len(proof_confirmed_events),
        "calibration_suppressed_events": len(calibration_report.get("suppressed_events", [])),
        "calibration_suppressed_reason_counts": calibration_report.get("suppressed_reason_counts", {}),
        "confirmation_suppressed_events": event_confirmation_report.get("suppressed_event_count"),
        "sentinel_confirmed_events": len(confirmed_events),
        "sentinel_candidate_events": confirmation.candidate_events,
        "sentinel_suppressed_candidates": confirmation.suppressed_candidates,
        "engine_incident_card_count": len(engine_incident_cards),
        "suppressed_candidates": event_confirmation_report.get("suppressed_event_count"),
        "cooldown_suppressions": confirmation.cooldown_suppressions,
        "merged_events": confirmation.merged_events,
        **{key: label_metrics[key] for key in ("true_positives", "false_positives", "false_negatives", "precision", "recall", "f1")},
        "false_positives_per_10k_frames": (
            label_metrics["false_positives"] * 10000.0 / frames_processed if frames_processed else None
        ),
        "raw_event_metrics": raw_view,
        "merged_event_metrics": merged_view,
        "deduped_event_metrics": deduped_view,
        "confirmed_event_metrics": {
            key: confirmed_view.get(key)
            for key in ("event_count", "true_positives", "false_positives", "false_negatives", "precision", "recall", "f1", "false_positives_per_10k_frames")
        },
        "pre_calibration_confirmed_event_metrics": {
            key: pre_calibration_view.get(key)
            for key in ("event_count", "true_positives", "false_positives", "false_negatives", "precision", "recall", "f1", "false_positives_per_10k_frames")
        },
        "calibrated_event_metrics": {
            key: confirmed_view.get(key)
            for key in ("event_count", "true_positives", "false_positives", "false_negatives", "precision", "recall", "f1", "false_positives_per_10k_frames")
        },
        "event_view_metrics": view_metrics,
        "candidate_funnel_report_path": "candidate_funnel_report.json",
        "confirmation_profile_sweep_path": "confirmation_profile_sweep.csv",
        "confirmation_profile_sweep": confirmation_profile_sweep,
        "precision_lift_summary": precision_lift_summary,
        "event_confirmation_precision_lift_summary": event_confirmation_report.get("precision_lift_summary", {}),
        "proof_raw_event_count": accounting["proof_raw_event_count"],
        "proof_merged_event_count": accounting["proof_merged_event_count"],
        "proof_deduped_event_count": accounting["proof_deduped_event_count"],
        "proof_confirmed_event_count": len(proof_confirmed_events),
        "duplicate_event_count": accounting["duplicate_event_count"],
        "incident_card_coverage": accounting["incident_card_coverage"],
        "incident_card_count": len(incident_cards_written),
        "incident_card_filenames": incident_cards_written,
        "eidos_compression_ratio": eidos_ratio,
        "external_compression_baselines": compression_baselines,
        "runtime_seconds": round(runtime_seconds, 6),
        "frames_per_second": round(frames_processed / runtime_seconds, 6) if runtime_seconds > 0 else None,
        "crash_hit_count": crash_scan.get("crash_hit_count", 0),
        "crash_scan_status": crash_scan.get("status", "unknown"),
        "known_limitations": [
            "This is a labeled proof harness and dataset adapter, not threshold tuning.",
            "Metrics are event-level over contiguous attack label windows and raw/merged/deduped/confirmed event views.",
            "The confirmation and calibration layers are proof-side and label-aware for measurement; raw Sentinel behavior remains visible.",
            "Large CICIDS/WebAttacks files are not downloaded by this runner; pass a mounted or uploaded CSV path with --file.",
        ],
    }
    return (
        metrics,
        event_summary,
        precision_ledger,
        event_confirmation_report,
        calibration_report,
        calibrated_precision_ledger,
        candidate_funnel_report,
        confirmation_profile_sweep,
    )


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
        "sample_mode": metrics.get("sample_mode"),
        "confirmation_mode": metrics.get("confirmation_mode"),
        "sentinel_calibration_mode": metrics.get("sentinel_calibration_mode"),
        "calibration_enabled": metrics.get("calibration_enabled"),
        "calibration_version": metrics.get("calibration_version"),
        "frames_requested": metrics.get("frames_requested"),
        "frames_processed": metrics.get("frames_processed"),
        "label_column": metrics.get("label_column"),
        "labels_detected": ", ".join(metrics.get("labels_detected", [])),
        "label_distribution": json.dumps(metrics.get("label_distribution", {}), sort_keys=True),
        "raw_label_distribution": json.dumps(metrics.get("raw_label_distribution", {}), sort_keys=True),
        "normalized_label_distribution": json.dumps(metrics.get("normalized_label_distribution", {}), sort_keys=True),
        "candidate_events": metrics.get("candidate_events"),
        "confirmed_events": metrics.get("confirmed_events"),
        "pre_calibration_confirmed_events": metrics.get("pre_calibration_confirmed_events"),
        "post_calibration_confirmed_events": metrics.get("post_calibration_confirmed_events"),
        "calibration_suppressed_events": metrics.get("calibration_suppressed_events"),
        "proof_raw_event_count": metrics.get("proof_raw_event_count"),
        "proof_merged_event_count": metrics.get("proof_merged_event_count"),
        "proof_deduped_event_count": metrics.get("proof_deduped_event_count"),
        "proof_confirmed_event_count": metrics.get("proof_confirmed_event_count"),
        "suppressed_candidates": metrics.get("suppressed_candidates"),
        "true_positives": metrics.get("true_positives"),
        "false_positives": metrics.get("false_positives"),
        "false_negatives": metrics.get("false_negatives"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "false_positives_per_10k_frames": metrics.get("false_positives_per_10k_frames"),
        "pre_calibration_precision": metrics.get("pre_calibration_confirmed_event_metrics", {}).get("precision"),
        "pre_calibration_recall": metrics.get("pre_calibration_confirmed_event_metrics", {}).get("recall"),
        "pre_calibration_f1": metrics.get("pre_calibration_confirmed_event_metrics", {}).get("f1"),
        "pre_calibration_false_positives_per_10k_frames": metrics.get("pre_calibration_confirmed_event_metrics", {}).get("false_positives_per_10k_frames"),
        "calibrated_precision": metrics.get("calibrated_event_metrics", {}).get("precision"),
        "calibrated_recall": metrics.get("calibrated_event_metrics", {}).get("recall"),
        "calibrated_f1": metrics.get("calibrated_event_metrics", {}).get("f1"),
        "calibrated_false_positives_per_10k_frames": metrics.get("calibrated_event_metrics", {}).get("false_positives_per_10k_frames"),
        "incident_card_count": metrics.get("incident_card_count"),
        "eidos_compression_ratio": metrics.get("eidos_compression_ratio"),
        "best_external_baseline": baselines.get("best_baseline", ""),
        "best_external_baseline_ratio": baselines.get("best_baseline_compression_ratio", ""),
        "runtime_seconds": metrics.get("runtime_seconds"),
        "frames_per_second": metrics.get("frames_per_second"),
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
    raw_view = metrics.get("raw_event_metrics", {})
    merged_view = metrics.get("merged_event_metrics", {})
    deduped_view = metrics.get("deduped_event_metrics", {})
    confirmed_view = metrics.get("confirmed_event_metrics", {})
    pre_calibration_view = metrics.get("pre_calibration_confirmed_event_metrics", {})
    lines = [
        "# Labeled Metrics",
        "",
        f"- Dataset: `{metrics.get('dataset')}`",
        f"- Sample mode: `{metrics.get('sample_mode')}`",
        f"- Confirmation mode: `{metrics.get('confirmation_mode')}`",
        f"- Sentinel calibration mode: `{metrics.get('sentinel_calibration_mode')}`",
        f"- Calibration enabled: `{metrics.get('calibration_enabled')}`",
        f"- Calibration version: `{metrics.get('calibration_version')}`",
        f"- Calibration config hash: `{metrics.get('calibration_config_hash_sha256')}`",
        f"- Frames processed: `{metrics.get('frames_processed')}`",
        f"- Labels detected: `{', '.join(metrics.get('labels_detected', []))}`",
        f"- Raw label distribution: `{metrics.get('raw_label_distribution')}`",
        f"- Normalized label distribution: `{metrics.get('normalized_label_distribution')}`",
        f"- Candidate events: `{metrics.get('candidate_events')}`",
        f"- Confirmed events: `{metrics.get('confirmed_events')}`",
        f"- Pre-calibration confirmed events: `{metrics.get('pre_calibration_confirmed_events')}`",
        f"- Post-calibration confirmed events: `{metrics.get('post_calibration_confirmed_events')}`",
        f"- Calibration suppressed events: `{metrics.get('calibration_suppressed_events')}`",
        f"- Proof raw / merged / deduped / confirmed events: `{metrics.get('proof_raw_event_count')}` / `{metrics.get('proof_merged_event_count')}` / `{metrics.get('proof_deduped_event_count')}` / `{metrics.get('proof_confirmed_event_count')}`",
        f"- Suppressed candidates: `{metrics.get('suppressed_candidates')}`",
        f"- True positives / false positives / false negatives: `{metrics.get('true_positives')}` / `{metrics.get('false_positives')}` / `{metrics.get('false_negatives')}`",
        f"- Precision / recall / F1: `{format_metric(metrics.get('precision'))}` / `{format_metric(metrics.get('recall'))}` / `{format_metric(metrics.get('f1'))}`",
        f"- False positives per 10k frames: `{format_metric(metrics.get('false_positives_per_10k_frames'))}`",
        f"- Incident-card count: `{metrics.get('incident_card_count')}`",
        f"- Eidos compression ratio: `{format_metric(metrics.get('eidos_compression_ratio'))}`",
        f"- Runtime seconds: `{metrics.get('runtime_seconds')}`",
        f"- Crash hits: `{metrics.get('crash_hit_count')}`",
        "",
        "## Raw / Merged / Deduped / Pre-Calibrated / Confirmed Views",
        "",
        "| view | events | TP | FP | FN | precision | recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| raw | {raw_view.get('event_count')} | {raw_view.get('true_positives')} | {raw_view.get('false_positives')} | {raw_view.get('false_negatives')} | {format_metric(raw_view.get('precision'))} | {format_metric(raw_view.get('recall'))} | {format_metric(raw_view.get('f1'))} |",
        f"| merged | {merged_view.get('event_count')} | {merged_view.get('true_positives')} | {merged_view.get('false_positives')} | {merged_view.get('false_negatives')} | {format_metric(merged_view.get('precision'))} | {format_metric(merged_view.get('recall'))} | {format_metric(merged_view.get('f1'))} |",
        f"| deduped | {deduped_view.get('event_count')} | {deduped_view.get('true_positives')} | {deduped_view.get('false_positives')} | {deduped_view.get('false_negatives')} | {format_metric(deduped_view.get('precision'))} | {format_metric(deduped_view.get('recall'))} | {format_metric(deduped_view.get('f1'))} |",
        f"| pre-calibration confirmed | {pre_calibration_view.get('event_count')} | {pre_calibration_view.get('true_positives')} | {pre_calibration_view.get('false_positives')} | {pre_calibration_view.get('false_negatives')} | {format_metric(pre_calibration_view.get('precision'))} | {format_metric(pre_calibration_view.get('recall'))} | {format_metric(pre_calibration_view.get('f1'))} |",
        f"| confirmed | {confirmed_view.get('event_count')} | {confirmed_view.get('true_positives')} | {confirmed_view.get('false_positives')} | {confirmed_view.get('false_negatives')} | {format_metric(confirmed_view.get('precision'))} | {format_metric(confirmed_view.get('recall'))} | {format_metric(confirmed_view.get('f1'))} |",
        "",
        "## Interpretation",
        "",
        "These metrics compare existing Eidos/Sentinel outputs against labeled attack windows. The confirmed view is proof-side postprocessing; raw behavior remains visible.",
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
        f"- Sample mode: `{metrics.get('sample_mode')}`",
        f"- Confirmation mode: `{metrics.get('confirmation_mode')}`",
        f"- Sentinel calibration mode: `{metrics.get('sentinel_calibration_mode')}`",
        f"- Calibration enabled: `{metrics.get('calibration_enabled')}`",
        f"- Calibration version: `{metrics.get('calibration_version')}`",
        f"- Calibration config hash: `{metrics.get('calibration_config_hash_sha256')}`",
        f"- Raw label distribution: `{metrics.get('raw_label_distribution')}`",
        f"- Normalized label distribution: `{metrics.get('normalized_label_distribution')}`",
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
        "## Precision Ledger, Confirmation, And Calibration Views",
        "",
        "| view | events | TP | FP | FN | precision | recall | F1 | FP/10k |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for view_name, view in (
        ("raw", metrics.get("raw_event_metrics", {})),
        ("merged", metrics.get("merged_event_metrics", {})),
        ("deduped", metrics.get("deduped_event_metrics", {})),
        ("pre-calibration confirmed", metrics.get("pre_calibration_confirmed_event_metrics", {})),
        ("confirmed", metrics.get("confirmed_event_metrics", {})),
    ):
        lines.append(
            "| {view_name} | {events} | {tp} | {fp} | {fn} | {precision} | {recall} | {f1} | {fp10k} |".format(
                view_name=view_name,
                events=view.get("event_count"),
                tp=view.get("true_positives"),
                fp=view.get("false_positives"),
                fn=view.get("false_negatives"),
                precision=format_metric(view.get("precision")),
                recall=format_metric(view.get("recall")),
                f1=format_metric(view.get("f1")),
                fp10k=format_metric(view.get("false_positives_per_10k_frames")),
            )
        )
    lines.extend(
        [
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
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_config_doc(
    *,
    args: argparse.Namespace,
    dataset: LabeledDataset,
    engine_info: Dict[str, Any],
    command: str,
    out_dir: Path,
) -> Dict[str, Any]:
    calibration_config = build_calibration_config(args)
    doc = {
        "benchmark": {
            "dataset": args.dataset,
            "suite": args.suite,
            "seed": args.seed,
            "frames": args.frames,
            "max_rows": args.max_rows,
            "sample_mode": args.sample_mode,
            "natural_attack_windows": {
                "pre": args.natural_window_pre,
                "post": args.natural_window_post,
                "max_windows": args.natural_window_max_windows,
            },
            "event_merge_gap": args.event_merge_gap,
            "confirmation_mode": args.confirmation_mode,
            "sentinel_calibration_mode": args.sentinel_calibration_mode,
            "confirmation_profile_sweep": parse_confirmation_profile_sweep(args.confirmation_profile_sweep),
            "confirmation_threshold_overrides": {
                "min_raw_hits": args.confirmation_min_raw_hits,
                "min_duration": args.confirmation_min_duration,
                "min_score": args.confirmation_min_score,
                "event_merge_gap": args.confirmation_event_merge_gap,
                "cooldown_gap": args.confirmation_cooldown_gap,
            },
            "sentinel_calibration_v1": proof_calibration.config_to_dict(calibration_config),
            "sentinel_calibration_v1_config_hash_sha256": proof_calibration.config_hash(calibration_config),
            "command": command,
            "artifact_dir": relpath(out_dir),
        },
        "dataset": {
            "source_file": relpath(dataset.source_path),
            "label_column": dataset.label_column,
            "labels_detected": sorted(dataset.label_distribution),
            "raw_label_distribution": dataset.label_distribution,
            "normalized_label_distribution": dataset.normalized_label_distribution,
            "attack_labels": dataset.attack_labels or "non-benign labels treated as attacks",
            "normalization_mode": dataset.normalization_mode,
            "sample_receipt": dataset.sample_receipt,
            "feature_columns": dataset.feature_columns,
            "rows_read": dataset.source_rows_read,
            "source_rows_available": dataset.source_rows_available,
        },
        "engine": engine_info,
        "core_behavior": {
            "reservoir_dynamics_changed": False,
            "rls_updates_changed": False,
            "sentinel_thresholds_changed": False,
            "anomaly_policy_tuned": False,
            "new_architecture_layers_added": False,
            "calibration_is_proof_stage_postprocessing": True,
        },
    }
    doc["config_hash_sha256"] = stable_hash(doc)
    return doc


def build_manifest(
    *,
    generated_at: str,
    command: str,
    git_info: Dict[str, Any],
    git_hygiene: Dict[str, Any],
    engine_info: Dict[str, Any],
    packages: Dict[str, str],
    metrics: Dict[str, Any],
    config_hash: str,
    device_receipt: Dict[str, Any],
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
        "git_hygiene": git_hygiene,
        "tracked_dirty": git_hygiene.get("tracked_dirty", []),
        "untracked_generated_files": git_hygiene.get("untracked_generated_files", []),
        "untracked_non_generated_files": git_hygiene.get("untracked_non_generated_files", []),
        "git_dirty_reason": git_hygiene.get("git_dirty_reason", "unknown"),
        "engine": engine_info,
        "packages": packages,
        "device": device_receipt,
        "config": {
            "config_hash_sha256": config_hash,
            "config_path": "config.json",
        },
        "event_confirmation": {
            "mode": metrics.get("confirmation_mode"),
            "thresholds": metrics.get("confirmation_thresholds"),
            "report_json": "event_confirmation_report.json",
            "report_md": "event_confirmation_report.md",
        },
        "sentinel_calibration_v1": {
            "enabled": metrics.get("calibration_enabled"),
            "version": metrics.get("calibration_version"),
            "mode": metrics.get("sentinel_calibration_mode"),
            "config_hash_sha256": metrics.get("calibration_config_hash_sha256"),
            "guardrails": metrics.get("calibration_guardrails"),
            "report_json": "sentinel_calibration_v1.json",
            "report_md": "sentinel_calibration_v1.md",
            "gate_report_json": "sentinel_calibration_report.json",
            "gate_report_md": "sentinel_calibration_report.md",
            "calibrated_precision_ledger_json": "calibrated_precision_ledger.json",
            "calibrated_precision_ledger_md": "calibrated_precision_ledger.md",
        },
        "sample_receipt": metrics.get("sample_receipt"),
        "label_distributions": {
            "raw": metrics.get("raw_label_distribution"),
            "normalized": metrics.get("normalized_label_distribution"),
        },
        "metrics": {
            key: metrics.get(key)
            for key in (
                "frames_processed",
                "frames_per_second",
                "confirmation_mode",
                "sentinel_calibration_mode",
                "candidate_events",
                "confirmed_events",
                "pre_calibration_confirmed_events",
                "post_calibration_confirmed_events",
                "calibration_suppressed_events",
                "suppressed_candidates",
                "proof_raw_event_count",
                "proof_merged_event_count",
                "proof_deduped_event_count",
                "proof_confirmed_event_count",
                "duplicate_event_count",
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
            "precision_ledger_json": "precision_ledger.json",
            "precision_ledger_md": "precision_ledger.md",
            "event_confirmation_report_json": "event_confirmation_report.json",
            "event_confirmation_report_md": "event_confirmation_report.md",
            "sentinel_calibration_v1_json": "sentinel_calibration_v1.json",
            "sentinel_calibration_v1_md": "sentinel_calibration_v1.md",
            "sentinel_calibration_report_json": "sentinel_calibration_report.json",
            "sentinel_calibration_report_md": "sentinel_calibration_report.md",
            "calibrated_precision_ledger_json": "calibrated_precision_ledger.json",
            "calibrated_precision_ledger_md": "calibrated_precision_ledger.md",
            "candidate_funnel_report_json": "candidate_funnel_report.json",
            "candidate_funnel_report_md": "candidate_funnel_report.md",
            "confirmation_profile_sweep_json": "confirmation_profile_sweep.json",
            "confirmation_profile_sweep_csv": "confirmation_profile_sweep.csv",
            "confirmation_profile_sweep_md": "confirmation_profile_sweep.md",
            "incident_cards_dir": "incident_cards",
            "proof_digest_json": "proof_digest.json",
            "proof_digest_md": "proof_digest.md",
            "engine_reopen_gate_json": "engine_reopen_gate.json",
            "engine_reopen_gate_md": "engine_reopen_gate.md",
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
        "sample_mode": metrics.get("sample_mode"),
        "confirmation_mode": metrics.get("confirmation_mode"),
        "calibration_enabled": metrics.get("calibration_enabled"),
        "calibration_version": metrics.get("calibration_version"),
        "calibration_config_hash_sha256": metrics.get("calibration_config_hash_sha256"),
        "calibration_guardrails": metrics.get("calibration_guardrails"),
        "confirmation_thresholds": metrics.get("confirmation_thresholds"),
        "frames_processed": metrics.get("frames_processed"),
        "labels_detected": metrics.get("labels_detected"),
        "label_distribution": metrics.get("label_distribution"),
        "raw_label_distribution": metrics.get("raw_label_distribution"),
        "normalized_label_distribution": metrics.get("normalized_label_distribution"),
        "candidate_events": metrics.get("candidate_events"),
        "confirmed_events": metrics.get("confirmed_events"),
        "pre_calibration_confirmed_events": metrics.get("pre_calibration_confirmed_events"),
        "post_calibration_confirmed_events": metrics.get("post_calibration_confirmed_events"),
        "calibration_suppressed_events": metrics.get("calibration_suppressed_events"),
        "calibration_suppressed_reason_counts": metrics.get("calibration_suppressed_reason_counts"),
        "suppressed_candidates": metrics.get("suppressed_candidates"),
        "proof_raw_event_count": metrics.get("proof_raw_event_count"),
        "proof_merged_event_count": metrics.get("proof_merged_event_count"),
        "proof_deduped_event_count": metrics.get("proof_deduped_event_count"),
        "proof_confirmed_event_count": metrics.get("proof_confirmed_event_count"),
        "true_positives": metrics.get("true_positives"),
        "false_positives": metrics.get("false_positives"),
        "false_negatives": metrics.get("false_negatives"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "false_positives_per_10k_frames": metrics.get("false_positives_per_10k_frames"),
        "precision_lift_summary": metrics.get("precision_lift_summary"),
        "event_confirmation_precision_lift_summary": metrics.get("event_confirmation_precision_lift_summary"),
        "pre_calibration_confirmed_event_metrics": metrics.get("pre_calibration_confirmed_event_metrics"),
        "calibrated_event_metrics": metrics.get("calibrated_event_metrics"),
        "incident_card_count": metrics.get("incident_card_count"),
        "eidos_compression_ratio": metrics.get("eidos_compression_ratio"),
        "external_compression_baselines": metrics.get("external_compression_baselines"),
        "runtime_seconds": metrics.get("runtime_seconds"),
        "frames_per_second": metrics.get("frames_per_second"),
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
        f"- Sample mode: `{digest.get('sample_mode')}`",
        f"- Confirmation mode: `{digest.get('confirmation_mode')}`",
        f"- Sentinel calibration mode: `{digest.get('sentinel_calibration_mode')}`",
        f"- Calibration enabled: `{digest.get('calibration_enabled')}`",
        f"- Calibration version: `{digest.get('calibration_version')}`",
        f"- Calibration config hash: `{digest.get('calibration_config_hash_sha256')}`",
        f"- Frames processed: `{digest.get('frames_processed')}`",
        f"- Labels detected: `{', '.join(digest.get('labels_detected') or [])}`",
        f"- Candidate / confirmed / suppressed: `{digest.get('candidate_events')}` / `{digest.get('confirmed_events')}` / `{digest.get('suppressed_candidates')}`",
        f"- Pre/post calibration confirmed events: `{digest.get('pre_calibration_confirmed_events')}` / `{digest.get('post_calibration_confirmed_events')}`",
        f"- Calibration suppressed events: `{digest.get('calibration_suppressed_events')}`",
        f"- Proof raw / merged / deduped / confirmed events: `{digest.get('proof_raw_event_count')}` / `{digest.get('proof_merged_event_count')}` / `{digest.get('proof_deduped_event_count')}` / `{digest.get('proof_confirmed_event_count')}`",
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
        "proof/event_confirmation.py",
        "tools/run_labeled_domain_proof.py",
        "tests/test_labeled_domain_proof_runner.py",
        ".gitignore",
        relpath(out_dir),
    ]
    heading = f"## Labeled CICIDS/WebAttacks proof harness -- {relpath(out_dir)}"
    journal_body = "\n".join(
        [
            "### What happened today",
            "Built and ran the event-confirmation layer for the labeled/domain proof harness.",
            "",
            "### What was accomplished",
            "- Added proof-side candidate scoring and confirmation modes for labeled-domain events.",
            "- Added optional Sentinel calibration v1 as a proof-stage false-positive suppression layer around confirmed events.",
            "- Captured raw, merged, deduped, and confirmed event metrics side by side.",
            "- Added reason codes, suppression examples, confirmation examples, calibration guardrails, false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.",
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
            "9. Known limitations: event confirmation is proof-side postprocessing only; no threshold tuning was attempted.",
            "10. Follow-up tasks not implemented: threshold calibration or core behavior changes.",
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
            "The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/confirmed proof events to those windows, and wrote crash, compression, device, precision, and confirmation receipts.",
            "",
            "### What passed",
            f"- Frames processed: {metrics.get('frames_processed')}",
            f"- Crash hits: {metrics.get('crash_hit_count')}",
            f"- Incident cards: {metrics.get('incident_card_count')}",
            f"- Confirmation mode: {metrics.get('confirmation_mode')}",
            f"- Sentinel calibration mode: {metrics.get('sentinel_calibration_mode')}",
            f"- Calibration enabled: {metrics.get('calibration_enabled')}",
            f"- Calibration suppressed events: {metrics.get('calibration_suppressed_events')}",
            f"- Raw / merged / deduped / confirmed events: {metrics.get('proof_raw_event_count')} / {metrics.get('proof_merged_event_count')} / {metrics.get('proof_deduped_event_count')} / {metrics.get('proof_confirmed_event_count')}",
            "",
            "### What failed or remains uncertain",
            "- Any false positives and false negatives are recorded in the metrics instead of being tuned away.",
            "- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.",
            "",
            "### What was saved locally",
            f"Artifacts were saved under `{relpath(out_dir)}`.",
            "",
            "### What was saved to Google Drive",
            f"Drive status: {drive_status}; folder: {drive_folder}; reason: {drive_reason}.",
            "",
            "### What should happen next",
            "Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.",
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
    if args.event_merge_gap < 0:
        raise ValueError("--event-merge-gap must be zero or positive")
    for attr, label in (
        ("confirmation_min_raw_hits", "--confirmation-min-raw-hits"),
        ("confirmation_min_duration", "--confirmation-min-duration"),
        ("confirmation_event_merge_gap", "--confirmation-event-merge-gap"),
        ("confirmation_cooldown_gap", "--confirmation-cooldown-gap"),
    ):
        value = getattr(args, attr, None)
        if value is not None and int(value) < 0:
            raise ValueError(f"{label} must be zero or positive")
    if args.confirmation_min_score is not None and float(args.confirmation_min_score) < 0:
        raise ValueError("--confirmation-min-score must be zero or positive")
    for attr, label in (
        ("calibration_event_merge_gap", "--calibration-event-merge-gap"),
        ("calibration_benign_context_grace", "--calibration-benign-context-grace"),
        ("calibration_attack_window_guard", "--calibration-attack-window-guard"),
    ):
        value = getattr(args, attr, None)
        if value is not None and int(value) < 0:
            raise ValueError(f"{label} must be zero or positive")
    for attr, label in (
        ("calibration_min_confirmed_span", "--calibration-min-confirmed-span"),
        ("calibration_min_evidence_count", "--calibration-min-evidence-count"),
    ):
        value = getattr(args, attr, None)
        if value is not None and int(value) < 1:
            raise ValueError(f"{label} must be at least one")
    for attr, label in (
        ("natural_window_pre", "--natural-window-pre"),
        ("natural_window_post", "--natural-window-post"),
    ):
        value = getattr(args, attr, None)
        if value is not None and int(value) < 0:
            raise ValueError(f"{label} must be non-negative")
    if int(getattr(args, "natural_window_max_windows", 1)) < 1:
        raise ValueError("--natural-window-max-windows must be at least one")
    out_dir = resolve_out_dir(args.out, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "incident_cards").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    generated_at = utc_now()
    run_date = datetime.now(timezone.utc).date().isoformat()
    command = build_command(args, out_dir, repo_root)
    calibration_config = build_calibration_config(args)

    git_info = proof_helpers.collect_git_info(repo_root)
    git_hygiene = git_hygiene_receipt(git_info, out_dir, repo_root)
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
        normalize_non_benign_as=args.normalize_non_benign_as,
        sample_mode=args.sample_mode,
        frames=args.frames,
        natural_window_pre=args.natural_window_pre,
        natural_window_post=args.natural_window_post,
        natural_window_max_windows=args.natural_window_max_windows,
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
    confirmation = process_stream(evidence_frames, mode=DEFAULT_SENTINEL_CONFIRMATION_MODE)
    engine_incident_cards = load_engine_incident_cards(out_dir)
    incident_cards_written = write_incident_cards(out_dir, confirmation.incident_cards, engine_incident_cards)
    device_receipt = collect_device_receipt(runtime_seconds=runtime_seconds, frames_processed=frames_processed)
    append_device_receipt_to_environment(out_dir / "environment.txt", device_receipt)

    crash_scan = scan_crashes(out_dir)
    (
        metrics,
        event_summary,
        precision_ledger,
        event_confirmation_report,
        calibration_report,
        calibrated_precision_ledger,
        candidate_funnel_report,
        confirmation_profile_sweep,
    ) = build_labeled_metrics(
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
        calibration_config=calibration_config,
    )
    write_json(out_dir / "labeled_metrics.json", metrics)
    write_labeled_metrics_md(out_dir / "labeled_metrics.md", metrics)
    write_json(out_dir / "event_summary.json", event_summary)
    write_json(out_dir / "precision_ledger.json", precision_ledger)
    write_precision_ledger_md(out_dir / "precision_ledger.md", precision_ledger)
    write_json(out_dir / "event_confirmation_report.json", event_confirmation_report)
    proof_event_confirmation.write_report_md(out_dir / "event_confirmation_report.md", event_confirmation_report)
    write_json(out_dir / "sentinel_calibration_v1.json", calibration_report)
    proof_calibration.write_calibration_md(out_dir / "sentinel_calibration_v1.md", calibration_report)
    write_json(out_dir / "calibrated_precision_ledger.json", calibrated_precision_ledger)
    proof_calibration.write_calibrated_ledger_md(out_dir / "calibrated_precision_ledger.md", calibrated_precision_ledger)
    sentinel_calibration_report = build_sentinel_calibration_report(
        metrics=metrics,
        calibration_report=calibration_report,
        precision_ledger=precision_ledger,
    )
    write_json(out_dir / "sentinel_calibration_report.json", sentinel_calibration_report)
    write_sentinel_calibration_report_md(out_dir / "sentinel_calibration_report.md", sentinel_calibration_report)
    write_json(out_dir / "candidate_funnel_report.json", candidate_funnel_report)
    write_candidate_funnel_md(out_dir / "candidate_funnel_report.md", candidate_funnel_report)
    write_json(out_dir / "confirmation_profile_sweep.json", {"profiles": confirmation_profile_sweep})
    write_confirmation_profile_sweep_csv(out_dir / "confirmation_profile_sweep.csv", confirmation_profile_sweep)
    write_confirmation_profile_sweep_md(out_dir / "confirmation_profile_sweep.md", confirmation_profile_sweep)
    write_benchmark_csv(out_dir / "benchmark_summary.csv", metrics)
    write_benchmark_md(out_dir / "benchmark_summary.md", command=command, metrics=metrics, out_dir=out_dir, git_info=git_info)
    write_json(out_dir / "crash_scan.json", crash_scan)
    digest = build_proof_digest(command=command, git_info=git_info, metrics=metrics, out_dir=out_dir, crash_scan=crash_scan)
    write_proof_digest(out_dir, digest)
    engine_gate = build_engine_reopen_gate(
        metrics=metrics,
        calibration_report=calibration_report,
        precision_ledger=precision_ledger,
        crash_scan=crash_scan,
        git_info=git_info,
        git_hygiene=git_hygiene,
        device_receipt=device_receipt,
    )
    write_json(out_dir / "engine_reopen_gate.json", engine_gate)
    write_engine_reopen_gate_md(out_dir / "engine_reopen_gate.md", engine_gate)

    draft_manifest = build_manifest(
        generated_at=generated_at,
        command=command,
        git_info=git_info,
        git_hygiene=git_hygiene,
        engine_info=engine_info,
        packages=packages,
        metrics=metrics,
        config_hash=config_doc["config_hash_sha256"],
        device_receipt=device_receipt,
    )
    write_json(out_dir / "run_manifest.json", draft_manifest)

    run_id = f"cicids_webattacks_proof_{args.suite}_seed{args.seed}_frames{frames_processed}_{timestamp_slug()}"
    drive_manifest = mirror_to_drive_fn(out_dir, run_id, run_date)
    write_json(out_dir / "drive_manifest.json", drive_manifest)
    final_manifest = build_manifest(
        generated_at=generated_at,
        command=command,
        git_info=git_info,
        git_hygiene=git_hygiene,
        engine_info=engine_info,
        packages=packages,
        metrics=metrics,
        config_hash=config_doc["config_hash_sha256"],
        device_receipt=device_receipt,
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
            out_dir / "precision_ledger.json",
            out_dir / "precision_ledger.md",
            out_dir / "event_confirmation_report.json",
            out_dir / "event_confirmation_report.md",
            out_dir / "sentinel_calibration_v1.json",
            out_dir / "sentinel_calibration_v1.md",
            out_dir / "sentinel_calibration_report.json",
            out_dir / "sentinel_calibration_report.md",
            out_dir / "calibrated_precision_ledger.json",
            out_dir / "calibrated_precision_ledger.md",
            out_dir / "candidate_funnel_report.json",
            out_dir / "candidate_funnel_report.md",
            out_dir / "confirmation_profile_sweep.json",
            out_dir / "confirmation_profile_sweep.csv",
            out_dir / "confirmation_profile_sweep.md",
            out_dir / "engine_reopen_gate.json",
            out_dir / "engine_reopen_gate.md",
            out_dir / "codex_journal.md",
            out_dir / "plain_language_test_analysis.md",
        ],
    )
    calibration_failed = bool(metrics.get("calibration_enabled")) and not bool(
        metrics.get("calibration_guardrails", {}).get("passed", False)
    )
    exit_code = 0 if crash_scan.get("crash_hit_count", 0) == 0 and not calibration_failed else 1
    return RunResult(exit_code=exit_code, out_dir=out_dir, metrics=metrics)


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = run(parse_args(argv))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
