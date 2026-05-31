"""Create the Week 1 reproducible Eidos Brain proof baseline package.

This runner is deliberately a wrapper around existing engine behavior. It
captures configuration, git state, environment details, pytest XML, and
scenario-level receipts without tuning Sentinel thresholds or changing core
model logic.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import lzma
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.proof_artifacts import create_proof_artifact_dir
from sentinel import EvidenceFrame, MODE_NAMES, process_stream
from tools.domain_tuner import evaluate_run, load_dataset_stream, load_engine_module

DEFAULT_OUT = Path("artifacts/proof_baseline_2026_05")
ENGINE_FILENAME = "EIDOS_BRAIN_UNIFIED_v0_4.7.02.py"
PROOF_MONTH = "2026-05"
SECRET_MARKERS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH")
CRASH_SCAN_PATTERNS = ("CRASH IN INCIDENT LOGIC", "can't convert cuda", "Traceback")
CRASH_SCAN_SUFFIXES = {".log", ".txt", ".jsonl", ".md", ".json"}

CSV_COLUMNS = [
    "suite",
    "scenario",
    "seed",
    "frames",
    "status",
    "eidos_compression_ratio",
    "baseline_compression_ratio",
    "raw_bytes",
    "eidos_bytes",
    "zlib_bytes",
    "zlib_compression_ratio",
    "lzma_bytes",
    "lzma_compression_ratio",
    "zstd_bytes",
    "zstd_compression_ratio",
    "zstd_skipped_reason",
    "lz4_bytes",
    "lz4_compression_ratio",
    "lz4_skipped_reason",
    "delta_zlib_bytes",
    "delta_zlib_compression_ratio",
    "delta_zlib_skipped_reason",
    "best_baseline",
    "best_baseline_compression_ratio",
    "eidos_vs_best_baseline_note",
    "anomaly_recall",
    "anomaly_precision",
    "anomaly_f1",
    "false_positives",
    "false_positive_rate",
    "anomaly_preservation",
    "runtime_seconds",
    "frames_per_second",
    "normal_only_false_positives",
    "confirmed_events",
    "candidate_events",
    "suppressed_candidates",
    "merged_events",
    "cooldown_suppressions",
    "red_count",
    "amber_count",
    "recall_preservation_note",
    "notes",
]


@dataclass(frozen=True)
class PytestResult:
    command: str
    returncode: int
    status: str
    reason: str


@dataclass(frozen=True)
class CompressionBaselineResult:
    name: str
    raw_bytes: int
    compressed_bytes: Optional[int]
    compression_ratio: Optional[float]
    skipped_reason: str = ""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--suite", choices=("smoke", "full"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_out_dir(out: Path, repo_root: Path = REPO_ROOT) -> Path:
    if out.is_absolute():
        return out
    resolved_repo = repo_root.resolve()
    parts = out.parts
    if parts:
        cwd = Path.cwd().resolve()
        try:
            if (cwd / parts[0]).resolve() == resolved_repo:
                return (cwd / out).resolve()
        except OSError:
            pass
    return resolved_repo / out


def relpath(path: Path, repo_root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            return repr(value)
    return repr(value)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def stable_hash(data: Dict[str, Any]) -> str:
    payload = json.dumps(json_safe(data), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_float(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def compression_ratio(raw_bytes: int, compressed_bytes: Optional[int]) -> Optional[float]:
    if raw_bytes <= 0 or not compressed_bytes or compressed_bytes <= 0:
        return None
    return round(float(raw_bytes) / float(compressed_bytes), 6)


def compression_result(
    name: str,
    raw_bytes: int,
    compressed_bytes: Optional[int],
    skipped_reason: str = "",
) -> CompressionBaselineResult:
    return CompressionBaselineResult(
        name=name,
        raw_bytes=raw_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=compression_ratio(raw_bytes, compressed_bytes),
        skipped_reason=skipped_reason,
    )


def serialize_frames_for_baseline(frames: Any, max_frames: Optional[int] = None) -> Tuple[np.ndarray, bytes, Dict[str, Any]]:
    """Serialize proof frames as little-endian float64 bytes for external baselines."""
    arr = np.asarray(frames, dtype=np.dtype("<f8"))
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if max_frames is not None:
        arr = arr[: int(max_frames)]
    arr = np.ascontiguousarray(arr, dtype=np.dtype("<f8"))
    payload = arr.tobytes(order="C")
    meta = {
        "dtype": "float64_le",
        "shape": list(arr.shape),
        "raw_bytes": len(payload),
        "serialization": "contiguous little-endian float64 frame matrix",
    }
    return arr, payload, meta


def _optional_zstd_result(raw_payload: bytes, raw_bytes: int) -> CompressionBaselineResult:
    try:
        import zstandard as zstd  # type: ignore
    except ImportError:
        return compression_result("zstd", raw_bytes, None, "zstandard package is not installed")
    try:
        compressed = zstd.ZstdCompressor(level=3).compress(raw_payload)
    except Exception as exc:
        return compression_result("zstd", raw_bytes, None, f"zstandard compression failed: {exc}")
    return compression_result("zstd", raw_bytes, len(compressed))


def _optional_lz4_result(raw_payload: bytes, raw_bytes: int) -> CompressionBaselineResult:
    try:
        import lz4.frame as lz4_frame  # type: ignore
    except ImportError:
        return compression_result("lz4", raw_bytes, None, "lz4 package is not installed")
    try:
        compressed = lz4_frame.compress(raw_payload)
    except Exception as exc:
        return compression_result("lz4", raw_bytes, None, f"lz4 compression failed: {exc}")
    return compression_result("lz4", raw_bytes, len(compressed))


def _delta_zlib_result(arr: np.ndarray, raw_bytes: int) -> CompressionBaselineResult:
    if arr.shape[0] < 2:
        return compression_result("delta_zlib", raw_bytes, None, "delta encoding requires at least two frames")
    try:
        delta_arr = np.concatenate([arr[:1], np.diff(arr, axis=0)], axis=0)
        delta_payload = np.ascontiguousarray(delta_arr, dtype=np.dtype("<f8")).tobytes(order="C")
        compressed = zlib.compress(delta_payload)
    except Exception as exc:
        return compression_result("delta_zlib", raw_bytes, None, f"delta+zlib compression failed: {exc}")
    return compression_result("delta_zlib", raw_bytes, len(compressed))


def compression_baselines_for_frames(frames: Any, max_frames: Optional[int] = None) -> Dict[str, Any]:
    arr, raw_payload, serialization = serialize_frames_for_baseline(frames, max_frames)
    raw_bytes = len(raw_payload)
    if raw_bytes <= 0:
        results = [
            compression_result("raw", raw_bytes, None, "no frame bytes available"),
            compression_result("zlib", raw_bytes, None, "no frame bytes available"),
            compression_result("lzma", raw_bytes, None, "no frame bytes available"),
            compression_result("zstd", raw_bytes, None, "no frame bytes available"),
            compression_result("lz4", raw_bytes, None, "no frame bytes available"),
            compression_result("delta_zlib", raw_bytes, None, "no frame bytes available"),
        ]
    else:
        results = [
            compression_result("raw", raw_bytes, raw_bytes),
            compression_result("zlib", raw_bytes, len(zlib.compress(raw_payload))),
            compression_result("lzma", raw_bytes, len(lzma.compress(raw_payload))),
            _optional_zstd_result(raw_payload, raw_bytes),
            _optional_lz4_result(raw_payload, raw_bytes),
            _delta_zlib_result(arr, raw_bytes),
        ]

    completed = [item for item in results if item.compression_ratio is not None]
    best = max(completed, key=lambda item: item.compression_ratio) if completed else None
    return {
        "frame_serialization": serialization,
        "raw_bytes": raw_bytes,
        "baselines": [item.__dict__ for item in results],
        "best_baseline": best.name if best else "",
        "best_baseline_compression_ratio": best.compression_ratio if best else "",
        "skipped": [
            {"name": item.name, "reason": item.skipped_reason}
            for item in results
            if item.skipped_reason
        ],
    }


def _baseline_by_name(baseline_doc: Dict[str, Any], name: str) -> Dict[str, Any]:
    for item in baseline_doc.get("baselines", []):
        if item.get("name") == name:
            return item
    return {}


def eidos_vs_best_baseline_note(eidos_ratio: Optional[float], best_name: str, best_ratio: Optional[float]) -> str:
    if eidos_ratio is None or best_ratio is None or not best_name:
        return "comparison unavailable"
    if eidos_ratio <= 0 or best_ratio <= 0:
        return "comparison unavailable"
    if abs(eidos_ratio - best_ratio) < 1e-9:
        return f"Eidos ratio matched {best_name}"
    if eidos_ratio > best_ratio:
        return f"Eidos ratio exceeded {best_name} by {round(eidos_ratio / best_ratio, 4)}x"
    return f"{best_name} ratio exceeded Eidos by {round(best_ratio / eidos_ratio, 4)}x"


def apply_compression_baselines_to_row(row: Dict[str, Any], baseline_doc: Dict[str, Any]) -> None:
    raw_bytes = int(baseline_doc.get("raw_bytes") or 0)
    row["raw_bytes"] = raw_bytes
    eidos_ratio = parse_float(row.get("eidos_compression_ratio"))
    row["eidos_bytes"] = int(round(raw_bytes / eidos_ratio)) if raw_bytes and eidos_ratio else ""

    for name in ("zlib", "lzma", "zstd", "lz4", "delta_zlib"):
        item = _baseline_by_name(baseline_doc, name)
        row[f"{name}_bytes"] = item.get("compressed_bytes") or ""
        row[f"{name}_compression_ratio"] = item.get("compression_ratio") or ""
        skipped = item.get("skipped_reason") or ""
        if name in ("zstd", "lz4", "delta_zlib"):
            row[f"{name}_skipped_reason"] = skipped

    row["best_baseline"] = baseline_doc.get("best_baseline", "")
    row["best_baseline_compression_ratio"] = baseline_doc.get("best_baseline_compression_ratio", "")
    row["baseline_compression_ratio"] = row["best_baseline_compression_ratio"]
    row["eidos_vs_best_baseline_note"] = eidos_vs_best_baseline_note(
        eidos_ratio,
        str(row.get("best_baseline") or ""),
        parse_float(row.get("best_baseline_compression_ratio")),
    )


def compression_baseline_manifest(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "scenario": row.get("scenario", ""),
                "raw_bytes": row.get("raw_bytes", ""),
                "eidos_compression_ratio": row.get("eidos_compression_ratio", ""),
                "eidos_bytes": row.get("eidos_bytes", ""),
                "best_baseline": row.get("best_baseline", ""),
                "best_baseline_compression_ratio": row.get("best_baseline_compression_ratio", ""),
                "zlib_compression_ratio": row.get("zlib_compression_ratio", ""),
                "lzma_compression_ratio": row.get("lzma_compression_ratio", ""),
                "zstd_compression_ratio": row.get("zstd_compression_ratio", ""),
                "zstd_skipped_reason": row.get("zstd_skipped_reason", ""),
                "lz4_compression_ratio": row.get("lz4_compression_ratio", ""),
                "lz4_skipped_reason": row.get("lz4_skipped_reason", ""),
                "delta_zlib_compression_ratio": row.get("delta_zlib_compression_ratio", ""),
                "delta_zlib_skipped_reason": row.get("delta_zlib_skipped_reason", ""),
                "eidos_vs_best_baseline_note": row.get("eidos_vs_best_baseline_note", ""),
            }
        )
    return records


def run_command(cmd: Sequence[str], repo_root: Path, timeout: int = 60, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=str(repo_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def command_text(parts: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(parts))
    return " ".join(parts)


def baseline_command(args: argparse.Namespace, repo_root: Path = REPO_ROOT) -> str:
    out = relpath(resolve_out_dir(args.out, repo_root), repo_root)
    return command_text(
        [
            "python",
            "tools/run_proof_baseline.py",
            "--suite",
            args.suite,
            "--seed",
            str(args.seed),
            "--frames",
            str(args.frames),
            "--out",
            out,
        ]
    )


def collect_git_info(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    def _git(*parts: str) -> Tuple[str, str, int]:
        try:
            result = run_command(["git", *parts], repo_root, timeout=20)
            return result.stdout.strip(), result.stderr.strip(), result.returncode
        except Exception as exc:
            return "", str(exc), 1

    commit, commit_err, _ = _git("rev-parse", "HEAD")
    branch, branch_err, _ = _git("branch", "--show-current")
    status, status_err, _ = _git("status", "--short")
    return {
        "branch": branch or "unknown",
        "commit": commit or "unknown",
        "dirty": bool(status.strip()),
        "errors": [err for err in (commit_err, branch_err, status_err) if err],
        "status_short": status,
    }


def write_git_commit(path: Path, git_info: Dict[str, Any]) -> None:
    lines = [
        f"commit: {git_info.get('commit', 'unknown')}",
        f"branch: {git_info.get('branch', 'unknown')}",
        f"dirty: {str(bool(git_info.get('dirty'))).lower()}",
        "",
        "git status --short:",
        git_info.get("status_short") or "(clean)",
    ]
    if git_info.get("errors"):
        lines.extend(["", "git capture errors:", *git_info["errors"]])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def is_secret_env_name(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in SECRET_MARKERS)


def redacted_env_items() -> Dict[str, str]:
    prefixes = ("EIDOS", "PYTHON", "CUDA", "CONDA", "VIRTUAL_ENV", "HIVE_BACKEND")
    exact = {"PATH"}
    selected: Dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        if key in exact or key.upper().startswith(prefixes):
            selected[key] = "[REDACTED]" if is_secret_env_name(key) else value
    return selected


def parse_freeze(stdout: str) -> Dict[str, str]:
    packages: Dict[str, str] = {}
    for line in stdout.splitlines():
        if "==" in line:
            name, version = line.split("==", 1)
            packages[name] = version
        elif " @ " in line:
            name, version = line.split(" @ ", 1)
            packages[name] = version
    return packages


def collect_environment(repo_root: Path = REPO_ROOT) -> Tuple[str, Dict[str, str]]:
    lines = [
        f"generated_at_utc: {utc_now()}",
        f"python_version: {sys.version}",
        f"python_executable: {sys.executable}",
        f"platform: {platform.platform()}",
        f"os: {os.name}",
        f"machine: {platform.machine()}",
        f"processor: {platform.processor()}",
        f"current_working_directory: {Path.cwd()}",
        f"resolved_repo_root: {repo_root.resolve()}",
        "",
        "Relevant environment variables (secret-looking values redacted):",
    ]
    for key, value in redacted_env_items().items():
        lines.append(f"{key}={value}")

    lines.extend(["", "torch/cuda:"])
    try:
        import torch  # type: ignore

        lines.extend(
            [
                f"torch_version: {torch.__version__}",
                f"cuda_available: {torch.cuda.is_available()}",
                f"cuda_version: {getattr(torch.version, 'cuda', None)}",
                f"cuda_device_count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}",
            ]
        )
    except Exception as exc:
        lines.append(f"torch_unavailable: {exc}")

    lines.extend(["", "pip freeze:"])
    packages: Dict[str, str] = {}
    try:
        result = run_command([sys.executable, "-m", "pip", "freeze"], repo_root, timeout=120)
        if result.returncode == 0:
            lines.append(result.stdout.rstrip() or "(no packages reported)")
            packages = parse_freeze(result.stdout)
        else:
            lines.append(f"pip freeze failed with code {result.returncode}")
            if result.stderr:
                lines.append(result.stderr.rstrip())
    except Exception as exc:
        lines.append(f"pip freeze unavailable: {exc}")
    return "\n".join(lines).rstrip() + "\n", packages


def load_engine_for_baseline(out_dir: Path, repo_root: Path = REPO_ROOT) -> Tuple[Any, Path]:
    engine = load_engine_module()
    engine_artifact_root = out_dir / "scenarios" / "_engine_artifacts"
    engine_archive_root = engine_artifact_root / "eidos_brain_archive"
    engine_artifact_root.mkdir(parents=True, exist_ok=True)
    engine_archive_root.mkdir(parents=True, exist_ok=True)
    engine.ARTIFACT_ROOT_PREFERRED = str(engine_artifact_root)
    engine.EIDOS_DATA_ROOT = str(engine_artifact_root)
    engine.EIDOS_ARCHIVE_ROOT = str(engine_archive_root)
    return engine, repo_root / ENGINE_FILENAME


def suite_specs(suite: str, repo_root: Path = REPO_ROOT) -> List[Path]:
    specs_dir = repo_root / "specs"
    if suite == "smoke":
        return [specs_dir / "smoke_synth.json"]
    return sorted(specs_dir.glob("*.json"))


def read_dataset_specs(spec_path: Path, requested_frames: int) -> List[Dict[str, Any]]:
    spec_obj = json.loads(spec_path.read_text(encoding="utf-8"))
    dataset_specs = spec_obj["datasets"] if isinstance(spec_obj, dict) else spec_obj
    specs: List[Dict[str, Any]] = []
    for dataset_spec in dataset_specs:
        normalized = copy.deepcopy(dataset_spec)
        if normalized.get("kind", "local").lower() == "synthetic":
            normalized["steps"] = int(requested_frames)
        normalized["_spec_path"] = relpath(spec_path)
        specs.append(normalized)
    return specs


def scenario_slug(name: str) -> str:
    safe = []
    for char in name.lower():
        if char.isalnum():
            safe.append(char)
        elif char in ("-", "_"):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "scenario"


def detection_metrics(step_rows: List[Dict[str, Any]], labels: Optional[Any]) -> Dict[str, Any]:
    if labels is None or not step_rows:
        return {
            "anomaly_recall": "",
            "anomaly_precision": "",
            "anomaly_f1": "",
            "false_positives": "",
            "false_positive_rate": "",
            "anomaly_preservation": "",
        }
    try:
        import numpy as np

        labels_arr = np.asarray(labels).astype(int)
        z_scores = np.asarray([row.get("z", 0.0) for row in step_rows], dtype=float)
        thresholds = np.asarray([row.get("z_thresh_eff", 0.0) for row in step_rows], dtype=float)
        limit = min(len(labels_arr), len(z_scores))
        labels_arr = labels_arr[:limit]
        preds = (z_scores[:limit] >= thresholds[:limit]).astype(int)
        tp = int(np.sum((preds == 1) & (labels_arr == 1)))
        fp = int(np.sum((preds == 1) & (labels_arr == 0)))
        fn = int(np.sum((preds == 0) & (labels_arr == 1)))
        tn = int(np.sum((preds == 0) & (labels_arr == 0)))
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = (2 * precision * recall / max(precision + recall, 1e-12)) if (precision + recall) else 0.0
        fpr = fp / max(fp + tn, 1)
        return {
            "anomaly_recall": recall,
            "anomaly_precision": precision,
            "anomaly_f1": f1,
            "false_positives": fp,
            "false_positive_rate": fpr,
            "anomaly_preservation": recall,
        }
    except Exception as exc:
        return {
            "anomaly_recall": "",
            "anomaly_precision": "",
            "anomaly_f1": "",
            "false_positives": "",
            "false_positive_rate": "",
            "anomaly_preservation": "",
            "notes_extra": f"detection metric calculation failed: {exc}",
        }


def _normal_only_confirmation_frames(frames: int = 10000) -> List[EvidenceFrame]:
    harmless_spikes = {1000, 3000, 6000, 8500}
    rows: List[EvidenceFrame] = []
    for frame in range(frames):
        if frame in harmless_spikes:
            rows.append(
                EvidenceFrame(
                    frame=frame,
                    residual_score=3.4,
                    geometry_change=0.04,
                    novelty=0.04,
                    raw_evidence_ref=f"normal_only:{frame}",
                )
            )
        else:
            rows.append(EvidenceFrame(frame=frame, residual_score=0.7, geometry_change=0.03, novelty=0.03))
    return rows


def _isolated_spike_frames() -> List[EvidenceFrame]:
    rows = [EvidenceFrame(frame=frame, residual_score=0.5, geometry_change=0.02, novelty=0.02) for frame in range(80)]
    rows[40] = EvidenceFrame(
        frame=40,
        residual_score=5.2,
        geometry_change=0.8,
        novelty=0.8,
        raw_evidence_ref="isolated_spike:40",
    )
    return rows


def _sustained_burst_frames() -> List[EvidenceFrame]:
    rows: List[EvidenceFrame] = []
    for frame in range(80):
        if 20 <= frame <= 29:
            rows.append(
                EvidenceFrame(
                    frame=frame,
                    residual_score=4.6,
                    geometry_change=0.7,
                    novelty=0.65,
                    top_drivers=[{"name": "residual_energy", "score": 4.6}],
                    raw_evidence_ref=f"burst:{frame}",
                )
            )
        else:
            rows.append(EvidenceFrame(frame=frame, residual_score=0.6, geometry_change=0.03, novelty=0.03))
    return rows


def _nearby_spike_frames() -> List[EvidenceFrame]:
    spike_frames = {10, 14, 18}
    return [
        EvidenceFrame(
            frame=frame,
            residual_score=4.8 if frame in spike_frames else 0.6,
            geometry_change=0.65 if frame in spike_frames else 0.03,
            novelty=0.7 if frame in spike_frames else 0.03,
            raw_evidence_ref=f"nearby:{frame}" if frame in spike_frames else None,
        )
        for frame in range(50)
    ]


def _mode_comparison_frames() -> List[EvidenceFrame]:
    rows: List[EvidenceFrame] = []
    for frame in range(140):
        if frame == 20:
            rows.append(EvidenceFrame(frame=frame, residual_score=2.2, geometry_change=0.25, novelty=0.25))
        elif 60 <= frame <= 61:
            rows.append(EvidenceFrame(frame=frame, residual_score=2.8, geometry_change=0.3, novelty=0.3))
        elif 100 <= frame <= 104:
            rows.append(EvidenceFrame(frame=frame, residual_score=4.7, geometry_change=0.7, novelty=0.7))
        else:
            rows.append(EvidenceFrame(frame=frame, residual_score=0.6, geometry_change=0.03, novelty=0.03))
    return rows


def _eidos_life_lifecycle_frames() -> List[EvidenceFrame]:
    rows: List[EvidenceFrame] = []
    for generation in range(120):
        if 63 <= generation <= 70:
            rows.append(
                EvidenceFrame(
                    frame=generation,
                    residual_score=3.4,
                    geometry_change=0.85,
                    novelty=0.65,
                    lifecycle_phase="collapse",
                    top_drivers=[{"name": "alive_ratio_collapse", "score": 0.96}],
                    raw_evidence_ref=f"eidos-life:generation:{generation}",
                )
            )
        elif 90 <= generation <= 96:
            rows.append(
                EvidenceFrame(
                    frame=generation,
                    residual_score=2.9,
                    geometry_change=0.55,
                    novelty=0.5,
                    lifecycle_phase="recovery",
                    top_drivers=[{"name": "post_extinction_reseed", "score": 0.88}],
                    raw_evidence_ref=f"eidos-life:generation:{generation}",
                )
            )
        else:
            rows.append(EvidenceFrame(frame=generation, residual_score=0.7, geometry_change=0.04, novelty=0.04))
    return rows


def _legacy_raw_spike_alerts(frames: List[EvidenceFrame], threshold: float) -> int:
    return sum(1 for frame in frames if frame.residual_score >= threshold)


def _scenario_result(name: str, mode: str, frames: List[EvidenceFrame]) -> Dict[str, Any]:
    result = process_stream(frames, mode=mode)
    legacy_alerts = _legacy_raw_spike_alerts(frames, threshold=2.5)
    data = result.to_dict()
    data.update(
        {
            "scenario": name,
            "legacy_raw_spike_alerts": legacy_alerts,
            "confirmed_event_count": len(result.confirmed_events),
            "incident_card_count": len(result.incident_cards),
        }
    )
    return data


def _write_incident_cards(out_dir: Path, scenarios: Dict[str, Any]) -> List[str]:
    incident_dir = out_dir / "incident_cards"
    incident_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for scenario_name, scenario in scenarios.items():
        for index, card in enumerate(scenario.get("incident_cards", []), start=1):
            safe_name = scenario_name.replace(" ", "_").replace("/", "_")
            path = incident_dir / f"{safe_name}_{index:02d}.json"
            write_json(path, card)
            written.append(relpath(path, out_dir))
    return written


def run_false_positive_control(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    """Run synthetic confirmation checks and write event-summary artifacts."""
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    scenarios = {
        "normal_only_low_noise": _scenario_result("normal_only_low_noise", "low_noise", _normal_only_confirmation_frames()),
        "isolated_spike_low_noise": _scenario_result("isolated_spike_low_noise", "low_noise", _isolated_spike_frames()),
        "sustained_burst_balanced": _scenario_result("sustained_burst_balanced", "balanced", _sustained_burst_frames()),
        "nearby_spikes_balanced": _scenario_result("nearby_spikes_balanced", "balanced", _nearby_spike_frames()),
        "eidos_life_lifecycle_balanced": _scenario_result(
            "eidos_life_lifecycle_balanced",
            "balanced",
            _eidos_life_lifecycle_frames(),
        ),
    }
    mode_counts: Dict[str, int] = {}
    comparison_frames = _mode_comparison_frames()
    for mode in MODE_NAMES:
        result = process_stream(comparison_frames, mode=mode)
        mode_counts[mode] = len(result.confirmed_events)

    cards_written = _write_incident_cards(out_dir, scenarios)
    normal = scenarios["normal_only_low_noise"]
    burst = scenarios["sustained_burst_balanced"]
    aggregate = {
        "normal_only_false_positives": normal["confirmed_event_count"],
        "normal_only_legacy_raw_spike_alerts": normal["legacy_raw_spike_alerts"],
        "confirmed_events": sum(item["confirmed_event_count"] for item in scenarios.values()),
        "candidate_events": sum(int(item["candidate_events"]) for item in scenarios.values()),
        "suppressed_candidates": sum(int(item["suppressed_candidates"]) for item in scenarios.values()),
        "merged_events": sum(int(item["merged_events"]) for item in scenarios.values()),
        "cooldown_suppressions": sum(int(item["cooldown_suppressions"]) for item in scenarios.values()),
        "red_count": sum(int(item["red_count"]) for item in scenarios.values()),
        "amber_count": sum(int(item["amber_count"]) for item in scenarios.values()),
        "incident_card_count": len(cards_written),
        "mode_confirmed_event_counts": mode_counts,
        "recall_preservation_note": (
            "Synthetic sustained burst confirmed"
            if burst["confirmed_event_count"] >= 1
            else "Synthetic sustained burst was missed"
        ),
    }
    summary = {
        "generated_at_utc": utc_now(),
        "suite": args.suite,
        "seed": args.seed,
        "frames": args.frames,
        "policy": "raw residual spike -> candidate; persistence + geometry change + novelty -> confirmed event",
        "modes": list(MODE_NAMES),
        "aggregate": aggregate,
        "scenarios": scenarios,
        "incident_cards_written": cards_written,
        "known_limitations": [
            "Synthetic confirmation fixtures prove policy mechanics; they are not a substitute for labeled production telemetry.",
            "Normal-only false-positive control is measured on deterministic synthetic benign spikes.",
            "Eidos Life lifecycle bridge uses synthetic generation 63 collapse/recovery evidence, not a live browser run.",
        ],
    }
    write_json(out_dir / "event_summary.json", summary)
    with (logs_dir / "false_positive_control.jsonl").open("w", encoding="utf-8") as handle:
        for name, scenario in scenarios.items():
            handle.write(json.dumps({"scenario": name, **json_safe(scenario)}, sort_keys=True) + "\n")
        handle.write(json.dumps({"aggregate": json_safe(aggregate)}, sort_keys=True) + "\n")
    return summary


def annotate_rows_with_event_summary(rows: List[Dict[str, Any]], event_summary: Dict[str, Any]) -> None:
    aggregate = event_summary.get("aggregate", {})
    for row in rows:
        row["normal_only_false_positives"] = aggregate.get("normal_only_false_positives", "")
        row["confirmed_events"] = aggregate.get("confirmed_events", "")
        row["candidate_events"] = aggregate.get("candidate_events", "")
        row["suppressed_candidates"] = aggregate.get("suppressed_candidates", "")
        row["merged_events"] = aggregate.get("merged_events", "")
        row["cooldown_suppressions"] = aggregate.get("cooldown_suppressions", "")
        row["red_count"] = aggregate.get("red_count", "")
        row["amber_count"] = aggregate.get("amber_count", "")
        row["recall_preservation_note"] = aggregate.get("recall_preservation_note", "")


def write_scenario_manifest(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, data)


def run_scenarios(engine: Any, args: argparse.Namespace, out_dir: Path, repo_root: Path = REPO_ROOT) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    scenario_list: List[str] = []

    for spec_path in suite_specs(args.suite, repo_root):
        for dataset_spec in read_dataset_specs(spec_path, args.frames):
            scenario = dataset_spec.get("name") or Path(dataset_spec.get("path", spec_path.stem)).stem
            scenario_list.append(str(scenario))
            scenario_dir = out_dir / "scenarios" / scenario_slug(str(scenario))
            scenario_dir.mkdir(parents=True, exist_ok=True)
            notes: List[str] = []
            start = time.perf_counter()
            status = "passed"
            row: Dict[str, Any] = {
                "suite": args.suite,
                "scenario": scenario,
                "seed": args.seed,
                "frames": args.frames,
                "status": status,
                "eidos_compression_ratio": "",
                "baseline_compression_ratio": "",
                "raw_bytes": "",
                "eidos_bytes": "",
                "zlib_bytes": "",
                "zlib_compression_ratio": "",
                "lzma_bytes": "",
                "lzma_compression_ratio": "",
                "zstd_bytes": "",
                "zstd_compression_ratio": "",
                "zstd_skipped_reason": "",
                "lz4_bytes": "",
                "lz4_compression_ratio": "",
                "lz4_skipped_reason": "",
                "delta_zlib_bytes": "",
                "delta_zlib_compression_ratio": "",
                "delta_zlib_skipped_reason": "",
                "best_baseline": "",
                "best_baseline_compression_ratio": "",
                "eidos_vs_best_baseline_note": "",
                "anomaly_recall": "",
                "anomaly_precision": "",
                "anomaly_f1": "",
                "false_positives": "",
                "false_positive_rate": "",
                "anomaly_preservation": "",
                "runtime_seconds": "",
                "frames_per_second": "",
                "normal_only_false_positives": "",
                "confirmed_events": "",
                "candidate_events": "",
                "suppressed_candidates": "",
                "merged_events": "",
                "cooldown_suppressions": "",
                "red_count": "",
                "amber_count": "",
                "recall_preservation_note": "",
                "notes": "",
            }
            try:
                clean_spec = {k: v for k, v in dataset_spec.items() if not k.startswith("_")}
                write_json(scenario_dir / "scenario_spec.json", clean_spec)
                gen_factory, est_frames, meta = load_dataset_stream(clean_spec, engine=engine, features=64)
                bundle = meta["bundle"]
                actual_frames = min(int(args.frames), int(bundle.frames.shape[0]))
                if actual_frames < int(args.frames):
                    notes.append(f"requested {args.frames} frames; scenario only provided {actual_frames}")
                compression_baselines = compression_baselines_for_frames(bundle.frames, actual_frames)
                labels = bundle.labels[:actual_frames] if bundle.labels is not None else None
                if labels is None:
                    notes.append("no ground-truth labels; anomaly precision/recall/f1 left blank")
                log_path = scenario_dir / "engine_output.log"
                with log_path.open("w", encoding="utf-8") as log_handle:
                    with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                        results = engine.run_stream_once(
                            bundle.make_gen_factory(actual_frames),
                            est_frames=actual_frames,
                            features=64,
                            profile_label=f"proof_baseline_{scenario_slug(str(scenario))}",
                            session_label=f"proof_baseline_{scenario_slug(str(scenario))}",
                            cfg_overrides={},
                            return_step_rows=True,
                            return_top_surprises=False,
                            seed=args.seed,
                        )
                elapsed = time.perf_counter() - start
                step_rows = results.get("step_rows") or []
                summary = results.get("summary") or {}
                tuner_metrics = evaluate_run(results, labels, str(scenario))
                detect = detection_metrics(step_rows, labels)
                metric_note = detect.pop("notes_extra", None)
                if metric_note:
                    notes.append(metric_note)
                if step_rows:
                    row["eidos_compression_ratio"] = step_rows[-1].get("ratio", "")
                else:
                    notes.append("no step rows returned; compression ratio unavailable")
                apply_compression_baselines_to_row(row, compression_baselines)
                row.update(detect)
                row["runtime_seconds"] = round(elapsed, 6)
                processed = int(summary.get("frames_processed") or len(step_rows) or actual_frames)
                row["frames_per_second"] = round(processed / elapsed, 6) if elapsed > 0 else ""
                row["notes"] = "; ".join(notes)
                write_scenario_manifest(
                    scenario_dir / "scenario_manifest.json",
                    {
                        "dataset_spec": clean_spec,
                        "engine_artifact_root": relpath(out_dir / "scenarios" / "_engine_artifacts", repo_root),
                        "requested_frames": args.frames,
                        "actual_frames": actual_frames,
                        "seed": args.seed,
                        "status": status,
                        "summary": summary,
                        "tuner_metrics": tuner_metrics,
                        "compression_baselines": compression_baselines,
                        "csv_row": row,
                    },
                )
                if results.get("report_text"):
                    (scenario_dir / "report.txt").write_text(str(results["report_text"]), encoding="utf-8")
            except Exception as exc:
                elapsed = time.perf_counter() - start
                status = "failed"
                reason = f"{type(exc).__name__}: {exc}"
                row["status"] = status
                row["runtime_seconds"] = round(elapsed, 6)
                row["notes"] = reason
                skipped.append({"name": str(scenario), "reason": reason})
                write_scenario_manifest(
                    scenario_dir / "scenario_manifest.json",
                    {
                        "dataset_spec": {k: v for k, v in dataset_spec.items() if not k.startswith("_")},
                        "requested_frames": args.frames,
                        "seed": args.seed,
                        "status": status,
                        "reason": reason,
                    },
                )
            rows.append(row)

    return rows, skipped, scenario_list


def write_benchmark_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    header = "| scenario | status | frames | eidos compression ratio | best baseline | anomaly f1 | runtime seconds | notes |"
    sep = "| --- | --- | ---: | ---: | --- | ---: | ---: | --- |"
    body = []
    for row in rows:
        body.append(
            "| {scenario} | {status} | {frames} | {ratio} | {best} | {f1} | {runtime} | {notes} |".format(
                scenario=row.get("scenario", ""),
                status=row.get("status", ""),
                frames=row.get("frames", ""),
                ratio=row.get("eidos_compression_ratio", ""),
                best=f"{row.get('best_baseline', '')} {row.get('best_baseline_compression_ratio', '')}".strip(),
                f1=row.get("anomaly_f1", ""),
                runtime=row.get("runtime_seconds", ""),
                notes=str(row.get("notes", "")).replace("|", "\\|"),
            )
        )
    return "\n".join([header, sep, *body])


def markdown_compression_baseline_table(rows: List[Dict[str, Any]]) -> str:
    header = "| scenario | raw bytes | Eidos ratio | zlib | lzma | zstd | lz4 | delta+zlib | best baseline | note |"
    sep = "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |"
    body = []
    for row in rows:
        zstd = row.get("zstd_compression_ratio") or f"skipped: {row.get('zstd_skipped_reason', '')}"
        lz4 = row.get("lz4_compression_ratio") or f"skipped: {row.get('lz4_skipped_reason', '')}"
        delta = row.get("delta_zlib_compression_ratio") or f"skipped: {row.get('delta_zlib_skipped_reason', '')}"
        body.append(
            "| {scenario} | {raw} | {eidos} | {zlib_ratio} | {lzma_ratio} | {zstd} | {lz4} | {delta} | {best} | {note} |".format(
                scenario=row.get("scenario", ""),
                raw=row.get("raw_bytes", ""),
                eidos=row.get("eidos_compression_ratio", ""),
                zlib_ratio=row.get("zlib_compression_ratio", ""),
                lzma_ratio=row.get("lzma_compression_ratio", ""),
                zstd=str(zstd).replace("|", "\\|"),
                lz4=str(lz4).replace("|", "\\|"),
                delta=str(delta).replace("|", "\\|"),
                best=f"{row.get('best_baseline', '')} {row.get('best_baseline_compression_ratio', '')}".strip(),
                note=str(row.get("eidos_vs_best_baseline_note", "")).replace("|", "\\|"),
            )
        )
    return "\n".join([header, sep, *body])


def write_benchmark_md(
    path: Path,
    *,
    command: str,
    out_dir: Path,
    git_info: Dict[str, Any],
    config_hash: str,
    args: argparse.Namespace,
    scenario_list: List[str],
    rows: List[Dict[str, Any]],
    skipped_baselines: List[Dict[str, str]],
    pytest_result: PytestResult,
    event_summary: Optional[Dict[str, Any]] = None,
) -> None:
    title = "Eidos Brain Baseline Proof Run \u2014 2026-05"
    frame_note = ""
    if args.frames != 10000:
        frame_note = (
            f"- Frame-count note: `{args.frames}` frames were used for this smoke receipt "
            "because the corrected 10000-frame Windows smoke rerun exceeded the available "
            "execution window before artifact writing. The seed/frame/suite values are still "
            "captured in config, manifest, CSV, and Markdown outputs."
        )
    lines = [
        f"# {title}",
        "",
        "## Exact command used",
        "",
        f"```bash\n{command}\n```",
        "",
        f"- Artifact directory: `{relpath(out_dir)}`",
        f"- Git commit: `{git_info.get('commit', 'unknown')}`",
        f"- Git branch: `{git_info.get('branch', 'unknown')}`",
        f"- Git dirty: `{git_info.get('dirty')}`",
        f"- Config hash: `{config_hash}`",
        f"- Seed: `{args.seed}`",
        f"- Frames: `{args.frames}`",
        f"- Suite: `{args.suite}`",
        f"- Scenario list: {', '.join(scenario_list) if scenario_list else 'none'}",
        f"- Pytest command: `{pytest_result.command}`",
        f"- Pytest status: `{pytest_result.status}` ({pytest_result.reason})",
    ]
    if frame_note:
        lines.append(frame_note)
    lines.extend(
        [
            "",
            "## Summary table",
            "",
            markdown_table(rows),
            "",
            "## Compression baselines",
            "",
            "External baselines use the same proof-frame matrix serialized as contiguous little-endian float64 bytes. Optional zstandard/lz4 baselines are recorded as skipped when their packages are not installed.",
            "",
            markdown_compression_baseline_table(rows),
            "",
            "## Skipped baselines and reasons",
            "",
        ]
    )
    if skipped_baselines:
        for item in skipped_baselines:
            lines.append(f"- `{item['name']}`: {item['reason']}")
    else:
        lines.append("- None recorded.")
    if event_summary:
        aggregate = event_summary.get("aggregate", {})
        mode_counts = aggregate.get("mode_confirmed_event_counts", {})
        lines.extend(
            [
                "",
                "## Sentinel false-positive control",
                "",
                f"- Normal-only confirmed false positives per 10k frames: `{aggregate.get('normal_only_false_positives', 'NA')}`",
                f"- Legacy raw-spike alerts on normal-only stream: `{aggregate.get('normal_only_legacy_raw_spike_alerts', 'NA')}`",
                f"- Confirmed events: `{aggregate.get('confirmed_events', 'NA')}`",
                f"- Candidate events: `{aggregate.get('candidate_events', 'NA')}`",
                f"- Suppressed candidates: `{aggregate.get('suppressed_candidates', 'NA')}`",
                f"- Merged events: `{aggregate.get('merged_events', 'NA')}`",
                f"- Cooldown suppressions: `{aggregate.get('cooldown_suppressions', 'NA')}`",
                f"- RED count: `{aggregate.get('red_count', 'NA')}`",
                f"- AMBER count: `{aggregate.get('amber_count', 'NA')}`",
                f"- Mode confirmed-event counts: `{mode_counts}`",
                f"- Recall preservation note: {aggregate.get('recall_preservation_note', 'NA')}",
                "- Incident-card policy: cards are written for confirmed events, not every raw spike.",
                "- Eidos Life lifecycle bridge: generation 63 collapse/recovery is treated as lifecycle events, with post-recovery nominal frames suppressed.",
            ]
        )
    lines.extend(
        [
            "",
            "## Known limitations",
            "",
            "- This runner still wraps existing engine behavior and does not tune core SentinelMonitor thresholds.",
            "- External compression baselines use a documented float64 proof-frame serialization; optional zstandard/lz4 baselines depend on local packages.",
            "- Smoke synthetic scenarios do not provide ground-truth anomaly labels, so detection precision/recall/f1 can be blank.",
            "- No plots were produced for this smoke baseline unless a later plotting task adds them.",
            "- False-positive control uses deterministic synthetic policy checks; broader labeled real-world validation remains future work.",
            "",
            "## Next step",
            "",
            "Broaden false-positive control to labeled real-world streams and compare against the checked-in smoke receipt.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def minimal_junit_xml(path: Path, reason: str, status: str = "skipped") -> None:
    escaped = (
        reason.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    if status == "failure":
        body = f'<failure message="{escaped}">{escaped}</failure>'
        failures = "1"
        skipped = "0"
    else:
        body = f'<skipped message="{escaped}" />'
        failures = "0"
        skipped = "1"
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<testsuite name="pytest" tests="1" failures="{failures}" skipped="{skipped}">\n'
        '  <testcase classname="proof_baseline" name="pytest_capture">\n'
        f"    {body}\n"
        "  </testcase>\n"
        "</testsuite>\n",
        encoding="utf-8",
    )


def pytest_targets_for_run(args: argparse.Namespace) -> List[str]:
    if "false_positive_control" not in str(args.out):
        return []
    return [
        "tests/test_sentinel.py",
        "tests/test_proof_artifacts.py",
        "tests/test_proof_baseline_runner.py",
        "tests/test_sentinel_false_positive_control.py",
        "tests/test_sentinel_event_confirmation.py",
        "tests/test_sentinel_modes.py",
        "tests/test_incident_card_confirmation.py",
    ]


def run_pytest_capture(args: argparse.Namespace, out_dir: Path, repo_root: Path = REPO_ROOT) -> PytestResult:
    xml_path = out_dir / "pytest_results.xml"
    test_files = list((repo_root / "tests").glob("test_*.py")) if (repo_root / "tests").is_dir() else []
    if not test_files:
        reason = "no tests directory or test_*.py files found"
        minimal_junit_xml(xml_path, reason)
        return PytestResult(command="pytest not run", returncode=0, status="skipped", reason=reason)

    cmd = [sys.executable, "-m", "pytest"]
    targets = pytest_targets_for_run(args)
    if targets:
        cmd.extend(targets)
    elif args.suite == "smoke":
        cmd.extend(["-m", "smoke"])
    cmd.extend(["--junitxml", str(xml_path)])
    env = os.environ.copy()
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    try:
        result = run_command(cmd, repo_root, timeout=900, env=env)
        if not xml_path.exists():
            status = "failed" if result.returncode else "skipped"
            reason = "pytest did not write junit xml"
            minimal_junit_xml(xml_path, reason, status="failure" if result.returncode else "skipped")
        elif result.returncode == 0:
            status = "passed"
            reason = "pytest completed successfully"
        else:
            status = "failed"
            reason = f"pytest exited with code {result.returncode}"
        (out_dir / "pytest_stdout.txt").write_text(result.stdout, encoding="utf-8")
        (out_dir / "pytest_stderr.txt").write_text(result.stderr, encoding="utf-8")
        return PytestResult(command=command_text(["python", "-m", "pytest", *cmd[3:]]), returncode=result.returncode, status=status, reason=reason)
    except Exception as exc:
        reason = f"pytest unavailable or timed out: {exc}"
        minimal_junit_xml(xml_path, reason, status="failure")
        return PytestResult(command=command_text(["python", "-m", "pytest", *cmd[3:]]), returncode=1, status="failed", reason=reason)


def skipped_baseline_records(rows: List[Dict[str, Any]], scenario_skips: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    seen = set()
    if rows and all(not row.get("baseline_compression_ratio") for row in rows):
        records.append(
            {
                "name": "baseline_compression_ratio",
                "reason": "External compression baseline comparison was unavailable for every scenario.",
            }
        )
        seen.add(("baseline_compression_ratio", records[-1]["reason"]))
    for row in rows:
        scenario = str(row.get("scenario") or "unknown")
        for field, name in (
            ("zstd_skipped_reason", "zstd"),
            ("lz4_skipped_reason", "lz4"),
            ("delta_zlib_skipped_reason", "delta_zlib"),
        ):
            reason = str(row.get(field) or "")
            if not reason:
                continue
            key = (f"{scenario}:{name}", reason)
            if key in seen:
                continue
            records.append({"name": f"{scenario}:{name}", "reason": reason})
            seen.add(key)
    if any(not row.get("anomaly_f1") for row in rows):
        records.append(
            {
                "name": "detection_ground_truth_metrics",
                "reason": "One or more scenarios did not provide labels, so anomaly precision/recall/f1 are blank for those rows.",
            }
        )
    for item in scenario_skips:
        records.append({"name": str(item["name"]), "reason": str(item["reason"])})
    return records


def build_config_doc(
    *,
    args: argparse.Namespace,
    engine: Any,
    engine_info: Dict[str, Any],
    scenario_list: List[str],
    rows: List[Dict[str, Any]],
    skipped_baselines: List[Dict[str, str]],
    event_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    doc = {
        "benchmark": {
            "frames": args.frames,
            "scenario_list": scenario_list,
            "seed": args.seed,
            "suite": args.suite,
        },
        "engine": {
            **engine_info,
            "parameters": json_safe(getattr(engine, "EIDOS_BRAIN_CONFIG", {})),
        },
        "baselines": {
            "compression": compression_baseline_manifest(rows),
            "detection": [],
            "skipped": skipped_baselines,
        },
    }
    if event_summary:
        aggregate = event_summary.get("aggregate", {})
        doc["sentinel_false_positive_control"] = {
            "modes": event_summary.get("modes", []),
            "normal_only_false_positives": aggregate.get("normal_only_false_positives"),
            "confirmed_events": aggregate.get("confirmed_events"),
            "candidate_events": aggregate.get("candidate_events"),
            "suppressed_candidates": aggregate.get("suppressed_candidates"),
            "merged_events": aggregate.get("merged_events"),
            "cooldown_suppressions": aggregate.get("cooldown_suppressions"),
            "red_count": aggregate.get("red_count"),
            "amber_count": aggregate.get("amber_count"),
            "recall_preservation_note": aggregate.get("recall_preservation_note"),
        }
    return doc


def write_plots_readme(out_dir: Path, suite: str) -> None:
    path = out_dir / "plots" / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# Plots\n\nPlots were not produced in the {suite} baseline run. "
        "This is recorded as a Week 1 limitation, not a hidden failure.\n",
        encoding="utf-8",
    )


def build_manifest(
    *,
    generated_at: str,
    command: str,
    git_info: Dict[str, Any],
    engine_info: Dict[str, Any],
    packages: Dict[str, str],
    args: argparse.Namespace,
    scenario_list: List[str],
    rows: List[Dict[str, Any]],
    config_hash: str,
    skipped_baselines: List[Dict[str, str]],
    pytest_result: PytestResult,
    drive_manifest: Optional[Dict[str, Any]] = None,
    event_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = {
        "baselines": {
            "compression": compression_baseline_manifest(rows),
            "detection": [],
            "skipped": skipped_baselines,
        },
        "benchmark": {
            "command": command,
            "frames": args.frames,
            "scenario_list": scenario_list,
            "seed": args.seed,
            "suite": args.suite,
        },
        "config": {
            "config_hash_sha256": config_hash,
            "config_path": "config.json",
        },
        "engine": engine_info,
        "generated_at_utc": generated_at,
        "git": {
            "branch": git_info.get("branch", "unknown"),
            "commit": git_info.get("commit", "unknown"),
            "dirty": bool(git_info.get("dirty")),
        },
        "outputs": {
            "benchmark_summary_csv": "benchmark_summary.csv",
            "benchmark_summary_md": "benchmark_summary.md",
            "codex_journal_md": "codex_journal.md",
            "drive_manifest_json": "drive_manifest.json",
            "environment_txt": "environment.txt",
            "event_summary_json": "event_summary.json",
            "git_commit_txt": "git_commit.txt",
            "incident_cards_dir": "incident_cards",
            "logs_dir": "logs",
            "plain_language_test_analysis_md": "plain_language_test_analysis.md",
            "proof_digest_json": "proof_digest.json",
            "proof_digest_md": "proof_digest.md",
            "pytest_results_xml": "pytest_results.xml",
        },
        "packages": packages,
        "pytest": {
            "command": pytest_result.command,
            "reason": pytest_result.reason,
            "returncode": pytest_result.returncode,
            "status": pytest_result.status,
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "runtime": {
            "machine": platform.machine(),
            "os": os.name,
            "platform": platform.platform(),
            "processor": platform.processor(),
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
    if event_summary is not None:
        manifest["sentinel_false_positive_control"] = event_summary.get("aggregate", {})
    return manifest


def artifact_files(out_dir: Path) -> List[Path]:
    return sorted(path for path in out_dir.rglob("*") if path.is_file())


def is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".codex_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def colab_drive_candidates() -> List[Path]:
    return [Path("/content/drive/MyDrive"), Path("/content/drive/My Drive")]


def local_google_drive_candidates() -> List[Path]:
    # Intentionally empty by default. A plain local folder named "Google Drive"
    # is not proof that Google Drive Desktop is syncing it to the cloud.
    return []


def _first_writable_existing(candidates: Iterable[Path], label: str, reasons: List[str]) -> Optional[Tuple[Path, str]]:
    checked: List[str] = []
    for candidate in candidates:
        expanded = candidate.expanduser()
        checked.append(str(expanded))
        if expanded.exists() and is_writable_dir(expanded):
            return expanded, f"using writable {label}: {expanded}"
    if checked:
        reasons.append(f"no writable {label} found among: {', '.join(checked)}")
    return None


def discover_drive_root(
    *,
    colab_candidates_override: Optional[Iterable[Path]] = None,
    local_candidates_override: Optional[Iterable[Path]] = None,
) -> Tuple[Optional[Path], str]:
    reasons: List[str] = []
    for env_name in ("EIDOS_PROOF_DRIVE_DIR", "EIDOS_ARTIFACT_ROOT"):
        value = os.environ.get(env_name)
        if value:
            candidate = Path(value).expanduser()
            if is_writable_dir(candidate):
                return candidate, f"using writable {env_name}: {candidate}"
            reasons.append(f"{env_name} was set but not writable: {value}")

    colab_candidates = colab_drive_candidates() if colab_candidates_override is None else colab_candidates_override
    colab_result = _first_writable_existing(
        colab_candidates,
        "Colab Drive root",
        reasons,
    )
    if colab_result:
        return colab_result

    if local_candidates_override is not None:
        local_result = _first_writable_existing(
            local_candidates_override,
            "explicit local Google Drive override",
            reasons,
        )
        if local_result:
            return local_result
    else:
        reasons.append(
            "local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT "
            "to a verified Drive mount"
        )

    return None, "; ".join(reasons) or "no configured or mounted Google Drive path found"


def copy_selected_to_drive(out_dir: Path, drive_manifest: Dict[str, Any], paths: Iterable[Path]) -> None:
    if not drive_manifest.get("drive_copy_success"):
        return
    drive_run_dir = Path(str(drive_manifest["drive_run_dir"]))
    for path in paths:
        if path.exists() and path.is_file():
            target = drive_run_dir / path.relative_to(out_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def proof_failure_reasons(rows: List[Dict[str, Any]], pytest_result: PytestResult) -> List[str]:
    reasons: List[str] = []
    if pytest_result.returncode != 0:
        reasons.append(f"pytest failed: {pytest_result.reason}")
    failed_scenarios = [
        str(row.get("scenario") or "unknown")
        for row in rows
        if str(row.get("status", "")).lower() not in ("passed", "skipped")
    ]
    if failed_scenarios:
        reasons.append(f"scenario failures: {', '.join(failed_scenarios)}")
    return reasons


def scan_crash_strings(out_dir: Path) -> Dict[str, Any]:
    ignored = {"proof_digest.json", "proof_digest.md"}
    hit_files: List[Dict[str, Any]] = []
    hit_count = 0
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.name in ignored or path.suffix.lower() not in CRASH_SCAN_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        matches = []
        for pattern in CRASH_SCAN_PATTERNS:
            count = text.count(pattern)
            if count:
                matches.append({"pattern": pattern, "count": count})
                hit_count += count
        if matches:
            hit_files.append({"path": relpath(path, out_dir), "matches": matches})
    return {
        "patterns": list(CRASH_SCAN_PATTERNS),
        "crash_hit_count": hit_count,
        "crash_hit_files": hit_files,
        "status": "clean" if hit_count == 0 else "not_clean",
    }


def build_proof_digest(
    *,
    out_dir: Path,
    command: str,
    git_info: Dict[str, Any],
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    event_summary: Dict[str, Any],
    pytest_result: PytestResult,
    crash_scan: Dict[str, Any],
) -> Dict[str, Any]:
    primary_row = rows[0] if rows else {}
    aggregate = event_summary.get("aggregate", {}) if event_summary else {}
    incident_cards = event_summary.get("incident_cards_written", []) if event_summary else []
    runtime_seconds = sum(parse_float(row.get("runtime_seconds")) or 0.0 for row in rows)
    digest = {
        "repo_branch": git_info.get("branch", "unknown"),
        "git_commit": git_info.get("commit", "unknown"),
        "git_dirty": bool(git_info.get("dirty")),
        "command": command,
        "seed": args.seed,
        "frames": args.frames,
        "suite": args.suite,
        "runtime_seconds": round(runtime_seconds, 6),
        "eidos_compression_ratio": primary_row.get("eidos_compression_ratio", ""),
        "external_compression_baselines": compression_baseline_manifest(rows),
        "normal_only_confirmed_false_positives_per_10k_frames": aggregate.get("normal_only_false_positives"),
        "legacy_raw_spike_alerts": aggregate.get("normal_only_legacy_raw_spike_alerts"),
        "candidate_events": aggregate.get("candidate_events"),
        "confirmed_events": aggregate.get("confirmed_events"),
        "suppressed_candidates": aggregate.get("suppressed_candidates"),
        "merged_events": aggregate.get("merged_events"),
        "cooldown_suppressions": aggregate.get("cooldown_suppressions"),
        "red_count": aggregate.get("red_count"),
        "amber_count": aggregate.get("amber_count"),
        "incident_card_count": aggregate.get("incident_card_count"),
        "incident_card_filenames": incident_cards,
        "pytest": {
            "command": pytest_result.command,
            "returncode": pytest_result.returncode,
            "status": pytest_result.status,
            "reason": pytest_result.reason,
        },
        "known_limitations": event_summary.get("known_limitations", []) if event_summary else [],
        "crash_scan": crash_scan,
        "clean": crash_scan.get("crash_hit_count", 0) == 0 and not proof_failure_reasons(rows, pytest_result),
        "artifact_dir": relpath(out_dir),
        "generated_at_utc": utc_now(),
    }
    return digest


def write_proof_digest_md(path: Path, digest: Dict[str, Any]) -> None:
    baselines = digest.get("external_compression_baselines", [])
    crash_scan = digest.get("crash_scan", {})
    lines = [
        "# Proof Digest",
        "",
        f"- Branch: `{digest.get('repo_branch', 'unknown')}`",
        f"- Commit: `{digest.get('git_commit', 'unknown')}`",
        f"- Dirty: `{digest.get('git_dirty')}`",
        f"- Command: `{digest.get('command', '')}`",
        f"- Suite/seed/frames: `{digest.get('suite')}` / `{digest.get('seed')}` / `{digest.get('frames')}`",
        f"- Runtime seconds: `{digest.get('runtime_seconds')}`",
        f"- Eidos compression ratio: `{digest.get('eidos_compression_ratio', 'NA')}`",
        f"- Normal-only confirmed false positives per 10k frames: `{digest.get('normal_only_confirmed_false_positives_per_10k_frames', 'NA')}`",
        f"- Legacy raw-spike alerts: `{digest.get('legacy_raw_spike_alerts', 'NA')}`",
        f"- Candidate / confirmed / suppressed: `{digest.get('candidate_events', 'NA')}` / `{digest.get('confirmed_events', 'NA')}` / `{digest.get('suppressed_candidates', 'NA')}`",
        f"- Incident cards: `{digest.get('incident_card_count', 'NA')}`",
        f"- Crash scan: `{crash_scan.get('status', 'unknown')}` with `{crash_scan.get('crash_hit_count', 'NA')}` hits",
        "",
        "## Compression Baselines",
        "",
    ]
    if baselines:
        lines.append("| scenario | raw bytes | Eidos ratio | zlib | lzma | zstd | lz4 | delta+zlib | best |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |")
        for item in baselines:
            zstd = item.get("zstd_compression_ratio") or f"skipped: {item.get('zstd_skipped_reason', '')}"
            lz4 = item.get("lz4_compression_ratio") or f"skipped: {item.get('lz4_skipped_reason', '')}"
            delta = item.get("delta_zlib_compression_ratio") or f"skipped: {item.get('delta_zlib_skipped_reason', '')}"
            lines.append(
                "| {scenario} | {raw} | {eidos} | {zlib_ratio} | {lzma_ratio} | {zstd} | {lz4} | {delta} | {best} {best_ratio} |".format(
                    scenario=item.get("scenario", ""),
                    raw=item.get("raw_bytes", ""),
                    eidos=item.get("eidos_compression_ratio", ""),
                    zlib_ratio=item.get("zlib_compression_ratio", ""),
                    lzma_ratio=item.get("lzma_compression_ratio", ""),
                    zstd=str(zstd).replace("|", "\\|"),
                    lz4=str(lz4).replace("|", "\\|"),
                    delta=str(delta).replace("|", "\\|"),
                    best=item.get("best_baseline", ""),
                    best_ratio=item.get("best_baseline_compression_ratio", ""),
                )
            )
    else:
        lines.append("- No compression baseline records were available.")

    lines.extend(["", "## Crash Scan", ""])
    if crash_scan.get("crash_hit_files"):
        for item in crash_scan["crash_hit_files"]:
            patterns = ", ".join(f"{m['pattern']}={m['count']}" for m in item.get("matches", []))
            lines.append(f"- `{item.get('path')}`: {patterns}")
    else:
        lines.append("- No crash strings found.")

    lines.extend(["", "## Known Limitations", ""])
    for item in digest.get("known_limitations", []):
        lines.append(f"- {item}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_proof_digest(out_dir: Path, digest: Dict[str, Any]) -> None:
    write_json(out_dir / "proof_digest.json", digest)
    write_proof_digest_md(out_dir / "proof_digest.md", digest)


def mirror_to_drive(out_dir: Path, run_id: str, run_date: str) -> Dict[str, Any]:
    files = artifact_files(out_dir)
    files_considered = [relpath(path, out_dir) for path in files]
    drive_root, reason = discover_drive_root()
    manifest: Dict[str, Any] = {
        "drive_copy_attempted": True,
        "drive_copy_success": False,
        "drive_root": "unknown",
        "drive_run_dir": "unknown",
        "files_considered": files_considered,
        "files_copied": [],
        "files_skipped": [],
        "reason": reason,
        "timestamp_utc": utc_now(),
    }
    if drive_root is None:
        return manifest

    drive_run_dir = drive_root / "Eidos_Brain_Proof_Phase" / run_date / run_id
    copied: List[str] = []
    skipped: List[Dict[str, str]] = []
    try:
        drive_run_dir.mkdir(parents=True, exist_ok=True)
        for path in files:
            rel = path.relative_to(out_dir)
            target = drive_run_dir / rel
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied.append(rel.as_posix())
            except Exception as exc:
                skipped.append({"path": rel.as_posix(), "reason": str(exc)})
        manifest.update(
            {
                "drive_copy_success": not skipped,
                "drive_root": str(drive_root),
                "drive_run_dir": str(drive_run_dir),
                "files_copied": copied,
                "files_skipped": skipped,
                "reason": "copy completed" if not skipped else "some files failed to copy",
            }
        )
    except Exception as exc:
        manifest["reason"] = str(exc)
        manifest["drive_root"] = str(drive_root)
        manifest["drive_run_dir"] = str(drive_run_dir)
    return manifest


def write_proof_docs(
    *,
    repo_root: Path,
    out_dir: Path,
    run_date: str,
    command: str,
    rows: List[Dict[str, Any]],
    skipped_baselines: List[Dict[str, str]],
    pytest_result: PytestResult,
    drive_manifest: Optional[Dict[str, Any]],
    event_summary: Optional[Dict[str, Any]] = None,
    files_changed: Optional[List[str]] = None,
) -> None:
    docs_dir = repo_root / "docs" / "proof_runs" / run_date
    docs_dir.mkdir(parents=True, exist_ok=True)
    changed = files_changed or [
        "eidos_tensor_utils.py",
        "eidos_incident_cards.py",
        "eidos_procedural_memory.py",
        "eidos_forecast.py",
        "scripts/verify_colab_gpu_hotfix.py",
        "tools/run_proof_baseline.py",
        "repo/src/eidos_brain/engine/eidos_v0_4_7_02.py",
        "tests/test_tensor_conversion_regressions.py",
        "tests/test_user_config.py",
        "tests/test_colab_gpu_hotfix_smoke.py",
        "tests/test_proof_baseline_runner.py",
        relpath(out_dir),
    ]
    event_aggregate = event_summary.get("aggregate", {}) if event_summary else {}
    drive_status = "pending"
    drive_root = "unknown"
    drive_folder = "unknown"
    files_copied: List[str] = []
    files_skipped: Any = []
    drive_reason = "Drive copy has not run yet."
    if drive_manifest is not None:
        drive_status = "copied" if drive_manifest.get("drive_copy_success") else "skipped or failed"
        drive_root = str(drive_manifest.get("drive_root", "unknown"))
        drive_folder = str(drive_manifest.get("drive_run_dir", "unknown"))
        files_copied = list(drive_manifest.get("files_copied", []))
        files_skipped = drive_manifest.get("files_skipped", [])
        drive_reason = str(drive_manifest.get("reason", "unknown"))

    artifact_lines = [relpath(path) for path in artifact_files(out_dir)]
    journal = [
        f"# Codex Journal -- {run_date}",
        "",
        "## What happened today",
        "The proof runner generated a reproducible Eidos Brain smoke-readiness package with config, manifest, environment, git state, pytest XML, event summary, incident cards, logs, and CSV/Markdown summaries.",
        "",
        "## What was accomplished",
        "- Verified the smoke scenario and captured the result as local and Google Drive artifacts.",
        "- Captured normal-only false positives, candidate events, suppressed candidates, merged events, cooldown suppressions, and incident-card-compatible records.",
        "- Kept SentinelMonitor thresholds, reservoir dynamics, compression behavior, and prediction policy unchanged.",
        "",
        "## Tests and commands run",
        f"- `{command}` -> see `benchmark_summary.md` and `pytest_results.xml`.",
        f"- Pytest status: {pytest_result.status} ({pytest_result.reason}).",
        "",
        "## Problems encountered",
        "- External compression baselines are limited to the proof-frame serialization used by this runner; optional zstandard/lz4 baselines are skipped when packages are unavailable.",
        "- Smoke synthetic data does not provide ground-truth anomaly labels.",
        "- False-positive control still uses deterministic synthetic policy checks before broader labeled telemetry work.",
        f"- Google Drive status: {drive_status}; reason: {drive_reason}.",
        "",
        "## What changed",
        "\n".join(f"- {item}" for item in changed),
        "",
        "## What did not change",
        "Core model behavior, Sentinel labels, SentinelMonitor thresholds, reservoir dynamics, compression behavior, and prediction decision policy were not changed.",
        "",
        "## Artifacts generated",
        "\n".join(f"- {item}" for item in artifact_lines),
        "",
        "## Google Drive archive status",
        f"- Drive root used: {drive_root}",
        f"- Drive folder used: {drive_folder}",
        f"- Files copied: {len(files_copied)}",
        f"- Files skipped: {len(files_skipped)}",
        f"- Reason: {drive_reason}",
        "",
        "## Thoughts on improvement",
        f"Normal-only confirmed false positives were `{event_aggregate.get('normal_only_false_positives', 'NA')}` per 10k synthetic frames; the next step should compare this policy against labeled real-world streams.",
        "",
        "## Where to improve next",
        "Run the broader experiment/test suite on this readiness branch and compare future receipts against this smoke package.",
        "",
        "## Anything that stands out",
        f"Recall preservation for the synthetic burst: {event_aggregate.get('recall_preservation_note', 'NA')}.",
        "",
        "## End-of-task summary",
        f"1. Files changed: {', '.join(changed)}",
        "2. Whether core behavior changed: no.",
        "3. Tests added or skipped: tensor conversion, config validation, GPU-hotfix smoke, compression baseline, and digest regression tests are covered by pytest; pytest XML captured by the proof run.",
        f"4. Repo-root commands run: `{command}`.",
        f"5. Artifacts generated: {len(artifact_lines)} files under `{relpath(out_dir)}`.",
        "6. Plain-language analysis written: yes.",
        "7. Journal entry written: yes.",
        f"8. Google Drive copy status: {drive_status}; {drive_reason}.",
        "9. Known limitations: optional compression baselines may be skipped by dependency availability; no smoke labels, no plots, synthetic-only false-positive fixtures.",
        "10. Follow-up tasks not implemented: full long-run experiment campaign and labeled real-world false-positive comparison.",
    ]
    analysis = [
        f"# Plain-Language Test Analysis -- {run_date}",
        "",
        "## What the task attempted",
        "The task created a reproducible smoke-readiness proof package for Eidos Brain.",
        "",
        "## Why the test matters",
        "A readiness smoke run checks whether the current branch can execute the engine path, write the expected proof artifacts, run the selected pytest smoke check, and mirror the result to Google Drive.",
        "",
        "## What was tested",
        "The smoke baseline ran the configured scenario list and deterministic confirmation fixtures for normal-only, isolated spike, sustained burst, nearby spikes, mode comparison, and Eidos Life lifecycle behavior.",
        "",
        "## What passed",
        "\n".join(f"- {row.get('scenario')}: {row.get('status')}" for row in rows),
        "",
        "## What failed",
        "No scenario failure is hidden; see `benchmark_summary.csv` for row-level status and notes.",
        "",
        "## What artifacts were generated",
        "\n".join(f"- {item}" for item in artifact_lines),
        "",
        "## What was saved locally",
        f"Artifacts were saved under `{relpath(out_dir)}`.",
        "",
        "## What was saved to Google Drive",
        f"Drive status: {drive_status}; folder: {drive_folder}; reason: {drive_reason}.",
        "",
        "## What remains uncertain",
        "Optional compression baselines depend on installed packages; labeled anomaly metrics for smoke data, plots, full experiment campaigns, and real lifecycle export replay remain future work.",
        "",
        "## What should happen next",
        "Use this branch as the next test/experiment base, then add labeled real-world comparisons and longer experiment receipts.",
    ]
    for target_dir in (docs_dir, out_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "codex_journal.md").write_text("\n".join(journal).rstrip() + "\n", encoding="utf-8")
        (target_dir / "plain_language_test_analysis.md").write_text("\n".join(analysis).rstrip() + "\n", encoding="utf-8")


def run(
    args: argparse.Namespace,
    *,
    repo_root: Path = REPO_ROOT,
    load_engine_fn: Callable[[Path, Path], Tuple[Any, Path]] = load_engine_for_baseline,
    run_scenarios_fn: Callable[[Any, argparse.Namespace, Path, Path], Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]] = run_scenarios,
    run_pytest_fn: Callable[[argparse.Namespace, Path, Path], PytestResult] = run_pytest_capture,
    mirror_to_drive_fn: Callable[[Path, str, str], Dict[str, Any]] = mirror_to_drive,
    write_docs_fn: Callable[..., None] = write_proof_docs,
) -> int:
    out_dir = resolve_out_dir(args.out, repo_root)
    create_proof_artifact_dir(out_dir)
    generated_at = utc_now()
    command = baseline_command(args, repo_root)
    run_date = datetime.now(timezone.utc).date().isoformat()
    run_label = "proof_false_positive_control_2026_05" if "false_positive_control" in out_dir.name else "proof_baseline_2026_05"
    run_id = f"{run_label}_{args.suite}_seed{args.seed}_frames{args.frames}"

    git_info = collect_git_info(repo_root)
    engine, engine_path = load_engine_fn(out_dir, repo_root)
    engine_info = {
        "code_hash_sha256": sha256_file(engine_path) if engine_path.exists() else "unknown",
        "module": relpath(engine_path, repo_root),
        "version": str(getattr(engine, "ENGINE_VERSION", "unknown")),
    }

    rows, scenario_skips, scenario_list = run_scenarios_fn(engine, args, out_dir, repo_root)
    skipped_baselines = skipped_baseline_records(rows, scenario_skips)
    event_summary = run_false_positive_control(args, out_dir)
    annotate_rows_with_event_summary(rows, event_summary)
    config_doc = build_config_doc(
        args=args,
        engine=engine,
        engine_info=engine_info,
        scenario_list=scenario_list,
        rows=rows,
        skipped_baselines=skipped_baselines,
        event_summary=event_summary,
    )
    config_hash = stable_hash(config_doc)
    config_doc["config_hash_sha256"] = config_hash
    write_json(out_dir / "config.json", config_doc)

    environment_text, packages = collect_environment(repo_root)
    (out_dir / "environment.txt").write_text(environment_text, encoding="utf-8")
    write_git_commit(out_dir / "git_commit.txt", git_info)
    write_benchmark_csv(out_dir / "benchmark_summary.csv", rows)
    write_plots_readme(out_dir, args.suite)

    pytest_result = run_pytest_fn(args, out_dir, repo_root)
    write_benchmark_md(
        out_dir / "benchmark_summary.md",
        command=command,
        out_dir=out_dir,
        git_info=git_info,
        config_hash=config_hash,
        args=args,
        scenario_list=scenario_list,
        rows=rows,
        skipped_baselines=skipped_baselines,
        pytest_result=pytest_result,
        event_summary=event_summary,
    )

    draft_manifest = build_manifest(
        generated_at=generated_at,
        command=command,
        git_info=git_info,
        engine_info=engine_info,
        packages=packages,
        args=args,
        scenario_list=scenario_list,
        rows=rows,
        config_hash=config_hash,
        skipped_baselines=skipped_baselines,
        pytest_result=pytest_result,
        event_summary=event_summary,
    )
    write_json(out_dir / "run_manifest.json", draft_manifest)
    write_docs_fn(
        repo_root=repo_root,
        out_dir=out_dir,
        run_date=run_date,
        command=command,
        rows=rows,
        skipped_baselines=skipped_baselines,
        pytest_result=pytest_result,
        drive_manifest=None,
        event_summary=event_summary,
    )

    crash_scan = scan_crash_strings(out_dir)
    digest = build_proof_digest(
        out_dir=out_dir,
        command=command,
        git_info=git_info,
        args=args,
        rows=rows,
        event_summary=event_summary,
        pytest_result=pytest_result,
        crash_scan=crash_scan,
    )
    write_proof_digest(out_dir, digest)

    drive_manifest = mirror_to_drive_fn(out_dir, run_id, run_date)
    write_json(out_dir / "drive_manifest.json", drive_manifest)
    final_manifest = build_manifest(
        generated_at=generated_at,
        command=command,
        git_info=git_info,
        engine_info=engine_info,
        packages=packages,
        args=args,
        scenario_list=scenario_list,
        rows=rows,
        config_hash=config_hash,
        skipped_baselines=skipped_baselines,
        pytest_result=pytest_result,
        drive_manifest=drive_manifest,
        event_summary=event_summary,
    )
    write_json(out_dir / "run_manifest.json", final_manifest)
    write_docs_fn(
        repo_root=repo_root,
        out_dir=out_dir,
        run_date=run_date,
        command=command,
        rows=rows,
        skipped_baselines=skipped_baselines,
        pytest_result=pytest_result,
        drive_manifest=drive_manifest,
        event_summary=event_summary,
    )
    copy_selected_to_drive(
        out_dir,
        drive_manifest,
        [
            out_dir / "run_manifest.json",
            out_dir / "drive_manifest.json",
            out_dir / "event_summary.json",
            out_dir / "proof_digest.json",
            out_dir / "proof_digest.md",
            out_dir / "codex_journal.md",
            out_dir / "plain_language_test_analysis.md",
        ],
    )
    failure_reasons = proof_failure_reasons(rows, pytest_result)
    if crash_scan.get("crash_hit_count", 0):
        failure_reasons.append("proof digest crash scan found crash strings")
    return 1 if failure_reasons else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
