"""Fail-closed, benchmark-only data gate for a future memory utility study.

This prepares inputs, never runs a detector or approves a memory policy.
Run from the git repository root: python -m proof.memory_utility_data --help.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets")
ATTACKS = {"webattackbruteforce", "webattackxss", "webattacksqlinjection"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def label_value(label):
    # CIC exports use several encodings of the dash, including replacement glyphs.
    # Only these three explicit attack names are accepted; unknown is never benign.
    key = re.sub("[^a-z0-9]", "", label.lower())
    if key == "benign":
        return 0
    if key in ATTACKS:
        return 1
    return None


def read_dataset(path, *, timestamp_column="Timestamp", timestamp_format=None, collect=False):
    """Audit every record. Never infer chronology from row number or label order."""
    path = Path(path)
    initial_hash = sha(path)
    counts, unknown, errors = Counter(), Counter(), Counter()
    finite = np.zeros(len(FEATURES), dtype=int)
    records, previous_time, reversals, max_duration = [], None, 0, 0.0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = [s.strip() for s in next(reader, [])]
        duplicates = {k: v for k, v in Counter(header).items() if v > 1}
        required = (*FEATURES, "Label")
        bad_columns = [name for name in required if header.count(name) != 1]
        timestamp_available = header.count(timestamp_column) == 1
        indices = [header.index(name) for name in FEATURES] if not bad_columns else []
        label_index = header.index("Label") if header.count("Label") == 1 else None
        ts_index = header.index(timestamp_column) if timestamp_available else None
        rows = 0
        for source_index, row in enumerate(reader):
            rows += 1
            if len(row) != len(header):
                errors["row_width_mismatch"] += 1
                continue
            raw_label = row[label_index].strip() if label_index is not None else ""
            counts[raw_label] += 1
            label = label_value(raw_label)
            if label is None:
                unknown[raw_label] += 1
            values = []
            for j, index in enumerate(indices):
                try:
                    value = float(row[index])
                except ValueError:
                    value = math.nan
                if math.isfinite(value):
                    finite[j] += 1
                values.append(value)
            duration = values[0] if values else math.nan
            if not math.isfinite(duration) or duration < 0:
                errors["invalid_flow_duration"] += 1
            else:
                max_duration = max(max_duration, duration / 1_000_000)
            start = end = None
            if timestamp_available and timestamp_format:
                try:
                    start = datetime.strptime(row[ts_index].strip(), timestamp_format)
                    if start.tzinfo is not None:
                        raise ValueError("use consistent naive capture-local timestamps")
                    if math.isfinite(duration) and duration >= 0:
                        end = start + timedelta(microseconds=duration)
                        reversals += int(previous_time is not None and end < previous_time)
                        previous_time = end
                except (ValueError, OverflowError):
                    errors["invalid_timestamp"] += 1
            if collect:
                records.append((source_index, values, label, start, end))
    blockers = []
    if not rows:
        blockers.append("empty_dataset")
    if bad_columns:
        blockers.append("missing_or_ambiguous_required_columns")
    if not timestamp_available:
        blockers.append("missing_or_ambiguous_timestamp_column")
    elif not timestamp_format:
        blockers.append("explicit_timestamp_format_required")
    if errors:
        blockers.extend(sorted(errors))
    if unknown:
        blockers.append("unrecognized_labels")
    if sha(path) != initial_hash:
        blockers.append("source_changed_during_read")
    audit = dict(
        source_name=path.name, source_sha256=initial_hash, source_bytes=path.stat().st_size,
        rows=rows, feature_columns=list(FEATURES), duplicate_headers=duplicates,
        bad_required_columns=bad_columns, label_counts=dict(counts), unknown_labels=dict(unknown),
        benign_rows=sum(v for k, v in counts.items() if label_value(k) == 0),
        attack_rows=sum(v for k, v in counts.items() if label_value(k) == 1),
        finite_feature_values=dict(zip(FEATURES, finite.tolist())), errors=dict(errors),
        timestamp_column=timestamp_column, timestamp_format=timestamp_format,
        timestamp_available=timestamp_available,
        availability_order_reversals=reversals if timestamp_available and timestamp_format else None,
        maximum_flow_duration_seconds=max_duration,
        blockers=blockers, schema_status="blocked" if blockers else "passed",
        chronology="unknown" if not timestamp_available or not timestamp_format else "parsed_capture_local_time",
        raw_data_saved=False,
        limitations=["Local byte identity is not publisher checksum authentication.",
                     "Labels are the supplied CICIDS research labels, not independently relabeled truth.",
                     "A valid schema does not establish operational utility or a valid holdout."],
    )
    return audit, records


def fit_scaler(prefix):
    """Prefix mean/std only; nonfinite values become prefix means, then zero."""
    prefix = np.asarray(prefix, dtype=np.float64)
    if prefix.ndim != 2 or not len(prefix):
        raise ValueError("nonempty two-dimensional exploration prefix required")
    finite = np.isfinite(prefix)
    count = finite.sum(axis=0)
    if np.any(count == 0):
        raise ValueError("feature has no finite exploration observations")
    mean = np.where(finite, prefix, 0).sum(axis=0) / count
    filled = np.where(finite, prefix, mean)
    scale = np.sqrt(np.mean((filled - mean) ** 2, axis=0))
    scale = np.maximum(scale, 1e-6)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("nonfinite scaler; feature magnitudes overflowed")
    return dict(mean=mean.tolist(), scale=scale.tolist(), clip=3.0,
                fit_rows=len(prefix), nonfinite_counts=(~finite).sum(axis=0).tolist(),
                transform="clip((finite_value_or_prefix_mean - prefix_mean)/prefix_std, -3, 3)/3")


def transform(values, scaler):
    values = np.asarray(values, dtype=np.float64)
    mean, scale = np.array(scaler["mean"]), np.array(scaler["scale"])
    filled = np.where(np.isfinite(values), values, mean)
    return np.clip((filled - mean) / scale, -3, 3) / 3


def split_records(records, cutoff, gap_seconds, max_duration):
    if cutoff.tzinfo is not None:
        raise ValueError("cutoff must use naive capture-local time, matching parsed timestamps")
    if not math.isfinite(gap_seconds) or gap_seconds < max_duration or gap_seconds <= 0:
        raise ValueError("positive gap must cover the maximum observed flow duration")
    # Complete-flow features only become available at flow completion, not at start.
    ordered = sorted(records, key=lambda r: (r[4], r[0]))
    boundary = cutoff + timedelta(seconds=gap_seconds)
    prefix = np.array([r[4] < cutoff for r in ordered])
    suffix = np.array([r[4] >= boundary for r in ordered])
    if prefix.sum() < 2 or suffix.sum() < 2:
        raise ValueError("need at least two exploration and two evaluation records")
    if any(r[3] < cutoff for r, use in zip(ordered, suffix) if use):
        raise ValueError("evaluation flow overlaps the exploration interval")
    labels = np.array([r[2] for r in ordered], dtype=np.int8)
    if set(labels[suffix].tolist()) != {0, 1}:
        raise ValueError("evaluation suffix must contain both benign and attack labels")
    values = np.array([r[1] for r in ordered], dtype=np.float64)
    scaler = fit_scaler(values[prefix])
    x = transform(values, scaler)
    partition = np.where(prefix, 0, np.where(suffix, 2, 1)).astype(np.int8)
    return ordered, x, labels, partition, scaler


def verify(out):
    out = Path(out)
    freeze = json.loads((out / "freeze.json").read_text(encoding="utf-8"))
    for name, digest in freeze["files"].items():
        if sha(out / name) != digest:
            raise ValueError(f"frozen artifact changed: {name}")
    return {"status": "passed", "files_verified": len(freeze["files"]),
            "scope": "byte integrity only; not a utility gate"}


def prepare(path, out, *, cutoff=None, gap_seconds=None, timestamp_column="Timestamp", timestamp_format=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="started", started_utc=datetime.now(timezone.utc).isoformat(),
                    core_behavior_changed=False, detector_exercised=False, utility_status="untested")
    write_json(out / "run_manifest.json", manifest)
    try:
        audit, records = read_dataset(path, timestamp_column=timestamp_column,
                                      timestamp_format=timestamp_format, collect=cutoff is not None)
        write_json(out / "dataset_audit.json", audit)
        blockers = list(audit["blockers"])
        if cutoff is None:
            blockers.append("exploration_cutoff_not_frozen")
        if gap_seconds is None:
            blockers.append("overlap_gap_not_frozen")
        spec = dict(dataset_sha256=audit["source_sha256"], feature_columns=list(FEATURES),
                    timestamp_column=timestamp_column, timestamp_format=timestamp_format,
                    exploration_cutoff=cutoff, gap_seconds=gap_seconds,
                    ordering="stable flow completion time = timestamp + Flow Duration in microseconds; source row breaks ties",
                    preprocessing="fit prefix only; replay prefix, gap and suffix with frozen scaler",
                    state_handoff="continuous; reset once before prefix; no reset at evaluation boundary",
                    labels="separate scoring_labels.npz; never pass labels, IPs or label windows to model",
                    prediction_targets="one-step targets must stay within their own partition",
                    adoption_status="inconclusive", acceptable_overhead_threshold=None,
                    smallest_useful_effect=None,
                    reason="Project FP reduction/recall preservation objective exists, but no task-specific effect magnitude or overhead budget is established.",
                    blockers=blockers)
        if not blockers:
            try:
                ordered, x, labels, partition, scaler = split_records(
                    records, datetime.fromisoformat(cutoff), gap_seconds,
                    audit["maximum_flow_duration_seconds"])
            except ValueError as exc:
                blockers.append(str(exc))
            else:
                np.savez_compressed(out / "model_inputs.npz", x=x, partition=partition,
                                    source_rows=np.array([r[0] for r in ordered], dtype=np.int64))
                np.savez_compressed(out / "scoring_labels.npz", labels=labels)
                write_json(out / "scaler.json", scaler)
                write_json(out / "order.json", {"completion_timestamps": [r[4].isoformat() for r in ordered]})
                spec["partition_rows"] = {k: int((partition == i).sum()) for i, k in enumerate(("exploration", "gap", "evaluation"))}
                spec["note"] = "Dataset preparation passed; no candidate evaluator or operational utility is validated."
        write_json(out / "protocol.json", spec)
        manifest.update(status="blocked" if blockers else "prepared", blockers=blockers,
                        completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        manifest.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        write_json(out / "run_manifest.json", manifest)
        raise
    write_json(out / "run_manifest.json", manifest)
    (out / "evaluator.py").write_bytes(Path(__file__).read_bytes())
    files = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    write_json(out / "freeze.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "files": files,
                                    "scope": "Immutable data-preparation receipt; not a frozen utility evaluator."})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="audit data and fail closed unless a chronological split is possible")
    prep.add_argument("--file", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--timestamp-column", default="Timestamp")
    prep.add_argument("--timestamp-format")
    prep.add_argument("--cutoff", help="ISO capture-local timestamp; choose before candidate outcomes")
    prep.add_argument("--gap-seconds", type=float)
    check = sub.add_parser("verify", help="check frozen artifacts without running candidates")
    check.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify(args.out)
    else:
        result = prepare(args.file, args.out, cutoff=args.cutoff, gap_seconds=args.gap_seconds,
                         timestamp_column=args.timestamp_column, timestamp_format=args.timestamp_format)
    print(json.dumps(result, indent=2))
    return 2 if result["status"] == "blocked" else 0


if __name__ == "__main__":
    sys.exit(main())
