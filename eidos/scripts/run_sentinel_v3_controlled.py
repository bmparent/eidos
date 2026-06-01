from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "repo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eidos_brain.experiments.controlled_regimes import (  # noqa: E402
    REGIME_ORDER,
    default_regime_frames,
    generate_controlled_regime_stream,
)
from eidos_brain.sentinel.calibration import finite_scalar, safe_sigma  # noqa: E402
from eidos_brain.sentinel.sentinel_v3 import SentinelV3, SentinelV3Config  # noqa: E402


STEP_FIELDS = [
    "reservoir",
    "t",
    "regime",
    "status",
    "is_surprise",
    "residual_evidence",
    "input_evidence",
    "geometry_evidence",
    "novelty_evidence",
    "event_score",
    "adaptation_frozen",
    "ema_err",
    "sigma",
    "z",
    "error_norm",
    "energy_z",
    "delta_z",
    "rolling_var_z",
    "period47_score",
    "input_evidence_score",
    "plasticity_z",
    "state_var_z",
    "eigen_dominance_z",
    "low_plasticity_score",
    "low_state_var_score",
    "rank_collapse_score",
    "threshold_multiplier",
    "collapse_counter",
    "freeze_remaining",
    "event_active",
    "episode_id",
    "episode_age",
    "merge_cooldown_remaining",
    "rls_recovered",
    "rls_p_clipped",
    "w_out_clipped",
    "recoveries",
]


ACCEPTANCE_FIELDS = [
    "reservoir",
    "normal_false_alert_rate",
    "abnormal_alert_rate",
    "frozen_red_rate",
    "noise_alert_rate",
    "backdoor_alert_rate",
    "finite_residual_stats",
    "pass_all",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Sentinel V3 controlled-regime calibration harness.")
    parser.add_argument("--reservoirs", nargs="+", type=int, default=[256, 768, 1536])
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "sentinel_v3_patch")
    parser.add_argument("--frames-per-regime", type=int, default=None)
    parser.add_argument("--normal-frames", type=int, default=None)
    parser.add_argument("--backdoor-frames", type=int, default=None)
    parser.add_argument("--noise-frames", type=int, default=None)
    parser.add_argument("--frozen-frames", type=int, default=None)
    parser.add_argument("--normal-false-alert-threshold", type=float, default=0.05)
    parser.add_argument("--abnormal-alert-threshold", type=float, default=0.80)
    parser.add_argument("--frozen-red-threshold", type=float, default=0.50)
    parser.add_argument("--noise-alert-threshold", type=float, default=0.80)
    parser.add_argument("--backdoor-alert-threshold", type=float, default=0.80)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out if args.out.is_absolute() else (REPO_ROOT / args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    regime_frames = default_regime_frames(args.frames_per_regime)
    overrides = {
        "NORMAL": args.normal_frames,
        "BACKDOOR_PERIODIC": args.backdoor_frames,
        "NOISE_CRASH": args.noise_frames,
        "FROZEN_LOW_VARIANCE": args.frozen_frames,
    }
    for name, value in overrides.items():
        if value is not None:
            regime_frames[name] = int(value)

    run_config = {
        "seed": int(args.seed),
        "reservoirs": [int(r) for r in args.reservoirs],
        "features": int(args.features),
        "warmup": int(args.warmup),
        "regime_frames": regime_frames,
        "thresholds": {
            "normal_false_alert_rate": args.normal_false_alert_threshold,
            "abnormal_alert_rate": args.abnormal_alert_threshold,
            "frozen_red_rate": args.frozen_red_threshold,
            "noise_alert_rate": args.noise_alert_threshold,
            "backdoor_alert_rate": args.backdoor_alert_threshold,
        },
    }
    config_hash = _hash_json(run_config)
    _write_json(out_dir / "config_v3_patch.json", {**run_config, "config_hash": config_hash})

    frames = list(
        generate_controlled_regime_stream(
            features=args.features,
            warmup=args.warmup,
            seed=args.seed,
            regime_frames=regime_frames,
        )
    )
    total_steps = len(frames)

    summary_rows: list[dict[str, Any]] = []
    acceptance_rows: list[dict[str, Any]] = []
    per_regime_rows: list[dict[str, Any]] = []
    full_runs: list[dict[str, Any]] = []

    for reservoir_size in args.reservoirs:
        result = _run_one_reservoir(
            frames=frames,
            out_dir=out_dir,
            reservoir_size=int(reservoir_size),
            total_steps=total_steps,
            args=args,
        )
        summary_rows.append(result["summary"])
        acceptance_rows.append(result["acceptance"])
        per_regime_rows.extend(result["per_regime"])
        full_runs.append(result["full"])

    _write_csv(out_dir / "summary_v3_patch.csv", summary_rows)
    _write_csv(out_dir / "acceptance_v3_patch.csv", acceptance_rows, fieldnames=ACCEPTANCE_FIELDS)
    _write_csv(out_dir / "per_regime_summary_v3_patch.csv", per_regime_rows)
    _write_json(
        out_dir / "summary_v3_patch_full.json",
        {
            "run_config": run_config,
            "config_hash": config_hash,
            "acceptance": acceptance_rows,
            "per_regime_summary": per_regime_rows,
            "reservoir_runs": full_runs,
        },
    )
    _write_json(out_dir / "run_manifest_v3_patch.json", _manifest(args, run_config, config_hash, total_steps))
    _write_drive_manifest(out_dir=out_dir, run_id=f"sentinel_v3_patch_{_utc_stamp()}")

    passed = any(str(row["pass_all"]).lower() == "true" for row in acceptance_rows)
    print(f"Wrote Sentinel V3 artifacts to {out_dir}")
    print(f"Acceptance pass_all present: {passed}")
    return 0 if passed else 2


def _run_one_reservoir(
    *,
    frames: Iterable[Any],
    out_dir: Path,
    reservoir_size: int,
    total_steps: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sentinel = SentinelV3(
        SentinelV3Config(
            features=int(args.features),
            reservoir_size=reservoir_size,
            warmup=int(args.warmup),
            seed=int(args.seed),
        )
    )
    step_path = out_dir / f"eidos_v3_steps_{total_steps}_reservoir_{reservoir_size}.csv"
    counts = _metric_accumulators()
    finite_residual_stats = True
    final_row: dict[str, Any] = {}
    frozen_steps = 0

    with step_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STEP_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for item in frames:
            row = sentinel.step(item.frame)
            row["reservoir"] = reservoir_size
            row["regime"] = item.regime
            final_row = row
            if item.regime != "WARMUP":
                finite_residual_stats = finite_residual_stats and _finite_residual_row(row)
                _update_counts(counts, item.regime, row)
                if row["adaptation_frozen"]:
                    frozen_steps += 1
            writer.writerow({field: _csv_value(row.get(field, "")) for field in STEP_FIELDS})

    acceptance = _acceptance_row(
        reservoir_size,
        counts,
        finite_residual_stats,
        thresholds={
            "normal": args.normal_false_alert_threshold,
            "abnormal": args.abnormal_alert_threshold,
            "frozen": args.frozen_red_threshold,
            "noise": args.noise_alert_threshold,
            "backdoor": args.backdoor_alert_threshold,
        },
    )
    per_regime = _per_regime_rows(reservoir_size, counts)
    non_warmup_frames = sum(counts[name]["frames"] for name in REGIME_ORDER)
    total_alerts = sum(counts[name]["alerts"] for name in REGIME_ORDER)
    summary = {
        **acceptance,
        "total_frames": int(total_steps),
        "evaluated_frames": int(non_warmup_frames),
        "alerts": int(total_alerts),
        "alert_rate": _rate(total_alerts, non_warmup_frames),
        "final_status": final_row.get("status", "UNKNOWN"),
        "final_ema_err": finite_scalar(final_row.get("ema_err", 0.0)),
        "final_sigma": safe_sigma(final_row.get("sigma", 1e-6), floor=1e-6),
        "final_event_score": finite_scalar(final_row.get("event_score", 0.0)),
        "adaptation_frozen_rate": _rate(frozen_steps, non_warmup_frames),
        "rls_recoveries": int(final_row.get("recoveries", 0) or 0),
        "step_csv": step_path.name,
    }
    return {
        "summary": summary,
        "acceptance": acceptance,
        "per_regime": per_regime,
        "full": {
            "reservoir": reservoir_size,
            "summary": summary,
            "acceptance": acceptance,
            "per_regime": per_regime,
            "step_csv": step_path.name,
        },
    }


def _metric_accumulators() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "frames": 0,
            "green": 0,
            "amber": 0,
            "red": 0,
            "alerts": 0,
            "residual_sum": 0.0,
            "input_sum": 0.0,
            "geometry_sum": 0.0,
            "event_sum": 0.0,
        }
        for name in REGIME_ORDER
    }


def _update_counts(counts: dict[str, dict[str, Any]], regime: str, row: dict[str, Any]) -> None:
    bucket = counts[regime]
    status = str(row["status"])
    bucket["frames"] += 1
    if status == "GREEN":
        bucket["green"] += 1
    elif status == "AMBER":
        bucket["amber"] += 1
        bucket["alerts"] += 1
    elif status == "RED":
        bucket["red"] += 1
        bucket["alerts"] += 1
    bucket["residual_sum"] += finite_scalar(row.get("residual_evidence", 0.0))
    bucket["input_sum"] += finite_scalar(row.get("input_evidence", 0.0))
    bucket["geometry_sum"] += finite_scalar(row.get("geometry_evidence", 0.0))
    bucket["event_sum"] += finite_scalar(row.get("event_score", 0.0))


def _acceptance_row(
    reservoir: int,
    counts: dict[str, dict[str, Any]],
    finite_residual_stats: bool,
    *,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    normal_false_alert_rate = _rate(counts["NORMAL"]["alerts"], counts["NORMAL"]["frames"])
    abnormal_frames = sum(counts[name]["frames"] for name in REGIME_ORDER if name != "NORMAL")
    abnormal_alerts = sum(counts[name]["alerts"] for name in REGIME_ORDER if name != "NORMAL")
    abnormal_alert_rate = _rate(abnormal_alerts, abnormal_frames)
    frozen_red_rate = _rate(counts["FROZEN_LOW_VARIANCE"]["red"], counts["FROZEN_LOW_VARIANCE"]["frames"])
    noise_alert_rate = _rate(counts["NOISE_CRASH"]["alerts"], counts["NOISE_CRASH"]["frames"])
    backdoor_alert_rate = _rate(counts["BACKDOOR_PERIODIC"]["alerts"], counts["BACKDOOR_PERIODIC"]["frames"])
    pass_all = (
        normal_false_alert_rate <= thresholds["normal"]
        and abnormal_alert_rate >= thresholds["abnormal"]
        and frozen_red_rate >= thresholds["frozen"]
        and noise_alert_rate >= thresholds["noise"]
        and backdoor_alert_rate >= thresholds["backdoor"]
        and finite_residual_stats
    )
    return {
        "reservoir": int(reservoir),
        "normal_false_alert_rate": normal_false_alert_rate,
        "abnormal_alert_rate": abnormal_alert_rate,
        "frozen_red_rate": frozen_red_rate,
        "noise_alert_rate": noise_alert_rate,
        "backdoor_alert_rate": backdoor_alert_rate,
        "finite_residual_stats": bool(finite_residual_stats),
        "pass_all": bool(pass_all),
    }


def _per_regime_rows(reservoir: int, counts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name in REGIME_ORDER:
        bucket = counts[name]
        frames = bucket["frames"]
        rows.append(
            {
                "reservoir": int(reservoir),
                "regime": name,
                "frames": int(frames),
                "green_rate": _rate(bucket["green"], frames),
                "amber_rate": _rate(bucket["amber"], frames),
                "red_rate": _rate(bucket["red"], frames),
                "alert_rate": _rate(bucket["alerts"], frames),
                "mean_residual_evidence": _rate(bucket["residual_sum"], frames),
                "mean_input_evidence": _rate(bucket["input_sum"], frames),
                "mean_geometry_evidence": _rate(bucket["geometry_sum"], frames),
                "mean_event_score": _rate(bucket["event_sum"], frames),
            }
        )
    return rows


def _finite_residual_row(row: dict[str, Any]) -> bool:
    try:
        ema = float(row.get("ema_err", float("nan")))
        sigma = float(row.get("sigma", float("nan")))
    except (TypeError, ValueError):
        return False
    return bool(math.isfinite(ema) and math.isfinite(sigma) and sigma > 0.0)


def _rate(numerator: float, denominator: float) -> float:
    denominator = finite_scalar(denominator)
    if denominator <= 0.0:
        return 0.0
    return finite_scalar(float(numerator) / denominator)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name, "")) for name in names})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return finite_scalar(value)
    return value


def _hash_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest(args: argparse.Namespace, run_config: dict[str, Any], config_hash: str, total_steps: int) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "git_dirty": bool(_git(["status", "--short"])),
        "python_version": sys.version,
        "platform": platform.platform(),
        "seed": run_config["seed"],
        "reservoir_grid": run_config["reservoirs"],
        "feature_count": run_config["features"],
        "warmup": run_config["warmup"],
        "frame_counts_per_regime": {"WARMUP": args.warmup, **run_config["regime_frames"]},
        "total_steps": total_steps,
        "config_hash": config_hash,
        "command_line_arguments": sys.argv,
    }


def _git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _write_drive_manifest(*, out_dir: Path, run_id: str) -> None:
    files = sorted(path for path in out_dir.iterdir() if path.is_file() and path.name != "drive_manifest.json")
    local_date = datetime.now().strftime("%Y-%m-%d")
    drive_root = _discover_drive_root()
    copied: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    drive_run_dir = "unknown"
    success = False
    reason = ""

    if drive_root is None:
        reason = "EIDOS_PROOF_DRIVE_DIR not set, EIDOS_ARTIFACT_ROOT not writable, and no mounted Colab Drive path found"
        skipped = [{"path": str(path), "reason": reason} for path in files]
    else:
        drive_run_path = drive_root / "Eidos_Brain_Proof_Phase" / local_date / run_id
        drive_run_path.mkdir(parents=True, exist_ok=True)
        drive_run_dir = str(drive_run_path)
        for path in files:
            dest = drive_run_path / path.name
            try:
                shutil.copy2(path, dest)
                copied.append({"path": str(path), "drive_path": str(dest), "sha256": _file_sha256(path)})
            except OSError as exc:
                skipped.append({"path": str(path), "reason": str(exc)})
        success = bool(copied) and not skipped
        reason = "copied" if success else "one or more files failed to copy"

    manifest = {
        "drive_copy_attempted": True,
        "drive_copy_success": success,
        "drive_root": str(drive_root) if drive_root is not None else "unknown",
        "drive_run_dir": drive_run_dir,
        "reason": reason,
        "files_considered": [str(path) for path in files],
        "files_copied": copied,
        "files_skipped": skipped,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(out_dir / "drive_manifest.json", manifest)
    if drive_root is not None and drive_run_dir != "unknown":
        try:
            shutil.copy2(out_dir / "drive_manifest.json", Path(drive_run_dir) / "drive_manifest.json")
        except OSError:
            pass


def _discover_drive_root() -> Path | None:
    for name in ("EIDOS_PROOF_DRIVE_DIR", "EIDOS_ARTIFACT_ROOT"):
        value = os.environ.get(name, "").strip()
        if value:
            path = Path(value)
            if _writable_dir(path):
                return path
    colab = Path("/content/drive/MyDrive")
    if colab.exists() and _writable_dir(colab):
        return colab
    return None


def _writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".eidos_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
