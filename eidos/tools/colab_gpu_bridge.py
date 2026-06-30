"""Colab GPU bridge for Eidos Brain/Sentinel proof runs.

This helper is orchestration only. It runs existing repo-root proof commands,
captures Colab/GPU/Drive context, and writes receipts. It does not change
reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, or
compression behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_proof_baseline as proof_helpers

DEFAULT_LABELED_FILE = Path(
    "artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
)
BRIDGE_FILES = (
    "colab_gpu_bridge_receipt.json",
    "colab_gpu_bridge_receipt.md",
    "colab_gpu_bridge_environment.txt",
    "colab_gpu_bridge_git_commit.txt",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def relpath(path: Path, root: Path = REPO_ROOT) -> str:
    return proof_helpers.relpath(path, root)


def command_text(parts: Sequence[str]) -> str:
    display_parts = ["python" if part == sys.executable else part for part in parts]
    return proof_helpers.command_text(display_parts)


def resolve_out_dir(out: Path) -> Path:
    return out if out.is_absolute() else REPO_ROOT / out


def flatten(values: Iterable[Any]) -> List[str]:
    flattened: List[str] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            flattened.extend(str(part) for part in value)
        else:
            flattened.append(str(value))
    return [item for item in flattened if item]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("tensor-smoke", "proof-baseline", "labeled-domain"),
        required=True,
        help="Existing Eidos proof command to orchestrate.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Repo-local artifact directory for this run.")
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=10000)
    parser.add_argument("--dataset", default="cicids_webattacks")
    parser.add_argument("--file", type=Path, default=DEFAULT_LABELED_FILE)
    parser.add_argument("--label-column", default="Label")
    parser.add_argument("--attack-labels", action="append", nargs="+", default=[])
    parser.add_argument("--normalize-non-benign-as", choices=("ATTACK",), default=None)
    parser.add_argument("--sample-mode", choices=("natural", "balanced", "transition"), default="natural")
    parser.add_argument(
        "--confirmation-mode",
        choices=("off", "low_noise", "balanced", "high_recall"),
        default="balanced",
    )
    parser.add_argument("--event-merge-gap", type=int, default=25)
    parser.add_argument("--calibration-enabled", action="store_true")
    parser.add_argument("--calibration-event-merge-gap", type=int, default=None)
    parser.add_argument("--calibration-benign-context-grace", type=int, default=0)
    parser.add_argument("--calibration-attack-window-guard", type=int, default=0)
    parser.add_argument("--calibration-min-confirmed-span", type=int, default=2)
    parser.add_argument("--calibration-min-evidence-count", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--mount-drive",
        action="store_true",
        help="In Colab, mount Google Drive before running the proof command.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail before the proof run if torch reports that CUDA is unavailable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write bridge receipts and the selected command without running the proof command.",
    )
    parser.add_argument(
        "--skip-drive-copy",
        action="store_true",
        help="Do not create or refresh drive_manifest.json from this bridge.",
    )
    return parser.parse_args(argv)


def build_child_command(args: argparse.Namespace) -> List[str]:
    out = relpath(resolve_out_dir(args.out))
    if args.mode == "tensor-smoke":
        return [sys.executable, "scripts/verify_colab_gpu_hotfix.py"]
    if args.mode == "proof-baseline":
        return [
            sys.executable,
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
    if args.mode == "labeled-domain":
        attack_labels = flatten(args.attack_labels)
        if not attack_labels and not args.normalize_non_benign_as:
            raise ValueError(
                "labeled-domain mode requires --attack-labels or --normalize-non-benign-as ATTACK"
            )
        cmd = [
            sys.executable,
            "tools/run_labeled_domain_proof.py",
            "--dataset",
            args.dataset,
            "--file",
            relpath(args.file if args.file.is_absolute() else REPO_ROOT / args.file),
            "--label-column",
            args.label_column,
            "--frames",
            str(args.frames),
            "--seed",
            str(args.seed),
            "--out",
            out,
            "--suite",
            args.suite,
            "--sample-mode",
            args.sample_mode,
            "--event-merge-gap",
            str(args.event_merge_gap),
            "--confirmation-mode",
            args.confirmation_mode,
        ]
        if args.normalize_non_benign_as:
            cmd.extend(["--normalize-non-benign-as", args.normalize_non_benign_as])
        for label in attack_labels:
            cmd.extend(["--attack-labels", label])
        if args.calibration_enabled:
            cmd.append("--calibration-enabled")
            if args.calibration_event_merge_gap is not None:
                cmd.extend(["--calibration-event-merge-gap", str(args.calibration_event_merge_gap)])
            cmd.extend(
                [
                    "--calibration-benign-context-grace",
                    str(args.calibration_benign_context_grace),
                    "--calibration-attack-window-guard",
                    str(args.calibration_attack_window_guard),
                    "--calibration-min-confirmed-span",
                    str(args.calibration_min_confirmed_span),
                    "--calibration-min-evidence-count",
                    str(args.calibration_min_evidence_count),
                ]
            )
        if args.max_rows is not None:
            cmd.extend(["--max-rows", str(args.max_rows)])
        return cmd
    raise ValueError(f"unsupported mode: {args.mode}")


def collect_device_receipt() -> Dict[str, Any]:
    receipt: Dict[str, Any] = {
        "captured_at_utc": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "cuda_available": False,
        "torch_available": False,
        "device_count": 0,
        "devices": [],
        "nvidia_smi": {"available": False},
    }
    try:
        import torch  # type: ignore

        receipt.update(
            {
                "torch_available": True,
                "torch_version": torch.__version__,
                "torch_cuda_version": getattr(torch.version, "cuda", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            }
        )
        if torch.cuda.is_available():
            receipt["devices"] = [
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "capability": list(torch.cuda.get_device_capability(idx)),
                }
                for idx in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        receipt["torch_error"] = str(exc)

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                cwd=str(REPO_ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
                check=False,
            )
            receipt["nvidia_smi"] = {
                "available": True,
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except Exception as exc:
            receipt["nvidia_smi"] = {"available": True, "error": str(exc)}
    return receipt


def maybe_mount_colab_drive(should_mount: bool) -> Dict[str, Any]:
    receipt: Dict[str, Any] = {
        "attempted": False,
        "success": False,
        "reason": "mount not requested",
        "drive_root_env": os.environ.get("EIDOS_PROOF_DRIVE_DIR") or os.environ.get("EIDOS_ARTIFACT_ROOT"),
    }
    if not should_mount:
        return receipt

    receipt["attempted"] = True
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
        for candidate in proof_helpers.colab_drive_candidates():
            if candidate.exists():
                os.environ.setdefault("EIDOS_PROOF_DRIVE_DIR", str(candidate))
                receipt.update(
                    {
                        "success": True,
                        "reason": f"mounted Colab Drive and selected {candidate}",
                        "drive_root_env": os.environ.get("EIDOS_PROOF_DRIVE_DIR"),
                    }
                )
                return receipt
        receipt["reason"] = "Colab Drive mount completed but no MyDrive path was found"
    except Exception as exc:
        receipt["reason"] = f"Colab Drive mount failed or is unavailable: {exc}"
    return receipt


def write_bridge_receipt(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    command: Sequence[str],
    status: str,
    device: Dict[str, Any],
    drive_mount: Dict[str, Any],
    child_result: Optional[subprocess.CompletedProcess[str]] = None,
    note: str = "",
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    git_info = proof_helpers.collect_git_info(REPO_ROOT)
    receipt: Dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "status": status,
        "note": note,
        "mode": args.mode,
        "dry_run": bool(args.dry_run),
        "require_cuda": bool(args.require_cuda),
        "repo_root": str(REPO_ROOT),
        "command": command_text(command),
        "argv": ["python" if item == sys.executable else item for item in command],
        "artifact_dir": relpath(out_dir),
        "core_behavior_changed": False,
        "core_behavior_note": (
            "Bridge orchestration only; existing proof runners own model execution. "
            "No reservoir, RLS, Sentinel threshold, anomaly policy, or compression behavior is changed."
        ),
        "device": device,
        "drive_mount": drive_mount,
        "environment": {
            "EIDOS_PROOF_DRIVE_DIR": os.environ.get("EIDOS_PROOF_DRIVE_DIR", ""),
            "EIDOS_ARTIFACT_ROOT": os.environ.get("EIDOS_ARTIFACT_ROOT", ""),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD", ""),
        },
        "git": git_info,
    }
    if child_result is not None:
        receipt["child"] = {
            "returncode": child_result.returncode,
            "stdout_tail": child_result.stdout[-4000:] if child_result.stdout else "",
            "stderr_tail": child_result.stderr[-4000:] if child_result.stderr else "",
        }

    proof_helpers.write_json(out_dir / "colab_gpu_bridge_receipt.json", receipt)
    write_receipt_md(out_dir / "colab_gpu_bridge_receipt.md", receipt)
    (out_dir / "colab_gpu_bridge_environment.txt").write_text(
        collect_bridge_environment(receipt),
        encoding="utf-8",
    )
    proof_helpers.write_git_commit(out_dir / "colab_gpu_bridge_git_commit.txt", git_info)
    return receipt


def collect_bridge_environment(receipt: Dict[str, Any]) -> str:
    device = receipt.get("device", {})
    lines = [
        f"generated_at_utc: {utc_now()}",
        f"python_version: {sys.version}",
        f"python_executable: {sys.executable}",
        f"platform: {platform.platform()}",
        f"os: {os.name}",
        f"machine: {platform.machine()}",
        f"processor: {platform.processor()}",
        f"current_working_directory: {Path.cwd()}",
        f"resolved_repo_root: {REPO_ROOT.resolve()}",
        "",
        "Relevant environment variables (secret-looking values redacted):",
    ]
    for key, value in proof_helpers.redacted_env_items().items():
        lines.append(f"{key}={value}")
    lines.extend(
        [
            "",
            "bridge_device_receipt:",
            f"torch_available: {device.get('torch_available')}",
            f"torch_version: {device.get('torch_version', 'unknown')}",
            f"cuda_available: {device.get('cuda_available')}",
            f"cuda_device_count: {device.get('device_count')}",
            "pip_freeze: skipped for bridge receipt; child proof runs write full environment.txt",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_receipt_md(path: Path, receipt: Dict[str, Any]) -> None:
    device = receipt.get("device", {})
    drive = receipt.get("drive_mount", {})
    child = receipt.get("child", {})
    lines = [
        "# Colab GPU Bridge Receipt",
        "",
        f"- Status: `{receipt.get('status')}`",
        f"- Mode: `{receipt.get('mode')}`",
        f"- Command: `{receipt.get('command')}`",
        f"- Artifact directory: `{receipt.get('artifact_dir')}`",
        f"- CUDA available: `{device.get('cuda_available')}`",
        f"- Device count: `{device.get('device_count')}`",
        f"- Drive mount attempted: `{drive.get('attempted')}`",
        f"- Drive mount status: `{drive.get('reason')}`",
        f"- Core behavior changed: `{receipt.get('core_behavior_changed')}`",
    ]
    if child:
        lines.append(f"- Child return code: `{child.get('returncode')}`")
    if receipt.get("note"):
        lines.extend(["", "## Note", "", str(receipt["note"])])
    lines.extend(["", "## Core Behavior", "", str(receipt.get("core_behavior_note", ""))])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def copy_bridge_files_to_drive(out_dir: Path, *, skip_drive_copy: bool) -> Dict[str, Any]:
    if skip_drive_copy:
        return {"attempted": False, "reason": "skip-drive-copy was requested"}

    selected_files = [out_dir / name for name in BRIDGE_FILES]
    existing_manifest_path = out_dir / "drive_manifest.json"
    if existing_manifest_path.exists():
        try:
            manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
            proof_helpers.copy_selected_to_drive(out_dir, manifest, selected_files)
            return {
                "attempted": True,
                "strategy": "copied bridge files to existing drive manifest destination",
                "drive_copy_success": manifest.get("drive_copy_success"),
                "drive_run_dir": manifest.get("drive_run_dir"),
                "reason": manifest.get("reason"),
            }
        except Exception as exc:
            return {"attempted": True, "reason": f"could not reuse existing drive_manifest.json: {exc}"}

    manifest = proof_helpers.mirror_to_drive(out_dir, out_dir.name, run_date())
    proof_helpers.write_json(existing_manifest_path, manifest)
    proof_helpers.copy_selected_to_drive(out_dir, manifest, [existing_manifest_path])
    return {
        "attempted": True,
        "strategy": "created bridge drive_manifest.json",
        "drive_copy_success": manifest.get("drive_copy_success"),
        "drive_run_dir": manifest.get("drive_run_dir"),
        "reason": manifest.get("reason"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = resolve_out_dir(args.out)

    try:
        command = build_child_command(args)
    except ValueError as exc:
        print(f"COLAB_GPU_BRIDGE_INVALID_ARGS {exc}", file=sys.stderr)
        return 2

    drive_mount = maybe_mount_colab_drive(args.mount_drive)
    device = collect_device_receipt()

    if args.require_cuda and not device.get("cuda_available"):
        write_bridge_receipt(
            out_dir,
            args=args,
            command=command,
            status="failed_preflight",
            device=device,
            drive_mount=drive_mount,
            note="CUDA was required, but torch did not report an available CUDA device.",
        )
        copy_bridge_files_to_drive(out_dir, skip_drive_copy=args.skip_drive_copy)
        print("COLAB_GPU_BRIDGE_FAILED cuda_unavailable")
        return 2

    if args.dry_run:
        write_bridge_receipt(
            out_dir,
            args=args,
            command=command,
            status="dry_run",
            device=device,
            drive_mount=drive_mount,
            note="Dry run only. The selected proof command was not executed.",
        )
        copy_bridge_files_to_drive(out_dir, skip_drive_copy=args.skip_drive_copy)
        print(f"COLAB_GPU_BRIDGE_DRY_RUN command={command_text(command)}")
        return 0

    preflight_receipt = write_bridge_receipt(
        out_dir,
        args=args,
        command=command,
        status="running",
        device=device,
        drive_mount=drive_mount,
    )
    print(f"COLAB_GPU_BRIDGE_RUNNING command={preflight_receipt['command']}")

    env = os.environ.copy()
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    status = "passed" if result.returncode == 0 else "failed"
    write_bridge_receipt(
        out_dir,
        args=args,
        command=command,
        status=status,
        device=collect_device_receipt(),
        drive_mount=drive_mount,
        child_result=result,
    )
    copy_bridge_files_to_drive(out_dir, skip_drive_copy=args.skip_drive_copy)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    print(f"COLAB_GPU_BRIDGE_DONE status={status} returncode={result.returncode}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
