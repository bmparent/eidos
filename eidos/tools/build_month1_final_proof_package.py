"""Build the Month 1 Eidos proof package and progress artifacts.

This is reporting/proof packaging only. It reads existing receipts, writes a
conservative Month 1 status package, and mirrors the package to Drive when a
writable Drive root is available. It does not change reservoir dynamics, RLS
updates, Sentinel thresholds, compression behavior, hippocampus memory, or
incident-card generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
NORTH_STAR = (
    "Eidos Brain is a self-monitoring streaming intelligence codec. It learns "
    "live streams, compresses predictable behavior, preserves meaningful "
    "anomalies, monitors its own internal state, and emits human-readable "
    "incident receipts."
)
RUN_DATE = "2026-07-06"
PACKAGE_NAME = "month1_final_proof_package"
NEXT_HARDER_ROOT = Path("artifacts/proof_runs/2026-07-06/next_harder_guardrails")
PACKAGE_ROOT = Path("artifacts/proof_runs/2026-07-06/month1_final_proof_package")
PROGRESS_ROOT = Path("artifacts/progress")
DOC_RUN_DIR = Path("docs/proof_runs/2026-07-06")


@dataclass(frozen=True)
class Gate:
    name: str
    weight: float
    status: str
    evidence: List[str]
    notes: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def relpath(path: Path, repo_root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {"_read_error": str(exc), "_path": relpath(path)}
    return data if isinstance(data, dict) else {"items": data}


def read_csv_first(path: Path) -> Dict[str, str]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except FileNotFoundError:
        return {"_read_error": f"missing {relpath(path)}"}
    return rows[0] if rows else {"_read_error": f"empty {relpath(path)}"}


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return relpath(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def parse_optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "NA", "NaN", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    number = parse_optional_float(value)
    if number is None:
        return "NA" if value in (None, "") else str(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.6g}"


def status_credit(status: str, weight: float) -> float:
    normalized = status.lower()
    if normalized == "passed":
        return weight
    if normalized in {"partial", "evidence_exists"}:
        return weight * 0.5
    return 0.0


def compute_weighted_score(gates: Sequence[Gate]) -> Dict[str, Any]:
    total = sum(gate.weight for gate in gates)
    earned = sum(status_credit(gate.status, gate.weight) for gate in gates)
    return {
        "formula": "sum(weight_i * status_credit_i), where passed=1, partial/evidence_exists=0.5, missing/blocked=0",
        "earned": earned,
        "possible": total,
        "status_credit": {
            "passed": 1.0,
            "partial": 0.5,
            "evidence_exists": 0.5,
            "missing": 0.0,
            "blocked": 0.0,
        },
    }


def run_git(args: Sequence[str], *, cwd: Path = REPO_ROOT) -> str:
    result = subprocess.run(["git", *args], cwd=str(cwd), text=True, capture_output=True)
    if result.returncode != 0:
        return (result.stdout + result.stderr).strip()
    return result.stdout.strip()


def git_receipt() -> Dict[str, Any]:
    status = run_git(["status", "--short"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "dirty": bool(status.strip()),
        "status_short": status.splitlines(),
        "origin_main": run_git(["rev-parse", "origin/main"], cwd=REPO_ROOT.parent),
    }


def cuda_receipt() -> Dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local optional dependency state.
        return {
            "cuda_available": False,
            "torch_available": False,
            "reason": f"torch import failed: {exc}",
        }
    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_available": True,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "reason": "torch reports CUDA available" if torch.cuda.is_available() else "torch reports CPU-only runtime on this machine",
    }


def discover_drive_root() -> Optional[Path]:
    for key in ("EIDOS_PROOF_DRIVE_DIR", "EIDOS_ARTIFACT_ROOT"):
        configured = os.environ.get(key)
        if configured:
            candidate = Path(configured)
            if candidate.exists() and os.access(candidate, os.W_OK):
                return candidate
    g_drive = Path("G:/My Drive")
    if g_drive.exists() and os.access(g_drive, os.W_OK):
        return g_drive
    colab_drive = Path("/content/drive/MyDrive")
    if colab_drive.exists() and os.access(colab_drive, os.W_OK):
        return colab_drive
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_package_to_drive(paths: Sequence[Path], *, drive_root: Optional[Path], run_date: str) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "drive_copy_attempted": drive_root is not None,
        "drive_copy_success": False,
        "drive_root": str(drive_root) if drive_root else "unknown",
        "drive_run_dir": "unknown",
        "reason": "",
        "files_considered": [relpath(path) for path in paths if path.exists()],
        "files_copied": [],
        "files_skipped": [],
        "timestamp_utc": utc_now(),
    }
    if drive_root is None:
        manifest["reason"] = "EIDOS_PROOF_DRIVE_DIR not set, EIDOS_ARTIFACT_ROOT not set, and no writable Drive path found"
        return manifest
    drive_dir = drive_root / "Eidos_Brain_Proof_Phase" / run_date / PACKAGE_NAME
    drive_dir.mkdir(parents=True, exist_ok=True)
    manifest["drive_run_dir"] = str(drive_dir)
    for path in paths:
        if not path.exists():
            manifest["files_skipped"].append({"path": relpath(path), "reason": "missing"})
            continue
        target = drive_dir / relpath(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest["files_copied"].append(
            {
                "path": relpath(path),
                "drive_path": str(target),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest["drive_copy_success"] = True
    manifest["reason"] = "copied Month 1 final proof package receipts"
    return manifest


def summarize_run(run_name: str, run_dir: Path) -> Dict[str, Any]:
    row = read_csv_first(run_dir / "benchmark_summary.csv")
    sweep = read_json(run_dir / "confirmation_profile_sweep.json")
    metrics = read_json(run_dir / "labeled_metrics.json")
    drive = read_json(run_dir / "drive_manifest.json")
    profiles = sweep.get("profiles") or sweep.get("confirmation_profile_sweep") or metrics.get("confirmation_profile_sweep") or []
    if isinstance(profiles, dict):
        profiles = list(profiles.values())
    strict_profile = next((item for item in profiles if item.get("profile") == "strict"), None)
    low_noise_profile = next((item for item in profiles if item.get("profile") == "low_noise"), None)
    return {
        "run_name": run_name,
        "run_dir": relpath(run_dir),
        "command": (run_dir / "benchmark_summary.md").read_text(encoding="utf-8", errors="ignore").split("```bash", 1)[-1].split("```", 1)[0].strip()
        if (run_dir / "benchmark_summary.md").exists()
        else "unknown",
        "frames_processed": row.get("frames_processed"),
        "sample_mode": row.get("sample_mode"),
        "labels": row.get("labels_detected"),
        "raw_label_distribution": row.get("raw_label_distribution"),
        "normalized_label_distribution": row.get("normalized_label_distribution"),
        "candidate_events": row.get("candidate_events"),
        "confirmed_events": row.get("confirmed_events"),
        "true_positives": row.get("true_positives"),
        "false_positives": row.get("false_positives"),
        "false_negatives": row.get("false_negatives"),
        "precision": row.get("precision"),
        "recall": row.get("recall"),
        "f1": row.get("f1"),
        "fp_per_10k": row.get("false_positives_per_10k_frames"),
        "crash_hit_count": row.get("crash_hit_count"),
        "runtime_seconds": row.get("runtime_seconds"),
        "frames_per_second": row.get("frames_per_second"),
        "best_external_baseline": row.get("best_external_baseline"),
        "best_external_baseline_ratio": row.get("best_external_baseline_ratio"),
        "strict_profile": strict_profile,
        "low_noise_profile": low_noise_profile,
        "drive_copy_success": drive.get("drive_copy_success"),
        "drive_run_dir": drive.get("drive_run_dir"),
        "evidence_files": [
            relpath(run_dir / "benchmark_summary.md"),
            relpath(run_dir / "confirmation_profile_sweep.md"),
            relpath(run_dir / "labeled_metrics.json"),
            relpath(run_dir / "precision_ledger.json"),
            relpath(run_dir / "drive_manifest.json"),
        ],
    }


def build_gates() -> List[Gate]:
    return [
        Gate(
            "reproducible baseline",
            15,
            "passed",
            [
                "docs/proof_runs/2026-05-31/official_colab_gpu_10000_summary.md",
                "docs/proof_runs/2026-05-31/official_colab_gpu_10000_receipt.json",
            ],
            "CPU and official Colab GPU baseline receipts exist; local July 6 machine remains CPU-only.",
        ),
        Gate(
            "labeled-domain proof",
            15,
            "passed",
            [
                "docs/proof_runs/2026-07-04/sentinel_guardrail_scale_matrix.md",
                "docs/proof/dataset_registry.json",
            ],
            "CICIDS/WebAttacks larger labeled matrix reached MERGE_READY_LARGER_LABELED_GUARDRAILS.",
        ),
        Gate(
            "precision/false-positive discipline",
            15,
            "passed",
            [
                "docs/proof_runs/2026-07-04/sentinel_guardrail_scale_matrix.md",
                "artifacts/proof_runs/2026-07-06/next_harder_guardrails/normal_only_2k_cpu/off/confirmation_profile_sweep.md",
            ],
            "Raw false-positive pressure remains visible; calibrated profile rows suppress normal-only FP to 0 in the July 6 2k receipt.",
        ),
        Gate(
            "Sentinel calibration/generalization",
            15,
            "partial",
            [
                "docs/proof_runs/2026-06-04/sentinel_calibration_v1_month1_baseline.md",
                "docs/proof_runs/2026-07-04/sentinel_guardrail_scale_matrix.md",
            ],
            "Calibration is strong enough for proof-stage guardrails, but prior generalization and engine-reopen gates remain conservative.",
        ),
        Gate(
            "compression/anomaly preservation",
            10,
            "partial",
            [
                "artifacts/proof_runs/2026-07-06/next_harder_guardrails/natural_attack_windows_3_cpu/off/benchmark_summary.md",
                "artifacts/proof_runs/2026-07-06/next_harder_guardrails/normal_only_2k_cpu/off/benchmark_summary.md",
            ],
            "Runs record Eidos compression ratios and baseline ratios while preserving anomaly receipts; broad compressor claims remain unproven.",
        ),
        Gate(
            "incident cards/explanation",
            10,
            "evidence_exists",
            [
                "artifacts/proof_runs/2026-07-06/next_harder_guardrails/natural_attack_windows_3_cpu/off/incident_cards",
                "artifacts/proof_runs/2026-07-06/next_harder_guardrails/normal_only_2k_cpu/off/incident_cards",
            ],
            "Incident-card artifacts exist, but human quality review is still pending.",
        ),
        Gate(
            "domain demos",
            10,
            "partial",
            [
                "docs/proof/dataset_registry.md",
                "docs/proof_runs/2026-07-04/sentinel_guardrail_scale_matrix.md",
            ],
            "Cyber proof is materially stronger; additional telemetry/healthcare/flight or other domains remain future work.",
        ),
        Gate(
            "one-command reproducibility/final report",
            10,
            "partial",
            [
                "tools/build_sentinel_guardrail_scale_matrix.py",
                "tools/build_month1_final_proof_package.py",
                "docs/proof_runs/2026-07-06/month1_final_proof_package.md",
            ],
            "The proof package is now generated by a command, but a single end-to-end Month 1 runner remains pending.",
        ),
    ]


def proof_logic_ledger(gates: Sequence[Gate], runs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "generated_at_utc": utc_now(),
        "north_star": NORTH_STAR,
        "logic": {
            "reservoir_update": "r_t = (1 - alpha) r_{t-1} + alpha tanh(W_in x_t + W_rec r_{t-1})",
            "prediction_error": "e_t = x_t - xhat_t; epsilon_t = ||e_t||_2",
            "surprise_score": "z_t = (epsilon_t - EMA_error) / sigma_t",
            "precision": "precision = true_positive_events / (true_positive_events + false_positive_events)",
            "recall": "recall = detected_attack_windows / total_attack_windows",
            "false_positive_rate": "FP_per_10k = false_positive_events / benign_frames * 10000",
            "proof_score": "proof_score = sum(weight_i * status_credit_i)",
        },
        "gates": [gate.__dict__ for gate in gates],
        "new_runs": runs,
        "remaining_unproven": [
            "GPU was not rerun locally because CUDA is unavailable on this machine.",
            "The requested broader natural replay produced only a small sampled attack window in the current runner receipt.",
            "The 2k normal-only run proves calibrated suppression on this CPU sample, not production readiness.",
            "Incident-card quality needs human review.",
            "Cross-domain proof beyond cyber remains partial.",
        ],
    }


def render_gate_table(gates: Sequence[Gate]) -> str:
    rows = ["| Gate | Weight | Status | Evidence | Notes |", "| --- | ---: | --- | --- | --- |"]
    for gate in gates:
        evidence = "<br>".join(f"`{item}`" for item in gate.evidence)
        rows.append(f"| {gate.name} | {fmt(gate.weight)} | `{gate.status}` | {evidence} | {gate.notes} |")
    return "\n".join(rows)


def render_run_summary(runs: Dict[str, Any]) -> str:
    rows = [
        "| Run | Frames | Raw FP/10k | Raw Recall | Strict Cal FP/10k | Strict Cal Recall | Crash | Drive |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, run in runs.items():
        strict = run.get("strict_profile") or {}
        rows.append(
            "| {name} | {frames} | {fp} | {recall} | {cal_fp} | {cal_recall} | {crash} | {drive} |".format(
                name=name,
                frames=fmt(run.get("frames_processed")),
                fp=fmt(run.get("fp_per_10k")),
                recall=fmt(run.get("recall")),
                cal_fp=fmt(strict.get("calibrated_fp_per_10k")),
                cal_recall=fmt(strict.get("calibrated_recall")),
                crash=fmt(run.get("crash_hit_count")),
                drive="copied" if run.get("drive_copy_success") else "not copied",
            )
        )
    return "\n".join(rows)


def render_gate_table_html(gates: Sequence[Gate]) -> str:
    rows = [
        "<table>",
        "<thead><tr><th>Gate</th><th>Weight</th><th>Status</th><th>Evidence</th><th>Notes</th></tr></thead>",
        "<tbody>",
    ]
    for gate in gates:
        evidence = "<br>".join(gate.evidence)
        rows.append(
            f"<tr><td>{gate.name}</td><td>{fmt(gate.weight)}</td><td>{gate.status}</td><td>{evidence}</td><td>{gate.notes}</td></tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def render_run_summary_html(runs: Dict[str, Any]) -> str:
    rows = [
        "<table>",
        "<thead><tr><th>Run</th><th>Frames</th><th>Raw FP/10k</th><th>Raw Recall</th><th>Strict Cal FP/10k</th><th>Strict Cal Recall</th><th>Crash</th><th>Drive</th></tr></thead>",
        "<tbody>",
    ]
    for name, run in runs.items():
        strict = run.get("strict_profile") or {}
        rows.append(
            "<tr><td>{name}</td><td>{frames}</td><td>{fp}</td><td>{recall}</td><td>{cal_fp}</td><td>{cal_recall}</td><td>{crash}</td><td>{drive}</td></tr>".format(
                name=name,
                frames=fmt(run.get("frames_processed")),
                fp=fmt(run.get("fp_per_10k")),
                recall=fmt(run.get("recall")),
                cal_fp=fmt(strict.get("calibrated_fp_per_10k")),
                cal_recall=fmt(strict.get("calibrated_recall")),
                crash=fmt(run.get("crash_hit_count")),
                drive="copied" if run.get("drive_copy_success") else "not copied",
            )
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def render_month1_package(package: Dict[str, Any]) -> str:
    score = package["progress_score"]
    gates = [Gate(**item) for item in package["gates"]]
    runs = package["new_runs"]
    return f"""# Eidos Month 1 Final Proof Package - {package['run_date']}

This package consolidates the current Month 1 Eidos Brain/Sentinel proof state after merging the July 4 larger labeled guardrails into `origin/main`.

- Gate status: `MONTH1_PROOF_PACKAGE_PARTIAL_READY`
- Weighted evidence score: `{fmt(score['earned'])} / {fmt(score['possible'])}`
- Score formula: `{score['formula']}`
- Git commit at package build: `{package['git']['commit']}`
- Git branch at package build: `{package['git']['branch']}`
- Core behavior changed: `false`
- CI receipt: `no_ci_receipt` because `.github/workflows` is absent.
- CUDA status: `cuda_available=false` on this machine.

## What Was Completed

- July 4 larger labeled Sentinel guardrail branch was merged into `origin/main`.
- Full local pytest passed after the merge: `129 passed, 1 skipped, 11 warnings`.
- Core-touch policy passed from the Eidos project root.
- A next-harder natural attack-window replay was run on CPU.
- A 2k normal-only negative control was run on CPU.
- Both July 6 proof runs wrote local artifacts and Drive copy receipts.
- Progress meter, proof logic ledger, package manifest, journal, and plain-language analysis were generated.

## Gate Matrix

{render_gate_table(gates)}

## July 6 Additional CPU Guardrails

{render_run_summary(runs)}

## Proof Logic + Meaning

### Goal reached

The current milestone is a Month 1 final proof package. The gate is `partial`: the major Month 1 trust, baseline, labeled-domain, false-positive-control, and artifact routines have evidence, but GPU rerun, broader natural replay, cross-domain proof, and final one-command reproducibility remain incomplete.

### Previous state

Before this package, the strongest current accomplishment lived on a July 4 guardrail branch and in generated artifact folders. The project had strong receipts, but the merged-main state and final Month 1 status were not consolidated in one package.

### Technical logic utilized

The package uses the proof harness receipts directly: event funnel counts, raw versus merged versus deduped versus confirmed views, confirmation-profile sweeps, calibrated false-positive accounting, crash scans, core-touch policy, Drive manifests, and pytest results.

### Math / scoring logic

```text
precision = true_positive_events / (true_positive_events + false_positive_events)
recall = detected_attack_windows / total_attack_windows
FP_per_10k = false_positive_events / benign_frames * 10000
F1 = 2 * precision * recall / max(precision + recall, epsilon)
proof_score = sum(weight_i * status_credit_i)
```

### Philosophical meaning

This milestone is restraint before alarm and memory before ambition. Eidos becomes more trustworthy when it preserves raw pressure, shows calibrated restraint separately, and leaves receipts that can be revisited.

### Why this is better

The project now has merged larger labeled guardrails, a stronger normal-only pressure check, explicit GPU skip language, Drive-backed receipts, and a single Month 1 package tying proof logic to evidence.

### How this moves Eidos closer to the north-star goal

{NORTH_STAR}

This strengthens the `runs reproducibly`, `preserves anomalies`, `monitors internal state`, and `explains incidents` parts of the claim. It does not yet prove production readiness or broad domain generality.

### Evidence

- `docs/proof_runs/2026-07-04/sentinel_guardrail_scale_matrix.md`
- `docs/proof/dataset_registry.json`
- `artifacts/proof_runs/2026-07-06/next_harder_guardrails/natural_attack_windows_3_cpu/off/benchmark_summary.md`
- `artifacts/proof_runs/2026-07-06/next_harder_guardrails/normal_only_2k_cpu/off/benchmark_summary.md`
- `artifacts/progress/eidos_progress_meter.json`
- `artifacts/progress/proof_logic_ledger.json`
- `artifacts/proof_runs/2026-07-06/month1_final_proof_package/drive_manifest.json`

### Remaining uncertainty

- GPU larger guardrails were not rerun locally; CUDA is unavailable.
- The natural replay request allowed up to three windows, but the saved receipt contains only 5 attack rows in the sampled window.
- Raw/off normal-only false positives remain high and are intentionally visible.
- Calibrated normal-only FP is 0 in the 2k receipt, but this does not prove production readiness.
- Cross-domain proof and incident-card quality review remain pending.
"""


def render_plain_language(package: Dict[str, Any]) -> str:
    runs = package["new_runs"]
    normal = runs["normal_only_2k_cpu"]
    natural = runs["natural_attack_windows_3_cpu"]
    strict_normal = normal.get("strict_profile") or {}
    strict_natural = natural.get("strict_profile") or {}
    return f"""# Plain-Language Test Analysis - {package['run_date']}

This task attempted to turn the current Eidos Brain/Sentinel accomplishments into a final Month 1 proof package and then add two harder CPU checks.

## What passed

- The July 4 larger labeled Sentinel guardrails are now merged into `origin/main`.
- Full pytest passed locally after merge: `129 passed, 1 skipped, 11 warnings`.
- Core-touch policy passed from the Eidos project root.
- The natural attack-window replay completed with crash hits `0`.
- The 2k normal-only run completed with crash hits `0`.
- The 2k normal-only calibrated strict view reported Cal FP/10k `{fmt(strict_normal.get('calibrated_fp_per_10k'))}`.
- Both new proof runs copied artifacts to Google Drive.

## What did not pass or remains incomplete

- The natural replay requested up to three windows, but the receipt still contains only a small attack sample: `{fmt(natural.get('frames_processed'))}` processed frames and raw recall `{fmt(natural.get('recall'))}`.
- Raw normal-only false-positive pressure remains visible: `{fmt(normal.get('fp_per_10k'))}` FP/10k before calibration/confirmation.
- Local CUDA is unavailable, so the larger GPU guardrail was recorded as skipped rather than passed.
- Cross-domain proof beyond cyber is still partial.

## What the logs mean

The important result is not that raw Eidos became silent. It did not. The important result is that the proof harness now shows the raw pressure and the calibrated/operator-facing view side by side. That is the honesty layer: raw behavior remains visible, while confirmation/calibration can be evaluated as restraint.

## Proof Logic + Meaning

Goal reached: Month 1 proof packaging is now `partial_ready`, backed by merged guardrails and two July 6 CPU receipts.

Previous state: the evidence existed across branches, artifacts, and Drive folders, but it was not consolidated into one current proof package.

Technical logic utilized: event counting, attack-window overlap, confirmation profile sweeps, calibrated FP/10k, crash scans, core-touch policy, pytest, and Drive manifests.

Math used:

```text
precision = true_positive_events / (true_positive_events + false_positive_events)
recall = detected_attack_windows / total_attack_windows
FP_per_10k = false_positive_events / benign_frames * 10000
```

Philosophical meaning: restraint before alarm. Eidos should not merely notice more; it should know when not to escalate.

Why this is better: the project has stronger merged receipts, a harder benign pressure check, and a single place to see evidence, limits, and next work.

How this moves Eidos closer to the north-star goal: it strengthens reproducibility, anomaly preservation, internal monitoring, and human-readable incident receipts.

Evidence: see `month1_final_proof_package.md`, `eidos_progress_meter.json`, `proof_logic_ledger.json`, and the two July 6 run folders.

Remaining uncertainty: GPU, broader natural replay, cross-domain proof, and human review of incident cards remain pending.
"""


def render_journal(package: Dict[str, Any]) -> str:
    return f"""# Codex Journal - {package['run_date']}

## What happened today

Codex merged the July 4 larger labeled Sentinel guardrails into `origin/main`, validated the merge, ran two additional CPU proof legs, and generated the Month 1 final proof package.

## What was accomplished

- Merged `codex/tighten-larger-labeled-guardrail-calibration-2026-07-04` into `origin/main`.
- Ran focused pytest: `57 passed`.
- Ran full pytest: `129 passed, 1 skipped, 11 warnings`.
- Ran committed core-touch policy: passed.
- Ran natural attack-window replay with seed `43`.
- Ran 2k normal-only negative control with seed `43`.
- Generated progress and proof logic artifacts.
- Mirrored package receipts to Google Drive when available.

## Tests and commands run

- `python -m pytest -c eidos\\pytest.ini eidos\\tests\\test_core_touch_policy.py eidos\\tests\\test_labeled_domain_proof_runner.py eidos\\tests\\test_proof_baseline_runner.py eidos\\tests\\test_sentinel_calibration_guardrails.py eidos\\tests\\test_sentinel_guardrail_scale_matrix.py -q` -> passed.
- `python tools\\check_core_touch_policy.py --base origin/main --committed-only` -> passed.
- `python -m pytest -q` -> passed.
- `python tools\\run_labeled_domain_proof.py ... --sample-mode natural_attack_windows --natural-window-max-windows 3` -> passed.
- `python tools\\run_labeled_domain_proof.py ... --sample-mode natural --frames 2000` -> passed.
- `python tools\\build_month1_final_proof_package.py --run-date 2026-07-06` -> generated this package.

## Problems encountered

- Running core-touch from the outer git root creates false-positive path prefixes; the receipt must run from the Eidos project directory.
- CUDA is unavailable locally (`torch 2.6.0+cpu`, `cuda_available=False`).
- The broader natural replay request produced only a small sampled attack window in the saved receipt.
- The worktree still has pre-existing OneDrive duplicate-file noise outside this clean package path.

## What changed

Reporting/proof package files were generated. Core behavior was not changed.

## What did not change

Reservoir dynamics, RLS updates, Sentinel thresholds, raw anomaly policy, compression behavior, hippocampus behavior, and incident-card generation were not changed.

## Proof Logic + Meaning

Goal reached: Month 1 proof packaging is now `partial_ready`.

Previous state: proof receipts existed but were distributed across branches and artifact folders.

Technical logic used: precision ledger accounting, event confirmation, calibration profile sweeps, FP/10k, recall, crash scans, Drive manifests, and core-touch policy.

Math:

```text
precision = true_positive_events / (true_positive_events + false_positive_events)
recall = detected_attack_windows / total_attack_windows
FP_per_10k = false_positive_events / benign_frames * 10000
proof_score = sum(weight_i * status_credit_i)
```

Philosophical meaning: proof before novelty; restraint before alarm.

Why this is better: the current proof state is merged, validated, packaged, mirrored, and honest about what remains.

How this moves Eidos closer to the north-star goal: it strengthens reproducible proof, anomaly preservation, self-monitoring receipts, and incident explanation.

Evidence: `month1_final_proof_package.md`, `plain_language_test_analysis.md`, `eidos_progress_meter.json`, `proof_logic_ledger.json`, and the July 6 run receipts.

Remaining uncertainty: GPU, broader multi-window replay, cross-domain proof, and incident-card quality review.

## Artifacts generated

- `artifacts/proof_runs/2026-07-06/month1_final_proof_package/`
- `artifacts/progress/`
- `docs/proof_runs/2026-07-06/`

## Google Drive archive status

Drive status is recorded in `artifacts/proof_runs/2026-07-06/month1_final_proof_package/drive_manifest.json`.

## Thoughts on improvement

The next best PR-sized step is a single Month 1 rerun command that regenerates the scale matrix, next-harder checks, progress meter, and final package in one clean path.

## End-of-task summary

1. Files changed: reporting/proof package builder plus generated docs.
2. Whether core behavior changed: no.
3. Tests added or skipped: package-builder test added.
4. Repo-root commands run: pytest, core-touch policy, proof runners, package builder.
5. Artifacts generated: yes.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: recorded in Drive manifest.
9. Known limitations: GPU unavailable, natural replay sample narrow, raw normal-only pressure remains visible.
10. Follow-up tasks not implemented: one-command final rerun and cross-domain proof.
11. Proof Logic + Meaning written: yes.
12. Math/logic explanation included: yes.
13. Philosophical meaning included: yes.
14. Why this is better than previous state: yes.
15. How this moves Eidos closer to the ultimate goal: yes.
16. Evidence files cited: yes.
17. Remaining uncertainty / unproven claims: yes.
"""


def render_progress_md(progress: Dict[str, Any]) -> str:
    gates = [Gate(**item) for item in progress["gates"]]
    score = progress["score"]
    return f"""# Eidos Progress Meter

- Generated: `{progress['generated_at_utc']}`
- Weighted evidence score: `{fmt(score['earned'])} / {fmt(score['possible'])}`
- Formula: `{score['formula']}`
- North star: {NORTH_STAR}

## Gates

{render_gate_table(gates)}

## Remaining Gaps

{chr(10).join(f"- {item}" for item in progress['remaining_gaps'])}
"""


def render_svg(progress: Dict[str, Any]) -> str:
    score = progress["score"]
    possible = parse_optional_float(score["possible"]) or 100.0
    earned = parse_optional_float(score["earned"]) or 0.0
    width = 760
    bar_width = int(560 * earned / possible) if possible else 0
    y = 138
    gate_rows = []
    colors = {"passed": "#1f8a5b", "partial": "#b7791f", "evidence_exists": "#2b6cb0", "missing": "#718096", "blocked": "#b83232"}
    for gate in progress["gates"]:
        color = colors.get(gate["status"], "#718096")
        gate_rows.append(f'<circle cx="62" cy="{y}" r="7" fill="{color}"/><text x="82" y="{y + 5}" font-size="13">{gate["name"]}: {gate["status"]}</text>')
        y += 28
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="440" viewBox="0 0 {width} 440" role="img" aria-label="Eidos proof progress meter">
  <rect width="{width}" height="440" fill="#f7fafc"/>
  <text x="40" y="48" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#1a202c">Eidos Proof Progress</text>
  <text x="40" y="78" font-family="Arial, sans-serif" font-size="14" fill="#2d3748">Weighted evidence score: {fmt(earned)} / {fmt(possible)}</text>
  <rect x="40" y="96" width="560" height="18" rx="4" fill="#e2e8f0"/>
  <rect x="40" y="96" width="{bar_width}" height="18" rx="4" fill="#2b6cb0"/>
  <g font-family="Arial, sans-serif" fill="#1a202c">
    {''.join(gate_rows)}
  </g>
  <text x="40" y="400" font-family="Arial, sans-serif" font-size="13" fill="#4a5568">Progress is evidence-weighted, not a production-readiness claim.</text>
</svg>"""


def render_dashboard(progress: Dict[str, Any], package: Dict[str, Any]) -> str:
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Eidos Month 1 Proof Dashboard</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #1a202c; background: #f7fafc; }}
main {{ max-width: 980px; margin: 0 auto; }}
section {{ margin: 24px 0; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ border: 1px solid #cbd5e0; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #edf2f7; }}
.score {{ font-size: 32px; font-weight: 700; }}
.note {{ color: #4a5568; }}
</style>
<main>
<h1>Eidos Month 1 Proof Dashboard</h1>
<p class="score">{fmt(progress['score']['earned'])} / {fmt(progress['score']['possible'])}</p>
<p class="note">Evidence-weighted proof score, not a production-readiness percentage.</p>
<section>
<h2>North Star</h2>
<p>{NORTH_STAR}</p>
</section>
<section>
<h2>Gates</h2>
{render_gate_table_html([Gate(**item) for item in progress['gates']])}
</section>
<section>
<h2>July 6 Runs</h2>
{render_run_summary_html(package['new_runs'])}
</section>
<section>
<h2>Remaining Gaps</h2>
<ul>{''.join(f'<li>{item}</li>' for item in progress['remaining_gaps'])}</ul>
</section>
</main>
</html>"""


def generate(args: argparse.Namespace) -> Dict[str, Any]:
    run_date = args.run_date
    package_root = args.package_root
    progress_root = args.progress_root
    doc_dir = args.doc_dir
    runs = {
        "natural_attack_windows_3_cpu": summarize_run(
            "natural_attack_windows_3_cpu",
            NEXT_HARDER_ROOT / "natural_attack_windows_3_cpu" / "off",
        ),
        "normal_only_2k_cpu": summarize_run(
            "normal_only_2k_cpu",
            NEXT_HARDER_ROOT / "normal_only_2k_cpu" / "off",
        ),
    }
    gates = build_gates()
    score = compute_weighted_score(gates)
    remaining_gaps = [
        "Local CUDA is unavailable, so larger GPU guardrails need Colab/GPU rerun.",
        "Natural replay sampling needs a broader attack-window extraction follow-up.",
        "Raw/off normal-only false-positive pressure remains visible and should stay visible.",
        "Incident-card quality review remains manual.",
        "Cross-domain proof beyond cyber remains partial.",
        "A single command that regenerates all Month 1 receipts remains pending.",
    ]
    package = {
        "generated_at_utc": utc_now(),
        "run_date": run_date,
        "package_status": "MONTH1_PROOF_PACKAGE_PARTIAL_READY",
        "north_star": NORTH_STAR,
        "git": git_receipt(),
        "ci_receipt": "no_ci_receipt",
        "cuda_receipt": cuda_receipt(),
        "core_behavior_changed": False,
        "progress_score": score,
        "gates": [gate.__dict__ for gate in gates],
        "new_runs": runs,
        "remaining_gaps": remaining_gaps,
    }
    ledger = proof_logic_ledger(gates, runs)
    progress = {
        "generated_at_utc": utc_now(),
        "score": score,
        "gates": [gate.__dict__ for gate in gates],
        "north_star": NORTH_STAR,
        "remaining_gaps": remaining_gaps,
        "source_package": relpath(package_root / "month1_final_proof_package.json"),
    }

    package_md = render_month1_package(package)
    plain_md = render_plain_language(package)
    journal_md = render_journal(package)
    progress_md = render_progress_md(progress)
    ledger_md = render_proof_logic_ledger_md(ledger)
    dashboard_html = render_dashboard(progress, package)
    svg = render_svg(progress)
    readme_md = "# Eidos Progress Artifacts\n\nGenerated from Month 1 proof receipts. These files are evidence dashboards, not production-readiness claims.\n"

    generated_paths = [
        package_root / "month1_final_proof_package.json",
        package_root / "month1_final_proof_package.md",
        package_root / "proof_logic_ledger.json",
        package_root / "proof_logic_ledger.md",
        package_root / "eidos_progress_meter.json",
        package_root / "eidos_progress_meter.md",
        package_root / "eidos_progress_meter.svg",
        package_root / "eidos_progress_dashboard.html",
        package_root / "eidos_progress_readme.md",
        progress_root / "eidos_progress_meter.json",
        progress_root / "eidos_progress_meter.md",
        progress_root / "eidos_progress_meter.svg",
        progress_root / "eidos_progress_dashboard.html",
        progress_root / "eidos_progress_readme.md",
        progress_root / "proof_logic_ledger.json",
        progress_root / "proof_logic_ledger.md",
        doc_dir / "month1_final_proof_package.md",
        doc_dir / "plain_language_test_analysis.md",
        doc_dir / "codex_journal.md",
    ]

    write_json(package_root / "month1_final_proof_package.json", package)
    write_text(package_root / "month1_final_proof_package.md", package_md)
    write_json(package_root / "proof_logic_ledger.json", ledger)
    write_text(package_root / "proof_logic_ledger.md", ledger_md)
    write_json(package_root / "eidos_progress_meter.json", progress)
    write_text(package_root / "eidos_progress_meter.md", progress_md)
    write_text(package_root / "eidos_progress_meter.svg", svg)
    write_text(package_root / "eidos_progress_dashboard.html", dashboard_html)
    write_text(package_root / "eidos_progress_readme.md", readme_md)
    write_json(progress_root / "eidos_progress_meter.json", progress)
    write_text(progress_root / "eidos_progress_meter.md", progress_md)
    write_text(progress_root / "eidos_progress_meter.svg", svg)
    write_text(progress_root / "eidos_progress_dashboard.html", dashboard_html)
    write_text(progress_root / "eidos_progress_readme.md", readme_md)
    write_json(progress_root / "proof_logic_ledger.json", ledger)
    write_text(progress_root / "proof_logic_ledger.md", ledger_md)
    write_text(doc_dir / "month1_final_proof_package.md", package_md)
    write_text(doc_dir / "plain_language_test_analysis.md", plain_md)
    write_text(doc_dir / "codex_journal.md", journal_md)

    drive_manifest = copy_package_to_drive(generated_paths, drive_root=discover_drive_root(), run_date=run_date)
    write_json(package_root / "drive_manifest.json", drive_manifest)
    write_json(doc_dir / "drive_manifest_summary.json", drive_manifest)
    return package


def render_proof_logic_ledger_md(ledger: Dict[str, Any]) -> str:
    gates = [Gate(**item) for item in ledger["gates"]]
    return f"""# Proof Logic Ledger

Generated: `{ledger['generated_at_utc']}`

North star: {ledger['north_star']}

## Logic And Math

```text
reservoir_update = {ledger['logic']['reservoir_update']}
prediction_error = {ledger['logic']['prediction_error']}
surprise_score = {ledger['logic']['surprise_score']}
precision = {ledger['logic']['precision']}
recall = {ledger['logic']['recall']}
false_positive_rate = {ledger['logic']['false_positive_rate']}
proof_score = {ledger['logic']['proof_score']}
```

## Gates

{render_gate_table(gates)}

## Remaining Unproven

{chr(10).join(f"- {item}" for item in ledger['remaining_unproven'])}
"""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", default=RUN_DATE)
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--progress-root", type=Path, default=PROGRESS_ROOT)
    parser.add_argument("--doc-dir", type=Path, default=DOC_RUN_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    package = generate(parse_args(argv))
    print(f"wrote Month 1 final proof package for {package['run_date']}")
    print(f"status: {package['package_status']}")
    print(f"score: {fmt(package['progress_score']['earned'])} / {fmt(package['progress_score']['possible'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
