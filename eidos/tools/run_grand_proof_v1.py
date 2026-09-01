"""Repository-root CLI for EIDOS-GP-v1.

Examples:
  python eidos/tools/run_grand_proof_v1.py resource-profile --out eidos/artifacts/grand_proof_v1_run
  python eidos/tools/run_grand_proof_v1.py lock --out eidos/artifacts/grand_proof_v1_run
  python eidos/tools/run_grand_proof_v1.py run --stage smoke --seeds 0,1 --out eidos/artifacts/grand_proof_v1_run
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EIDOS_ROOT = REPO_ROOT / "eidos"
SRC_ROOT = EIDOS_ROOT / "repo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eidos_brain.proof.frame_observer import canonical_sha256  # noqa: E402
from eidos_brain.proof.grand_proof_runner import (  # noqa: E402
    BYTE_OPERATING_POINTS,
    GrandProofRunner,
    RunnerConfig,
    build_run_lock,
    capture_live_scenario,
    git_output,
    utc_now,
    verify_run_lock,
    write_json,
    synthetic_domain_contract,
)
from eidos_brain.proof.grand_proof_scenarios import (  # noqa: E402
    SCENARIO_IDS,
    ScenarioConfig,
    generate_scenario,
)


DEFAULT_THRESHOLDS = {
    "eidos_ms_full": 0.25,
    "eidos_live_current": 1.5,
    "eidos_minimal": 3.0,
    "rolling_z": 3.0,
    "ewma": 3.0,
    "cusum": 8.0,
    "isolation_forest": 0.65,
    "knn_episode": 0.45,
}


def parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def parse_scenarios(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(items) - set(SCENARIO_IDS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown scenarios: {unknown}")
    return items


def default_out() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return EIDOS_ROOT / "artifacts" / f"grand_proof_v1_{stamp}"


def command_line() -> str:
    return subprocess.list2cmdline([sys.executable, *sys.argv])


def initialize_artifacts(out: Path) -> None:
    for subdir in (
        "protocol",
        "provenance",
        "configs",
        "captures",
        "scenarios",
        "domains",
        "ablations",
        "statistics",
        "failures/counterexamples",
        "reports",
    ):
        (out / subdir).mkdir(parents=True, exist_ok=True)
    design = EIDOS_ROOT / "docs" / "proof" / "design_freeze"
    copies = {
        "meaningful_surprise_v1_spec.md": "meaningful_surprise_v1_spec.md",
        "grand_proof_protocol_v1.md": "grand_proof_protocol_v1.md",
        "local_codex_execution_brief_v1.md": "execution_prompt.md",
        "design_freeze_manifest_v1.json": "design_freeze_manifest_v1.json",
    }
    for source, destination in copies.items():
        shutil.copy2(design / source, out / "protocol" / destination)


def append_command(out: Path, *, status: str, exit_code: int | None = None) -> None:
    path = out / "provenance" / "commands.jsonl"
    row = {
        "timestamp_utc": utc_now(),
        "command": command_line(),
        "status": status,
        "exit_code": exit_code,
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_environment(out: Path) -> None:
    modules: dict[str, str] = {}
    for name in ("numpy", "pandas", "pytest", "torch", "sklearn", "zstandard", "psutil"):
        try:
            module = __import__(name)
            modules[name] = str(getattr(module, "__version__", "present"))
        except Exception as exc:
            modules[name] = f"MISSING:{type(exc).__name__}"
    text = "\n".join(
        [
            f"timestamp_utc={utc_now()}",
            f"python={sys.version}",
            f"platform={platform.platform()}",
            f"cpu_count={os.cpu_count()}",
            *[f"{name}={value}" for name, value in sorted(modules.items())],
        ]
    )
    (out / "provenance" / "environment.txt").write_text(text + "\n", encoding="utf-8")
    write_json(out / "provenance" / "dependency_inventory.json", modules)


def identity_receipt(out: Path) -> dict[str, Any]:
    identities = {}
    for value in (
        "f676fe2342b98886bc04cd8f4b0e943fce77ec9a",
        "6f98eebad6a60b50c85ae5fceade9f1857a88177",
        "00489e358865994ec4b40a5c5bfdfa034560773a",
        "4a639cd693701fb764fe30ba672d4811bdbf5a75",
    ):
        probe = subprocess.run(
            ["git", "cat-file", "-e", f"{value}^{{commit}}"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if probe.returncode == 0:
            identities[value] = {
                "present": True,
                "subject": git_output(REPO_ROOT, "show", "-s", "--format=%s", value),
                "parents": git_output(REPO_ROOT, "show", "-s", "--format=%P", value).split(),
            }
        else:
            identities[value] = {"present": False, "subject": None, "parents": []}
    receipt = {
        "receipt_version": "EIDOS-AUGUST-SHA-IDENTITY-v1",
        "timestamp_utc": utc_now(),
        "identities": identities,
        "mapping": {
            "main_operator_explanation": "f676fe2342b98886bc04cd8f4b0e943fce77ec9a",
            "evidence_link_code": "6f98eebad6a60b50c85ae5fceade9f1857a88177",
            "proof_receipts": "00489e358865994ec4b40a5c5bfdfa034560773a",
            "picqr_v2_cited_implementation": "4a639cd693701fb764fe30ba672d4811bdbf5a75",
        },
        "resolution": "UNRESOLVED",
        "reason": (
            "The fetched repository proves 6f98eeb is the evidence-link implementation and "
            "00489e3 adds its receipts, but object 4a639cd is absent from all fetched refs. "
            "No verifiable mapping can equate the cited object with 6f98eeb."
        ),
        "gate_effect": "PICQR-v2 independent human gate blocked; engineering work may proceed.",
    }
    write_json(out / "provenance" / "august_sha_identity.json", receipt)
    return receipt


def verify_design_freeze(out: Path) -> dict[str, Any]:
    manifest_path = EIDOS_ROOT / "docs" / "proof" / "design_freeze" / "design_freeze_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for entry in manifest["files"]:
        path = REPO_ROOT / entry["path"]
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        actual_bytes = path.stat().st_size
        rows.append(
            {
                "path": entry["path"],
                "expected_bytes": entry["bytes"],
                "actual_bytes": actual_bytes,
                "expected_sha256": entry["sha256"],
                "actual_sha256": actual_hash,
                "verified": actual_bytes == entry["bytes"] and actual_hash == entry["sha256"],
            }
        )
    receipt = {"manifest_version": manifest["manifest_version"], "files": rows, "all_verified": all(row["verified"] for row in rows)}
    write_json(out / "provenance" / "design_freeze_verification.json", receipt)
    return receipt


def dataset_receipts(
    cicids_path: Path | None,
    noncyber_path: Path | None,
    discovery_receipt: Path | None = None,
) -> dict[str, Any]:
    expected_cicids = "d67066211fb1689c78406f1506f4c44704ecb92088353d5c96d96d6474eb819d"
    discovery = _read_json(discovery_receipt) if discovery_receipt is not None else None
    discovery = discovery or {}
    receipts: dict[str, Any] = {}
    if cicids_path is None or not cicids_path.is_file():
        receipts["cicids_webattacks"] = {
            "status": "BLOCKED_DATA_MISSING",
            "expected_sha256": expected_cicids,
            "path": None if cicids_path is None else str(cicids_path),
            "drive_discovery": discovery.get("cicids_webattacks"),
        }
    else:
        actual = hashlib.sha256(cicids_path.read_bytes()).hexdigest()
        receipts["cicids_webattacks"] = {
            "status": "VERIFIED" if actual == expected_cicids else "BLOCKED_IDENTITY_MISMATCH",
            "path": str(cicids_path),
            "bytes": cicids_path.stat().st_size,
            "expected_sha256": expected_cicids,
            "actual_sha256": actual,
        }
    if noncyber_path is None or not noncyber_path.is_file():
        receipts["real_noncyber"] = {
            "status": (
                "BLOCKED_PROVENANCE_AND_LOCAL_MATERIALIZATION"
                if discovery.get("real_noncyber", {}).get("raw_drive_file_found")
                else "BLOCKED_DATA_MISSING"
            ),
            "path": None if noncyber_path is None else str(noncyber_path),
            "reason": (
                "raw source was found in Drive, but no license/README was found and no local byte stream was available for hashing"
                if discovery.get("real_noncyber", {}).get("raw_drive_file_found")
                else "eligible raw epileptic-seizure-recognition source was not found locally"
            ),
            "drive_discovery": discovery.get("real_noncyber"),
            "verdict_cap": "SYNTHETIC_AND_CYBER_ONLY",
        }
    else:
        license_candidates = [
            noncyber_path.with_name("LICENSE"),
            noncyber_path.with_name("LICENSE.txt"),
            noncyber_path.with_name("README.md"),
        ]
        license_path = next((path for path in license_candidates if path.is_file()), None)
        receipts["real_noncyber"] = {
            "status": "VERIFIED_SOURCE_ONLY" if license_path else "BLOCKED_PROVENANCE",
            "path": str(noncyber_path),
            "bytes": noncyber_path.stat().st_size,
            "sha256": hashlib.sha256(noncyber_path.read_bytes()).hexdigest(),
            "license_path": None if license_path is None else str(license_path),
            "statement": "signal benchmark only; not medical validation or clinical advice",
        }
    return receipts


def drive_manifest(out: Path) -> dict[str, Any]:
    candidates = [os.environ.get("EIDOS_PROOF_DRIVE_DIR"), os.environ.get("EIDOS_ARTIFACT_ROOT")]
    root = next((Path(value) for value in candidates if value and Path(value).exists() and os.access(value, os.W_OK)), None)
    if root is None:
        receipt = {
            "drive_copy_attempted": False,
            "drive_copy_success": False,
            "drive_root": "unknown",
            "drive_run_dir": "unknown",
            "reason": "EIDOS_PROOF_DRIVE_DIR/EIDOS_ARTIFACT_ROOT not set to a writable mounted Drive path",
            "files_considered": [],
            "files_copied": [],
            "files_skipped": [],
            "timestamp_utc": utc_now(),
        }
    else:
        target = root / "Eidos_Brain_Proof_Phase" / datetime.now(timezone.utc).strftime("%Y-%m-%d") / out.name
        target.mkdir(parents=True, exist_ok=True)
        considered = []
        copied = []
        for source in out.rglob("*"):
            if source.is_file() and source.name != "drive_manifest.json":
                relative = source.relative_to(out)
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                considered.append(relative.as_posix())
                copied.append(relative.as_posix())
        receipt = {
            "drive_copy_attempted": True,
            "drive_copy_success": True,
            "drive_root": str(root),
            "drive_run_dir": str(target),
            "reason": None,
            "files_considered": considered,
            "files_copied": copied,
            "files_skipped": [],
            "timestamp_utc": utc_now(),
        }
    write_json(out / "drive_manifest.json", receipt)
    return receipt


def resource_profile(out: Path, *, time_budget_seconds: float) -> dict[str, Any]:
    import psutil

    config = ScenarioConfig(features=64, warmup_frames=200, scored_frames=1800, outcome_horizon=64)
    scenario = generate_scenario("S0_nominal", seed=0, config=config)
    total_suite_frames = (2 + 10 + 20) * len(SCENARIO_IDS) * ScenarioConfig().total_frames
    available_limit = int(psutil.virtual_memory().total * 0.70)
    rows = []
    selected = None
    for reservoir in (128, 256, 512, 1024):
        process = psutil.Process()
        before = process.memory_info().rss
        peak_samples = [before]
        stop_sampling = threading.Event()

        def sample_memory() -> None:
            while not stop_sampling.wait(0.02):
                try:
                    peak_samples.append(process.memory_info().rss)
                except psutil.Error:
                    return

        sampler = threading.Thread(target=sample_memory, name=f"gp-memory-{reservoir}", daemon=True)
        sampler.start()
        try:
            records, receipt = capture_live_scenario(
                scenario,
                out_dir=out / "resource_profiles" / str(reservoir),
                reservoir=reservoir,
                code_commit=git_output(REPO_ROOT, "rev-parse", "HEAD"),
                replay_command=command_line(),
            )
        finally:
            stop_sampling.set()
            sampler.join(timeout=2.0)
        after = process.memory_info().rss
        peak_samples.append(after)
        measured_peak = max(peak_samples)
        projected = total_suite_frames / max(receipt["frames_per_second"], 1e-12)
        eligible = projected <= time_budget_seconds and measured_peak <= available_limit
        rows.append(
            {
                "reservoir": reservoir,
                "input_frames": config.total_frames,
                "observed_scored_frames": len(records),
                "runtime_seconds": receipt["runtime_seconds"],
                "frames_per_second": receipt["frames_per_second"],
                "projected_full_suite_seconds": projected,
                "rss_before_bytes": before,
                "rss_after_bytes": after,
                "measured_peak_rss_bytes": measured_peak,
                "memory_samples": len(peak_samples),
                "memory_limit_70pct_bytes": available_limit,
                "eligible": eligible,
                "selection_inputs_only": ["runtime", "memory"],
            }
        )
        if eligible:
            selected = reservoir
    receipt = {
        "receipt_version": "EIDOS-GP-v1-RESOURCE-v1",
        "timestamp_utc": utc_now(),
        "code_commit": git_output(REPO_ROOT, "rev-parse", "HEAD"),
        "time_budget_seconds": time_budget_seconds,
        "minimum_projected_synthetic_frames": total_suite_frames,
        "projection_scope": "smoke+calibration+heldout synthetic suite only; real domains and transfer stress would add work",
        "profiles": rows,
        "selected_reservoir": selected,
        "selection_status": "SELECTED" if selected is not None else "BLOCKED_RESOURCE",
        "quality_metrics_used_for_selection": False,
    }
    write_json(out / "provenance" / "resource_profile.json", receipt)
    return receipt


def artifact_manifest(out: Path) -> dict[str, Any]:
    files = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {
                    "path": path.relative_to(out).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    manifest = {"generated_at_utc": utc_now(), "files": files, "file_count": len(files)}
    write_json(out / "provenance" / "artifact_manifest.json", manifest)
    return manifest


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_final_reports(out: Path) -> dict[str, Any]:
    resource = _read_json(out / "provenance" / "resource_profile.json")
    datasets = _read_json(out / "provenance" / "dataset_manifest.json") or {}
    identity = _read_json(out / "provenance" / "august_sha_identity.json") or {}
    lock = _read_json(out / "protocol" / "run_lock.json")
    smoke = _read_json(out / "provenance" / "run_smoke_result.json")
    calibration = _read_json(out / "provenance" / "run_calibration_result.json")
    heldout = _read_json(out / "provenance" / "run_heldout_result.json")
    lock_failures = ["run_lock.json is missing"] if lock is None else verify_run_lock(lock, repo_root=REPO_ROOT)
    data_blockers = [
        {"domain": name, **receipt}
        for name, receipt in datasets.items()
        if str(receipt.get("status", "")).startswith("BLOCKED")
    ]
    resource_blocked = resource is None or resource.get("selection_status") != "SELECTED"
    if lock_failures and not resource_blocked:
        verdict = "INVALID_RUN"
        rationale = "The execution lock did not verify; no empirical value claim is valid."
    elif heldout is None:
        verdict = "BLOCKED_RESOURCE_BEFORE_HELDOUT"
        rationale = (
            "The held-out phase was not opened. The implementation, tests, smoke evidence, "
            "resource receipt, and execution lock are preserved without reducing the protocol."
        )
    else:
        verdict = "MECHANISM_NOT_SUPPORTED"
        rationale = (
            "Held-out receipts exist, but automatic acceptance-gate adjudication has not established "
            "all preregistered mechanism and matched-budget conditions."
        )
    verdict_receipt = {
        "protocol_id": "EIDOS-GP-v1-2026-09-01",
        "generated_at_utc": utc_now(),
        "verdict": verdict,
        "rationale": rationale,
        "design_lock_verified": bool((_read_json(out / "provenance" / "design_freeze_verification.json") or {}).get("all_verified")),
        "execution_lock_present": lock is not None,
        "execution_lock_verified": not lock_failures,
        "execution_lock_failures": lock_failures,
        "resource_status": None if resource is None else resource.get("selection_status"),
        "smoke_status": "COMPLETE" if smoke and not smoke.get("failures") else "MISSING_OR_FAILED",
        "calibration_status": "COMPLETE" if calibration and not calibration.get("failures") else "NOT_RUN",
        "heldout_status": "COMPLETE" if heldout and not heldout.get("failures") else "NOT_RUN",
        "data_blockers": data_blockers,
        "identity_blocker": identity.get("resolution") != "RESOLVED",
        "independent_review_status": "BLOCKED_IDENTITY_AND_NOT_EXECUTED",
        "claim_boundary": "No production-readiness, state-of-the-art, clinical, attack, compromise, or cross-domain claim is supported.",
    }
    write_json(out / "reports" / "final_verdict.json", verdict_receipt)

    proof_logic = f"""# Proof Logic + Meaning

## Goal reached

The `EIDOS-MS-v1` shadow observer and proof harness were implemented against the locked design. The current Grand Proof gate is **{verdict}**: tests and smoke/resource receipts are evidence, but they are not a held-out value result.

## Previous state

The live engine had no append-only seam that captured the completed predictor, residual, Sentinel, HDC, thermodynamic, and codec decisions together. Meaningful Surprise and Grand Proof existed only as locked specifications, not as a runnable, causal full-engine shadow path.

## Technical logic utilized

The live engine runs once per stream. An instrumentation-only observer records the completed decision. Past-only representation lifts, delayed consequence memory, value-of-information lower bounds, a quotient residual, persistence, disagreement, and an unprojected raw-residual escape are then evaluated in shadow. Every baseline and A0-A7 ablation consumes the same capture.

## Math / scoring logic

The observer retains the live residual `e_t = x_t - xhat_t`, normalized error `epsilon_t = ||e_t||_2 / sqrt(d)`, and surprise score. Event precision is `TP / (TP + FP)`, recall is `TP / (TP + FN)`, and `FP_per_10k = FP / nominal_frames * 10000`. Paired intervals resample whole seeds 10,000 times and Holm-adjust each registered component family. Byte cost includes payload, index, cards, model state, and manifest before division by raw float bytes.

## Philosophical meaning

This milestone represents restraint before alarm and reproducibility before claim: the system may propose why a deviation matters, but it cannot promote its own hypothesis into validated meaning or alter the live engine.

## Why this is better

The implementation removes the proxy/full-engine mismatch and creates replayable, hashed receipts. Missing data, missing review, optional dependencies, resource limits, negative ablations, false positives, and partial runs remain visible rather than becoming zeros or favorable claims.

## How this moves Eidos closer to the north-star goal

It strengthens the “monitors its own internal state,” “preserves meaningful anomalies,” “explains incidents,” and “runs reproducibly” parts of the north-star claim. It does not yet prove value beyond normal compressors or detectors.

## Evidence

- `provenance/design_freeze_verification.json`
- `provenance/resource_profile.json`
- `protocol/run_lock.json`
- `ablations/paired_results.csv`
- `statistics/paired_intervals.csv`
- `reports/final_verdict.json`

## Remaining uncertainty

Held-out synthetic evidence is {"present" if heldout else "absent"}. CICIDS/WebAttacks and a real non-cyber source remain subject to the dataset receipts. The PICQR implementation SHA is unresolved, and no independent operator review was performed. GPU performance was not tested. Smoke results cannot establish a Grand Proof acceptance gate.

## Exactly one next experiment

On a machine/session that satisfies the locked resource budget, rerun the unchanged execution lock through calibration and held-out seeds before inspecting any held-out outcomes.
"""
    (out / "reports" / "proof_logic_meaning.md").write_text(proof_logic, encoding="utf-8")

    benchmark_report = f"""# EIDOS Grand Proof v1 — Bounded Execution Report

## Status

**{verdict}**

{rationale}

## Run stages

- Focused implementation tests: see `provenance/pytest_focused.xml` when present.
- Compatible repository tests: see `provenance/pytest_results.xml` when present.
- Smoke: {verdict_receipt['smoke_status']}.
- Calibration: {verdict_receipt['calibration_status']}.
- Held-out: {verdict_receipt['heldout_status']}.

## Blockers

- Data blockers: {json.dumps(data_blockers, sort_keys=True)}
- Identity: {identity.get('resolution', 'UNKNOWN')} — {identity.get('reason', 'receipt missing')}
- Independent review: not performed by Codex; the implementation team is ineligible as the primary reviewer.

## Claim boundary

{verdict_receipt['claim_boundary']}
"""
    (out / "reports" / "benchmark_report.md").write_text(benchmark_report, encoding="utf-8")
    theorem_status = f"""# Theorem / Acceptance-Gate Status

- G0 identity and reproducibility: **partial** — design receipts exist; execution lock verification failures: {lock_failures or 'none'}.
- G1 causality and label isolation: **implementation-tested**, not independently audited.
- G2 safety invariants: **not established on held-out seeds**.
- G3 mechanism support: **not established**.
- G4 matched-budget joint value: **not established**.
- G5 cross-domain value: **blocked by dataset receipts and absent held-out evidence**.
- G6 independent operator evidence: **blocked/pending**.

Final protocol status: **{verdict}**.
"""
    (out / "reports" / "theorem_status.md").write_text(theorem_status, encoding="utf-8")
    plain = f"""# Plain-Language Test Analysis

This task added a read-only observer to the actual Eidos streaming engine and ran the proposed Meaningful Surprise logic beside it. The shadow path cannot change predictions, thresholds, Sentinel labels, memory writes, or codec output. Tests verify that turning the observer off creates no artifact, invalid numbers fail visibly, interrupted captures can resume, and permuting sealed labels cannot change online decisions.

The current result is **{verdict}**. That means the work produced implementation and bounded execution evidence, not proof that Eidos beats another detector or compressor. Missing raw-domain data, the unresolved August identity object, machine resource limits, and independent human review are kept as explicit blockers. See `reports/proof_logic_meaning.md` for the logic, equations, evidence, philosophical meaning, and the one registered next experiment.
"""
    (out / "reports" / "plain_language_test_analysis.md").write_text(plain, encoding="utf-8")
    return verdict_receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "resource-profile", "lock", "run", "verify", "finalize"):
        item = sub.add_parser(name)
        item.add_argument("--out", type=Path, required=True)
        item.add_argument("--cicids-path", type=Path)
        item.add_argument("--noncyber-path", type=Path)
        item.add_argument("--dataset-discovery-receipt", type=Path)
    sub.choices["resource-profile"].add_argument("--time-budget-seconds", type=float, default=21600.0)
    sub.choices["run"].add_argument("--stage", choices=("smoke", "calibration", "heldout"), required=True)
    sub.choices["run"].add_argument("--seeds", type=parse_seeds, required=True)
    sub.choices["run"].add_argument("--scenarios", type=parse_scenarios, default=list(SCENARIO_IDS))
    sub.choices["run"].add_argument("--reservoir", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out.resolve()
    initialize_artifacts(out)
    append_command(out, status="STARTED")
    write_environment(out)
    (out / "provenance" / "git_commit.txt").write_text(git_output(REPO_ROOT, "rev-parse", "HEAD") + "\n", encoding="utf-8")
    (out / "provenance" / "git_status.txt").write_text(git_output(REPO_ROOT, "status", "--short", "--branch") + "\n", encoding="utf-8")
    try:
        design = verify_design_freeze(out)
        identity = identity_receipt(out)
        datasets = dataset_receipts(args.cicids_path, args.noncyber_path, args.dataset_discovery_receipt)
        write_json(out / "provenance" / "dataset_manifest.json", datasets)
        if not design["all_verified"]:
            raise RuntimeError("design-freeze verification failed")
        if args.command == "preflight":
            result = {"design": design, "identity": identity, "datasets": datasets}
        elif args.command == "resource-profile":
            result = resource_profile(out, time_budget_seconds=args.time_budget_seconds)
        elif args.command == "lock":
            resource_path = out / "provenance" / "resource_profile.json"
            if not resource_path.is_file():
                raise RuntimeError("resource profile must exist before execution lock")
            resource = json.loads(resource_path.read_text(encoding="utf-8"))
            selected = resource.get("selected_reservoir")
            if selected is None:
                # Preserve an execution lock even when no profile is eligible.
                # The heldout_allowed field remains false and held-out execution
                # fails closed; 128 is only the bounded smoke implementation profile.
                selected = 128
            lock = build_run_lock(
                repo_root=REPO_ROOT,
                artifact_root=out,
                reservoir=int(selected),
                scenario_config=ScenarioConfig(),
                thresholds=DEFAULT_THRESHOLDS,
                datasets=datasets,
                resource_receipt=resource,
            )
            write_json(out / "protocol" / "run_lock.json", lock)
            write_json(
                out / "protocol" / "lock_hashes.json",
                {
                    "run_lock_sha256": lock["run_lock_sha256"],
                    "design_freeze": lock["locked_files"],
                },
            )
            write_json(
                out / "configs" / "engine_config.json",
                {
                    "reservoir": int(selected),
                    "features": ScenarioConfig().features,
                    "warmup_frames": ScenarioConfig().warmup_frames,
                    "residual_codec_enabled": True,
                    "observer_mode": "shadow_only",
                },
            )
            write_json(out / "configs" / "domain_contracts.json", {"synthetic": lock["domain_contract"]})
            write_json(out / "configs" / "scenario_config.json", lock["scenario_config"])
            write_json(
                out / "configs" / "baseline_config.json",
                {
                    "systems": lock["systems"],
                    "thresholds": lock["thresholds"],
                    "missing_value_policy": "SKIPPED with explicit reason; never zero",
                },
            )
            write_json(
                out / "configs" / "metric_config.json",
                {
                    "primary_fp_per_10k_limit": 5.0,
                    "byte_operating_points": list(BYTE_OPERATING_POINTS),
                    "bootstrap_resamples": 10000,
                    "multiple_comparison_correction": "Holm within stage/scenario/component family",
                },
            )
            result = lock
        elif args.command == "verify":
            lock_path = out / "protocol" / "run_lock.json"
            if not lock_path.is_file():
                raise RuntimeError("run_lock.json is missing")
            failures = verify_run_lock(json.loads(lock_path.read_text(encoding="utf-8")), repo_root=REPO_ROOT)
            result = {"verified": not failures, "failures": failures}
            if failures:
                raise RuntimeError("; ".join(failures))
        elif args.command == "run":
            lock_path = out / "protocol" / "run_lock.json"
            lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.is_file() else None
            if args.stage in {"calibration", "heldout"} and lock is None:
                raise RuntimeError("execution lock is required before calibration/heldout")
            if args.stage == "heldout":
                failures = verify_run_lock(lock, repo_root=REPO_ROOT)
                if failures:
                    raise RuntimeError("heldout preflight failed: " + "; ".join(failures))
                if not lock.get("heldout_allowed", False):
                    raise RuntimeError("heldout blocked by the locked resource-profile receipt")
            reservoir = args.reservoir or (int(lock["reservoir"]) if lock else 128)
            config = ScenarioConfig.smoke() if args.stage == "smoke" else ScenarioConfig()
            runner = GrandProofRunner(
                RunnerConfig(
                    artifact_root=out,
                    repo_root=REPO_ROOT,
                    reservoir=reservoir,
                    scenario_config=config,
                    code_commit=git_output(REPO_ROOT, "rev-parse", "HEAD"),
                    thresholds=DEFAULT_THRESHOLDS if lock is None else lock["thresholds"],
                )
            )
            result = runner.run(stage=args.stage, seeds=args.seeds, scenarios=args.scenarios)
        else:
            failure_ledger = out / "failures" / "failure_ledger.jsonl"
            if not failure_ledger.exists():
                failure_ledger.write_text("", encoding="utf-8")
            verdict = write_final_reports(out)
            result = {
                "final_verdict": verdict,
                "artifact_manifest": artifact_manifest(out),
                "drive_manifest": drive_manifest(out),
            }
        result_name = f"run_{args.stage}_result.json" if args.command == "run" else f"{args.command}_result.json"
        write_json(out / "provenance" / result_name, result)
        artifact_manifest(out)
        append_command(out, status="COMPLETED", exit_code=0)
        return 0
    except Exception as exc:
        write_json(
            out / "failures" / f"{args.command}_failure.json",
            {"timestamp_utc": utc_now(), "error_type": type(exc).__name__, "error": str(exc)},
        )
        artifact_manifest(out)
        append_command(out, status="FAILED", exit_code=1)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
