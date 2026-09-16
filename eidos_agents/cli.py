"""Command line interface for the Eidos Research Council."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .approvals import ApprovalManager, ApprovalRequired
from .config import LabConfig
from .doctor import run_doctor
from .missions import MissionDefinition
from .orchestrator import EidosOrchestrator
from .schemas import ApprovalRecord


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eidos_agents", description="Eidos proof-first agent research lab")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check local prerequisites")
    doctor.add_argument("--live-model-check", action="store_true")
    for name in ("investigate", "run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("objective")
        cmd.add_argument("--dry-run", action="store_true")
        cmd.add_argument("--mock", action="store_true")
    mission = sub.add_parser("mission")
    mission.add_argument("path", type=Path)
    mission.add_argument("--dry-run", action="store_true")
    mission.add_argument("--mock", action="store_true")
    for name in ("task", "evidence", "costs", "resume"):
        cmd = sub.add_parser(name)
        cmd.add_argument("task_id")
    approve = sub.add_parser("approve")
    approve.add_argument("task_id")
    approve.add_argument("action")
    approve.add_argument("--actor", default="Brent")
    sub.add_parser("hypotheses")
    sub.add_parser("experiments")
    return parser


def _config(args: argparse.Namespace) -> LabConfig:
    config = LabConfig.from_env(args.repo_root, dry_run=bool(getattr(args, "dry_run", False)))
    if args.artifact_root:
        config.artifact_root = args.artifact_root.resolve()
    return config


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _config(args)
    if args.command == "doctor":
        ok, checks = run_doctor(config, live_model_check=args.live_model_check)
        _print({"ok": ok, "checks": checks})
        return 0 if ok else 1
    orchestrator = EidosOrchestrator(config)
    if args.command in {"investigate", "run"}:
        research_only = args.command == "investigate"
        if args.mock or args.dry_run:
            decision = orchestrator.run_mocked(args.objective, research_only=research_only)
        else:
            try:
                decision = asyncio.run(
                    orchestrator.run_live(args.objective, research_only=research_only)
                )
            except ApprovalRequired as exc:
                _print({"status": "AWAITING_APPROVAL", "task_id": exc.task_id, "action": exc.action})
                return 3
        _print(decision.model_dump(mode="json"))
        return 0
    if args.command == "mission":
        mission = MissionDefinition.load(args.path)
        if not (args.mock or args.dry_run):
            print("Mission execution requires --dry-run/--mock in v1 until its experiment is human-authorized.", file=sys.stderr)
            return 2
        decision = orchestrator.run_mocked(mission.objective, research_only=True, mission=mission)
        _print(decision.model_dump(mode="json"))
        return 0
    if args.command == "approve":
        orchestrator.store.add_approval(ApprovalRecord(task_id=args.task_id, action=args.action, approved=True, actor=args.actor))
        _print({"task_id": args.task_id, "action": args.action, "approved": True, "actor": args.actor})
        return 0
    if args.command == "resume":
        state_path = orchestrator.store.task_dir(args.task_id) / "sdk_run_state.json"
        if state_path.exists():
            try:
                decision = asyncio.run(orchestrator.resume_live(args.task_id))
            except ApprovalRequired as exc:
                _print({"status": "AWAITING_APPROVAL", "task_id": exc.task_id, "action": exc.action})
                return 3
            _print(decision.model_dump(mode="json"))
            return 0
        _print(ApprovalManager(orchestrator.store).resume(args.task_id))
        return 0
    if args.command == "hypotheses":
        _print(orchestrator.store.list_records("hypothesis"))
        return 0
    if args.command == "experiments":
        _print(orchestrator.store.list_records("experiment"))
        return 0
    task_dir = orchestrator.store.task_dir(args.task_id)
    lookup = {
        "task": task_dir / "task.json",
        "evidence": task_dir / "evidence" / "evidence_packet.json",
        "costs": task_dir / "cost_receipt.json",
    }
    path = lookup[args.command]
    if not path.exists():
        print(f"not found: {path}", file=sys.stderr)
        return 1
    print(path.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
