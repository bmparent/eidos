"""Deterministic invariant guards; prompts are defense in depth only."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import PurePosixPath
from typing import ClassVar

from .permissions import Capability, PermissionPolicy
from .schemas import AuditResult, AuditVerdict, Claim, ClaimStatus, ImplementationResult


class GuardrailBlock(RuntimeError):
    pass


def _matches(path: str, patterns: list[str]) -> bool:
    normalized = PurePosixPath(path.replace("\\", "/")).as_posix()
    return any(fnmatch.fnmatch(normalized, pattern) or normalized == pattern.rstrip("/") for pattern in patterns)


class ScopeGuard:
    def check(self, changed_files: list[str], allowed_scope: list[str]) -> None:
        outside = [path for path in changed_files if not _matches(path, allowed_scope)]
        if outside:
            raise GuardrailBlock("files outside allowed scope: " + ", ".join(outside))


class ForbiddenScopeGuard:
    def check(self, changed_files: list[str], forbidden_scope: list[str]) -> None:
        denied = [path for path in changed_files if _matches(path, forbidden_scope)]
        if denied:
            raise GuardrailBlock("forbidden files changed: " + ", ".join(denied))


class SecretGuard:
    PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
    ]

    def check_text(self, text: str) -> None:
        if any(pattern.search(text) for pattern in self.PATTERNS):
            raise GuardrailBlock("credential-like content detected")

    def redacted_environment(self) -> dict[str, str]:
        safe: dict[str, str] = {}
        for key, value in os.environ.items():
            if any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")):
                safe[key] = "<redacted>"
            elif key in {"PATH", "PYTHONPATH", "VIRTUAL_ENV"}:
                safe[key] = value
        return safe


class MainBranchGuard:
    def check_write(self, branch: str) -> None:
        if branch in {"main", "master"}:
            raise GuardrailBlock("source writes are forbidden on protected branch")


class AuditSeparationGuard:
    def check(self, implementation: ImplementationResult, audit: AuditResult) -> None:
        if implementation.agent.lower() == audit.auditor.lower():
            raise GuardrailBlock("implementation agent cannot verify its own work")


class EvidenceGuard:
    def check_claim(self, claim: Claim) -> None:
        if claim.status == ClaimStatus.KNOWN and len(claim.evidence_refs) < 2:
            raise GuardrailBlock("unsupported KNOWN claim")


class CouncilGate:
    def check(self, *, enabled: bool, approved: bool, actor: str) -> None:
        if actor != "director":
            raise GuardrailBlock("only Director may invoke Council")
        if not enabled:
            raise GuardrailBlock("Council is disabled")
        if not approved:
            raise GuardrailBlock("Council requires human approval")


class MergeGate:
    def check(
        self,
        *,
        tests_passed: bool,
        benchmark_receipts_exist: bool,
        audit: AuditResult,
        git_state_understood: bool,
        human_approval: bool,
    ) -> None:
        failures: list[str] = []
        if not tests_passed:
            failures.append("tests")
        if not benchmark_receipts_exist:
            failures.append("benchmark receipts")
        if audit.verdict not in {AuditVerdict.PASS, AuditVerdict.PASS_WITH_LIMITATIONS}:
            failures.append(f"audit verdict {audit.verdict}")
        if not git_state_understood:
            failures.append("git state")
        if not human_approval:
            failures.append("human approval")
        if failures:
            raise GuardrailBlock("merge gate blocked by: " + ", ".join(failures))


class WriteAuthorityGuard:
    def __init__(self, permissions: PermissionPolicy | None = None) -> None:
        self.permissions = permissions or PermissionPolicy()

    def check(self, role: str, *, source: bool) -> None:
        self.permissions.require(role, Capability.WRITE_SOURCE if source else Capability.WRITE_ARTIFACTS)
