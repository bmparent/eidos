"""Code-enforced role capabilities."""

from __future__ import annotations

from enum import StrEnum


class Capability(StrEnum):
    READ_REPO = "READ_REPO"
    READ_ARTIFACTS = "READ_ARTIFACTS"
    WEB_SEARCH = "WEB_SEARCH"
    INVOKE_SPECIALISTS = "INVOKE_SPECIALISTS"
    WRITE_SOURCE = "WRITE_SOURCE"
    WRITE_ARTIFACTS = "WRITE_ARTIFACTS"
    RUN_TESTS = "RUN_TESTS"
    SAFE_SHELL = "SAFE_SHELL"
    COMMIT_FEATURE_BRANCH = "COMMIT_FEATURE_BRANCH"
    MERGE = "MERGE"
    DEPLOY = "DEPLOY"
    INVOKE_COUNCIL = "INVOKE_COUNCIL"


ROLE_CAPABILITIES: dict[str, frozenset[Capability]] = {
    "director": frozenset({Capability.READ_REPO, Capability.READ_ARTIFACTS, Capability.INVOKE_SPECIALISTS, Capability.INVOKE_COUNCIL}),
    "archivist": frozenset({Capability.READ_REPO, Capability.READ_ARTIFACTS, Capability.WEB_SEARCH}),
    "curie": frozenset({Capability.READ_REPO, Capability.READ_ARTIFACTS}),
    "sentry": frozenset({Capability.READ_REPO, Capability.READ_ARTIFACTS}),
    "gauss": frozenset({Capability.READ_REPO, Capability.READ_ARTIFACTS, Capability.WEB_SEARCH}),
    "forge": frozenset({Capability.READ_REPO, Capability.READ_ARTIFACTS, Capability.WRITE_SOURCE, Capability.RUN_TESTS, Capability.SAFE_SHELL, Capability.COMMIT_FEATURE_BRANCH}),
    "bench": frozenset({Capability.READ_REPO, Capability.READ_ARTIFACTS, Capability.WRITE_ARTIFACTS, Capability.RUN_TESTS, Capability.SAFE_SHELL}),
    "auditor": frozenset({Capability.READ_REPO, Capability.READ_ARTIFACTS, Capability.RUN_TESTS, Capability.SAFE_SHELL}),
    "council": frozenset({Capability.READ_REPO, Capability.READ_ARTIFACTS}),
}


class PermissionDenied(RuntimeError):
    pass


class PermissionPolicy:
    def require(self, role: str, capability: Capability) -> None:
        if capability not in ROLE_CAPABILITIES.get(role, frozenset()):
            raise PermissionDenied(f"{role} lacks {capability}")

    def can(self, role: str, capability: Capability) -> bool:
        return capability in ROLE_CAPABILITIES.get(role, frozenset())
