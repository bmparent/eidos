"""Check that gated Sentinel calibration work did not touch core behavior.

The policy is intentionally conservative. Calibration, proof-runner, reporting,
manifest, test, and docs work is allowed. Reservoir/RLS/hippocampus/compression/
thermodynamic/prediction-selection work is blocked unless a future task
explicitly opens that gate.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


FORBIDDEN_PATH_PATTERNS = (
    re.compile(r"(^|/)EIDOS_BRAIN_UNIFIED", re.IGNORECASE),
    re.compile(r"(^|/)repo/src/eidos_brain/engine/", re.IGNORECASE),
    re.compile(r"(^|/)repo/src/eidos_brain/prediction/", re.IGNORECASE),
    re.compile(r"(^|/)eidos_forecast\.py$", re.IGNORECASE),
    re.compile(r"(^|/)eidos_procedural_memory\.py$", re.IGNORECASE),
    re.compile(r"(^|/)eidos_tensor_utils\.py$", re.IGNORECASE),
)

FORBIDDEN_CONTENT_PATTERNS = (
    ("reservoir update", re.compile(r"\b(update|adapt|step|train).*reservoir|\breservoir.*(update|adapt|step|train)", re.IGNORECASE)),
    ("RLS adapt/update", re.compile(r"\brls\b.*\b(adapt|update|fit)|\b(adapt|update|fit).*rls\b", re.IGNORECASE)),
    ("hippocampus write/freeze", re.compile(r"\bhippocampus\b.*\b(write|freeze|snapshot|save)|\b(write|freeze|snapshot|save).*hippocampus\b", re.IGNORECASE)),
    ("compression codec", re.compile(r"\b(codec|compress|compression|ratio_accounting|ratio accounting)\b", re.IGNORECASE)),
    ("thermodynamic controller", re.compile(r"\b(thermodynamic|thermodynamics|active thermodynamics|controller)\b", re.IGNORECASE)),
    ("prediction selection", re.compile(r"\bprediction\b.*\b(select|selection|choose|rank)|\b(select|selection|choose|rank).*prediction\b", re.IGNORECASE)),
    ("default dynamics", re.compile(r"\b(spectral_radius|leak|forgetting|weight_decay)\b", re.IGNORECASE)),
)

ALLOWED_PATH_PATTERNS = (
    re.compile(r"^sentinel/(calibration|hysteresis|normal_suppression|event_merge|__init__)\.py$", re.IGNORECASE),
    re.compile(r"^proof/", re.IGNORECASE),
    re.compile(r"^tools/(run_labeled_domain_proof|colab_gpu_bridge|build_.*|check_core_touch_policy)\.py$", re.IGNORECASE),
    re.compile(r"^tests/", re.IGNORECASE),
    re.compile(r"^docs/", re.IGNORECASE),
    re.compile(r"^artifacts/", re.IGNORECASE),
    re.compile(r"^\.gitignore$", re.IGNORECASE),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_git(args: Sequence[str], *, cwd: Path, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=str(cwd), text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def git_prefix(*, cwd: Path) -> str:
    prefix = run_git(["rev-parse", "--show-prefix"], cwd=cwd, check=False).strip().replace("\\", "/")
    return prefix


def normalize_path(path: str, *, prefix: str = "") -> str:
    normalized = path.replace("\\", "/").strip().lstrip("./")
    if prefix and normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
    return normalized.lstrip("./")


def changed_paths(base: str, *, cwd: Path, include_worktree: bool) -> List[str]:
    prefix = git_prefix(cwd=cwd)
    paths = set(
        normalize_path(line, prefix=prefix)
        for line in run_git(["diff", "--name-only", "--diff-filter=ACMRTUXB", f"{base}...HEAD"], cwd=cwd).splitlines()
        if line.strip()
    )
    if include_worktree:
        for args in (
            ["diff", "--name-only", "--diff-filter=ACMRTUXB"],
            ["diff", "--cached", "--name-only", "--diff-filter=ACMRTUXB"],
            ["ls-files", "--others", "--exclude-standard"],
        ):
            output = run_git(args, cwd=cwd)
            paths.update(normalize_path(line, prefix=prefix) for line in output.splitlines() if line.strip())
    return sorted(paths)


def is_allowed_path(path: str) -> bool:
    return any(pattern.search(path) for pattern in ALLOWED_PATH_PATTERNS)


def forbidden_path_reason(path: str) -> Optional[str]:
    for pattern in FORBIDDEN_PATH_PATTERNS:
        if pattern.search(path):
            return f"forbidden core path: {path}"
    return None


def diff_for_path(path: str, base: str, *, cwd: Path, include_worktree: bool) -> str:
    chunks: List[str] = []
    for args in (["diff", f"{base}...HEAD", "--", path],):
        chunks.append(run_git(args, cwd=cwd, check=False))
    if include_worktree:
        chunks.append(run_git(["diff", "--", path], cwd=cwd, check=False))
        chunks.append(run_git(["diff", "--cached", "--", path], cwd=cwd, check=False))
        if not chunks[-3].strip() and not chunks[-2].strip() and not chunks[-1].strip():
            candidate = cwd / path
            if candidate.is_file():
                try:
                    chunks.append(candidate.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    pass
    return "\n".join(chunks)


def forbidden_content_reasons(text: str) -> List[str]:
    reasons: List[str] = []
    for label, pattern in FORBIDDEN_CONTENT_PATTERNS:
        if pattern.search(text):
            reasons.append(label)
    return reasons


def evaluate(base: str, *, cwd: Path, include_worktree: bool = True) -> Dict[str, Any]:
    paths = changed_paths(base, cwd=cwd, include_worktree=include_worktree)
    failures: List[Dict[str, Any]] = []
    allowed: List[str] = []
    for path in paths:
        if path.endswith(".pyc") or "/__pycache__/" in path:
            continue
        path_reason = forbidden_path_reason(path)
        diff_text = diff_for_path(path, base, cwd=cwd, include_worktree=include_worktree)
        content_reasons = forbidden_content_reasons(diff_text)
        if path_reason:
            failures.append({"path": path, "reason": path_reason, "content_reasons": content_reasons})
            continue
        if is_allowed_path(path):
            allowed.append(path)
            continue
        if content_reasons:
            failures.append({"path": path, "reason": "forbidden core-behavior content", "content_reasons": content_reasons})
        else:
            allowed.append(path)
    return {
        "generated_at_utc": utc_now(),
        "base": base,
        "include_worktree": include_worktree,
        "passed": not failures,
        "changed_paths": paths,
        "allowed_paths": allowed,
        "failures": failures,
        "policy": {
            "allowed": "calibration/proof-runner/reports/tests/docs/manifests/event postprocessing",
            "forbidden": "reservoir dynamics, RLS adapt/update, hippocampus write/freeze, compression codec, thermodynamic controller, prediction selection, default dynamics",
        },
    }


def write_report_md(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# Core Touch Policy",
        "",
        f"- Base: `{report.get('base')}`",
        f"- Passed: `{report.get('passed')}`",
        f"- Include worktree: `{report.get('include_worktree')}`",
        "",
        "## Failures",
        "",
    ]
    failures = report.get("failures") or []
    if not failures:
        lines.append("- No forbidden core touches found.")
    else:
        for item in failures:
            lines.append(f"- `{item.get('path')}`: {item.get('reason')} `{item.get('content_reasons')}`")
    lines.extend(["", "## Allowed Paths", ""])
    for path_item in report.get("allowed_paths", []):
        lines.append(f"- `{path_item}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="main", help="Base ref to compare against. Defaults to main.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--committed-only", action="store_true", help="Ignore unstaged/staged worktree changes.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = evaluate(args.base, cwd=args.repo_root, include_worktree=not args.committed_only)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    if args.md_out:
        write_report_md(args.md_out, report)
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
