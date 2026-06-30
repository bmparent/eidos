"""Minimal baseline registry for Eidos proof receipts.

This scaffold names the baselines expected by the proof plan and records which
ones were actually executed by the current proof wrapper. It is deliberately not
the full Week 3 competitor matrix.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "baseline_scaffold_v1"
COMPRESSION_BASELINES = ("raw", "gzip", "zstd", "brotli", "lzma", "delta", "delta+zstd")
DETECTOR_BASELINES = ("rolling_zscore", "ewma", "cusum", "isolation_forest", "noop", "random")


def _existing_compression_by_name(compression_baselines: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("name")): dict(item)
        for item in compression_baselines.get("baselines", [])
        if item.get("name")
    }


def _compression_record(
    name: str,
    *,
    existing_name: Optional[str],
    existing: Dict[str, Dict[str, Any]],
    skip_reason: str,
    runtime_seconds: Optional[float],
) -> Dict[str, Any]:
    source = existing.get(existing_name or name, {})
    ratio = source.get("compression_ratio")
    available = ratio is not None
    reason = "" if available else (source.get("skipped_reason") or skip_reason)
    return {
        "name": name,
        "type": "compression",
        "available": bool(available),
        "skip_reason": reason,
        "command": "python tools/run_labeled_domain_proof.py ...",
        "function": "tools.run_proof_baseline.compression_baselines_for_frames"
        if available
        else "registered in proof.baseline_scaffold",
        "metrics_emitted": {
            "raw_bytes": source.get("raw_bytes"),
            "compressed_bytes": source.get("compressed_bytes"),
            "compression_ratio": ratio,
        },
        "runtime_seconds": runtime_seconds if available else None,
    }


def _detector_record(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "type": "detector",
        "available": False,
        "skip_reason": "detector baseline scaffold registered for Week 3 expansion; not executed in this proof PR",
        "command": "not executed",
        "function": "registered in proof.baseline_scaffold",
        "metrics_emitted": {},
        "runtime_seconds": None,
    }


def build_baseline_scaffold(
    compression_baselines: Dict[str, Any],
    *,
    runtime_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    existing = _existing_compression_by_name(compression_baselines)
    compression_records = [
        _compression_record(
            "raw",
            existing_name="raw",
            existing=existing,
            skip_reason="no raw-byte baseline was emitted by the proof helper",
            runtime_seconds=runtime_seconds,
        ),
        _compression_record(
            "gzip",
            existing_name=None,
            existing=existing,
            skip_reason="gzip is registered but not executed in this PR; existing helper emits zlib, not gzip",
            runtime_seconds=runtime_seconds,
        ),
        _compression_record(
            "zstd",
            existing_name="zstd",
            existing=existing,
            skip_reason="zstandard package is not installed or zstd was not executed",
            runtime_seconds=runtime_seconds,
        ),
        _compression_record(
            "brotli",
            existing_name=None,
            existing=existing,
            skip_reason="brotli is registered but not executed in this proof PR",
            runtime_seconds=runtime_seconds,
        ),
        _compression_record(
            "lzma",
            existing_name="lzma",
            existing=existing,
            skip_reason="lzma was not emitted by the proof helper",
            runtime_seconds=runtime_seconds,
        ),
        _compression_record(
            "delta",
            existing_name=None,
            existing=existing,
            skip_reason="delta-only baseline is registered but not executed in this proof PR",
            runtime_seconds=runtime_seconds,
        ),
        _compression_record(
            "delta+zstd",
            existing_name=None,
            existing=existing,
            skip_reason="delta+zstd is registered but not executed in this proof PR",
            runtime_seconds=runtime_seconds,
        ),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "minimal scaffold only; not the full Week 3 competitor matrix",
        "compression_baselines": compression_records,
        "detector_baselines": [_detector_record(name) for name in DETECTOR_BASELINES],
        "best_external_compression_baseline": {
            "name": compression_baselines.get("best_baseline") or None,
            "compression_ratio": compression_baselines.get("best_baseline_compression_ratio") or None,
        },
        "skip_policy": "Every registered baseline that was not executed must carry an explicit skip_reason.",
    }


def write_baseline_scaffold_md(path: Any, scaffold: Dict[str, Any]) -> None:
    lines: List[str] = [
        "# Baseline Scaffold",
        "",
        f"- Schema: `{scaffold.get('schema_version')}`",
        f"- Scope: {scaffold.get('scope')}",
        "",
        "## Compression Baselines",
        "",
        "| name | available | ratio | skip reason | function |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for item in scaffold.get("compression_baselines", []):
        metrics = item.get("metrics_emitted") or {}
        lines.append(
            "| {name} | `{available}` | `{ratio}` | {reason} | `{function}` |".format(
                name=item.get("name"),
                available=item.get("available"),
                ratio=metrics.get("compression_ratio"),
                reason=item.get("skip_reason") or "",
                function=item.get("function"),
            )
        )
    lines.extend(
        [
            "",
            "## Detector Baselines",
            "",
            "| name | available | skip reason |",
            "| --- | --- | --- |",
        ]
    )
    for item in scaffold.get("detector_baselines", []):
        lines.append(
            f"| {item.get('name')} | `{item.get('available')}` | {item.get('skip_reason') or ''} |"
        )
    lines.extend(["", "## Policy", "", str(scaffold.get("skip_policy", ""))])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
