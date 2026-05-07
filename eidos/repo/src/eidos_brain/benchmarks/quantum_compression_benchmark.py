"""Synthetic benchmark for quantum-aware anomaly-preserving compression."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import lzma
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from eidos_brain.adapters import (
    BinaryRiverAdapter,
    CryptoAgilityAdapter,
    QuantumSyndromeAdapter,
    generate_binary_stream,
    generate_crypto_agility_stream,
    generate_quantum_telemetry_stream,
)
from eidos_brain.compression import (
    OptionalCodecUnavailable,
    ResidualFirstCodec,
    anomaly_preservation_score,
    pack_tokens,
    reconstruction_error,
    tokens_to_jsonl,
)


@dataclass
class BenchmarkResult:
    scenario: str
    method: str
    raw_bytes: int
    token_bytes: int | None
    compressed_bytes: int
    compression_ratio: float
    reconstruction_error: float | None
    anomaly_preservation_score: float | None
    anomaly_capsule_count: int | None
    normal_frame_compression_ratio: float | None
    anomaly_frame_compression_ratio: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    false_positives: int | None
    false_negatives: int | None
    detection_delay: int | None
    regime_status_distribution: dict[str, int]
    surprise_z_mean: float | None
    surprise_z_p95: float | None
    surprise_z_separation: float | None
    residual_norm_mean: float | None
    residual_norm_p95: float | None
    compression_collapse_score: float | None
    notes: str


def run_benchmark(
    output_dir: str | Path = "artifacts/benchmarks",
    n_frames: int = 96,
    seed: int = 7,
    include_optional: bool = True,
) -> list[dict[str, Any]]:
    """Run all local synthetic scenarios and write JSON/Markdown/CSV reports."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results: list[BenchmarkResult] = []
    for scenario_name, frames, metadata, labels in _scenario_frames(n_frames=n_frames, seed=seed):
        results.extend(_run_scenario(scenario_name, frames, metadata, labels, include_optional=include_optional))

    result_dicts = [asdict(result) for result in results]
    json_path = output_path / "quantum_compression_comparison.json"
    md_path = output_path / "quantum_compression_comparison.md"
    csv_path = output_path / "quantum_compression_comparison.csv"
    json_path.write_text(json.dumps(result_dicts, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown_report(results), encoding="utf-8")
    _write_csv(csv_path, results)
    return result_dicts


def _scenario_frames(n_frames: int, seed: int) -> list[tuple[str, np.ndarray, list[dict[str, Any]], np.ndarray]]:
    scenarios: list[tuple[str, np.ndarray, list[dict[str, Any]], np.ndarray]] = []

    quantum_adapter = QuantumSyndromeAdapter(features=64, hash_seed=seed)
    for offset, scenario in enumerate(["normal", "decoherence_drift", "syndrome_burst", "readout_bias_shift"]):
        quantum_adapter.reset()
        events = generate_quantum_telemetry_stream(scenario=scenario, n_frames=n_frames, seed=seed + offset)
        frames, metadata = quantum_adapter.transform_many(events)
        labels = np.asarray([bool(event["is_anomaly"]) for event in events], dtype=bool)
        scenarios.append((f"quantum_{scenario}", frames, metadata, labels))

    crypto_adapter = CryptoAgilityAdapter(features=64, hash_seed=seed + 100)
    crypto_events = generate_crypto_agility_stream(
        scenario="crypto_downgrade_exfiltration_risk",
        n_frames=n_frames,
        seed=seed + 20,
    )
    frames, metadata = crypto_adapter.transform_many(crypto_events)
    labels = np.asarray([bool(event["is_anomaly"]) for event in crypto_events], dtype=bool)
    scenarios.append(("crypto_downgrade_exfiltration_risk", frames, metadata, labels))

    binary_adapter = BinaryRiverAdapter(features=64, hash_seed=seed + 200)
    binary_events = generate_binary_stream(scenario="high_entropy_blob", n_windows=n_frames, seed=seed + 30)
    frames, metadata = binary_adapter.transform_many(binary_events)
    labels = np.asarray([bool(event["is_anomaly"]) for event in binary_events], dtype=bool)
    scenarios.append(("binary_high_entropy_blob", frames, metadata, labels))
    return scenarios


def _run_scenario(
    scenario: str,
    frames: np.ndarray,
    metadata: list[dict[str, Any]],
    labels: np.ndarray,
    include_optional: bool,
) -> list[BenchmarkResult]:
    frames32 = np.asarray(frames, dtype=np.float32)
    raw_data = frames32.tobytes()
    raw_bytes = len(raw_data)
    results = [_run_eidos(scenario, frames, metadata, labels, raw_bytes)]
    results.extend(_compression_baselines(scenario, frames, raw_data, raw_bytes, include_optional=include_optional))
    results.append(_rolling_zscore_baseline(scenario, frames, labels, raw_bytes))
    isolation = _isolation_forest_baseline(scenario, frames, labels, raw_bytes) if include_optional else None
    if isolation is not None:
        results.append(isolation)
    river_result = _river_baseline(scenario, frames, labels, raw_bytes) if include_optional else None
    if river_result is not None:
        results.append(river_result)
    return results


def _run_eidos(
    scenario: str,
    frames: np.ndarray,
    metadata: list[dict[str, Any]],
    labels: np.ndarray,
    raw_bytes: int,
) -> BenchmarkResult:
    annotated_metadata, proxy = _sentinel_proxy_metadata(frames, metadata)
    feature_names = annotated_metadata[0].get("feature_names", []) if annotated_metadata else []
    encoder = ResidualFirstCodec(feature_names=feature_names, source_id=scenario)
    tokens = encoder.encode_stream(frames, annotated_metadata)
    decoder = ResidualFirstCodec(feature_names=feature_names, source_id=scenario)
    reconstructed = decoder.decode_stream(tokens)
    jsonl = tokens_to_jsonl(tokens)
    packed = pack_tokens(tokens, codec="gzip")
    predicted = np.asarray(
        [token["compression_mode"] in {"structured_residual", "anomaly_capsule", "raw_frame_plus_full_context"} for token in tokens],
        dtype=bool,
    )
    metrics = _detection_metrics(predicted, labels)
    normal_ratio, anomaly_ratio = _frame_ratios(tokens, labels, frames.shape[1] * 4)
    status_distribution: dict[str, int] = {}
    for token in tokens:
        status = str(token.get("sentinel_status", "UNKNOWN"))
        status_distribution[status] = status_distribution.get(status, 0) + 1
    surprise = np.asarray([float(token.get("surprise_z", 0.0)) for token in tokens], dtype=np.float64)
    residuals = np.asarray([float(token.get("residual_norm", 0.0)) for token in tokens], dtype=np.float64)
    return BenchmarkResult(
        scenario=scenario,
        method="eidos_residual_codec_gzip",
        raw_bytes=raw_bytes,
        token_bytes=len(jsonl),
        compressed_bytes=len(packed.data),
        compression_ratio=_ratio(raw_bytes, len(packed.data)),
        reconstruction_error=reconstruction_error(frames, reconstructed),
        anomaly_preservation_score=anomaly_preservation_score(tokens, labels.tolist()),
        anomaly_capsule_count=sum(
            1 for token in tokens if token.get("compression_mode") in {"anomaly_capsule", "raw_frame_plus_full_context"}
        ),
        normal_frame_compression_ratio=normal_ratio,
        anomaly_frame_compression_ratio=anomaly_ratio,
        regime_status_distribution=status_distribution,
        surprise_z_mean=float(np.mean(surprise)),
        surprise_z_p95=float(np.percentile(surprise, 95)),
        surprise_z_separation=_surprise_separation(surprise, labels),
        residual_norm_mean=float(np.mean(residuals)),
        residual_norm_p95=float(np.percentile(residuals, 95)),
        compression_collapse_score=_compression_collapse_score(normal_ratio, anomaly_ratio),
        notes=f"Predictive Eidos token stream with Sentinel proxy; proxy warmup={proxy['warmup']}.",
        **metrics,
    )


def _compression_baselines(
    scenario: str,
    frames: np.ndarray,
    raw_data: bytes,
    raw_bytes: int,
    include_optional: bool,
) -> list[BenchmarkResult]:
    rows = [
        _compression_row(scenario, "raw_uncompressed", raw_bytes, raw_bytes, 0.0, "Raw feature bytes."),
        _compression_row(scenario, "gzip_raw_frames", raw_bytes, len(gzip.compress(raw_data)), 0.0, "gzip over raw frames."),
        _compression_row(scenario, "lzma_raw_frames", raw_bytes, len(lzma.compress(raw_data)), 0.0, "lzma over raw frames."),
    ]
    if include_optional:
        try:
            zstd_bytes = len(_zstd_compress(raw_data))
        except OptionalCodecUnavailable:
            zstd_bytes = None
        if zstd_bytes is not None:
            rows.append(_compression_row(scenario, "zstd_raw_frames", raw_bytes, zstd_bytes, 0.0, "zstd over raw frames."))
    encoded, reconstructed = _naive_delta_encode(frames)
    rows.append(
        _compression_row(
            scenario,
            "naive_delta_int16",
            raw_bytes,
            len(encoded),
            reconstruction_error(frames, reconstructed),
            "First frame plus int16 quantized deltas; defensive baseline, not anomaly-aware.",
        )
    )
    return rows


def _compression_row(
    scenario: str,
    method: str,
    raw_bytes: int,
    compressed_bytes: int,
    error: float,
    notes: str,
) -> BenchmarkResult:
    return BenchmarkResult(
        scenario=scenario,
        method=method,
        raw_bytes=raw_bytes,
        token_bytes=None,
        compressed_bytes=compressed_bytes,
        compression_ratio=_ratio(raw_bytes, compressed_bytes),
        reconstruction_error=error,
        anomaly_preservation_score=None,
        anomaly_capsule_count=None,
        normal_frame_compression_ratio=None,
        anomaly_frame_compression_ratio=None,
        precision=None,
        recall=None,
        f1=None,
        false_positives=None,
        false_negatives=None,
        detection_delay=None,
        regime_status_distribution={},
        surprise_z_mean=None,
        surprise_z_p95=None,
        surprise_z_separation=None,
        residual_norm_mean=None,
        residual_norm_p95=None,
        compression_collapse_score=None,
        notes=notes,
    )


def _rolling_zscore_baseline(
    scenario: str,
    frames: np.ndarray,
    labels: np.ndarray,
    raw_bytes: int,
) -> BenchmarkResult:
    predictions, scores = _rolling_zscore_predictions(frames)
    metrics = _detection_metrics(predictions, labels)
    return BenchmarkResult(
        scenario=scenario,
        method="rolling_zscore_detector",
        raw_bytes=raw_bytes,
        token_bytes=None,
        compressed_bytes=raw_bytes,
        compression_ratio=1.0,
        reconstruction_error=0.0,
        anomaly_preservation_score=None,
        anomaly_capsule_count=None,
        normal_frame_compression_ratio=None,
        anomaly_frame_compression_ratio=None,
        regime_status_distribution={},
        surprise_z_mean=float(np.mean(scores)),
        surprise_z_p95=float(np.percentile(scores, 95)),
        surprise_z_separation=_surprise_separation(scores, labels),
        residual_norm_mean=None,
        residual_norm_p95=None,
        compression_collapse_score=None,
        notes="Simple rolling z-score detector over feature energy.",
        **metrics,
    )


def _isolation_forest_baseline(
    scenario: str,
    frames: np.ndarray,
    labels: np.ndarray,
    raw_bytes: int,
) -> BenchmarkResult | None:
    try:
        from sklearn.ensemble import IsolationForest  # type: ignore
    except ImportError:
        return None

    contamination = float(np.clip(np.mean(labels), 0.01, 0.49))
    model = IsolationForest(n_estimators=80, contamination=contamination, random_state=17)
    predictions = model.fit_predict(frames) == -1
    scores = -model.score_samples(frames)
    metrics = _detection_metrics(predictions, labels)
    return BenchmarkResult(
        scenario=scenario,
        method="isolation_forest_detector",
        raw_bytes=raw_bytes,
        token_bytes=None,
        compressed_bytes=raw_bytes,
        compression_ratio=1.0,
        reconstruction_error=0.0,
        anomaly_preservation_score=None,
        anomaly_capsule_count=None,
        normal_frame_compression_ratio=None,
        anomaly_frame_compression_ratio=None,
        regime_status_distribution={},
        surprise_z_mean=float(np.mean(scores)),
        surprise_z_p95=float(np.percentile(scores, 95)),
        surprise_z_separation=_surprise_separation(scores, labels),
        residual_norm_mean=None,
        residual_norm_p95=None,
        compression_collapse_score=None,
        notes="Optional scikit-learn IsolationForest baseline.",
        **metrics,
    )


def _river_baseline(
    scenario: str,
    frames: np.ndarray,
    labels: np.ndarray,
    raw_bytes: int,
) -> BenchmarkResult | None:
    try:
        from river import anomaly  # type: ignore
    except ImportError:
        return None

    model = anomaly.HalfSpaceTrees(seed=23)
    scores: list[float] = []
    for frame in frames:
        sample = {f"f{i}": float(value) for i, value in enumerate(frame)}
        score = float(model.score_one(sample) or 0.0)
        model.learn_one(sample)
        scores.append(score)
    score_arr = np.asarray(scores, dtype=np.float64)
    threshold = float(np.percentile(score_arr, 90))
    predictions = score_arr >= threshold
    metrics = _detection_metrics(predictions, labels)
    return BenchmarkResult(
        scenario=scenario,
        method="river_half_space_trees",
        raw_bytes=raw_bytes,
        token_bytes=None,
        compressed_bytes=raw_bytes,
        compression_ratio=1.0,
        reconstruction_error=0.0,
        anomaly_preservation_score=None,
        anomaly_capsule_count=None,
        normal_frame_compression_ratio=None,
        anomaly_frame_compression_ratio=None,
        regime_status_distribution={},
        surprise_z_mean=float(np.mean(score_arr)),
        surprise_z_p95=float(np.percentile(score_arr, 95)),
        surprise_z_separation=_surprise_separation(score_arr, labels),
        residual_norm_mean=None,
        residual_norm_p95=None,
        compression_collapse_score=None,
        notes="Optional River online anomaly baseline.",
        **metrics,
    )


def _sentinel_proxy_metadata(
    frames: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    warmup: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    energies = [float(np.linalg.norm(frame) / np.sqrt(frame.size)) for frame in frames]
    residuals = [
        0.0
        if idx == 0
        else float(np.linalg.norm(frames[idx] - frames[idx - 1]) / np.sqrt(frames[idx].size))
        for idx in range(len(frames))
    ]
    energy_baseline = energies[:warmup]
    residual_baseline = residuals[1:warmup] or [0.0]
    previous: np.ndarray | None = None
    annotated: list[dict[str, Any]] = []
    for idx, frame in enumerate(frames):
        residual_norm = float(np.linalg.norm(frame - previous) / np.sqrt(frame.size)) if previous is not None else 0.0
        energy = float(np.linalg.norm(frame) / np.sqrt(frame.size))
        z_residual = _baseline_z(residual_norm, residual_baseline)
        z_energy = _baseline_z(energy, energy_baseline)
        surprise = 0.0 if idx < warmup else max(z_residual, z_energy)
        status = _status_from_surprise(surprise)
        meta = dict(metadata[idx])
        meta.update(
            {
                "frame_id": idx,
                "surprise_z": surprise,
                "sentinel_status": status,
                "sentinel_regime": status,
                "sentinel_metrics": {"residual_z": z_residual, "energy_z": z_energy},
            }
        )
        annotated.append(meta)
        previous = np.asarray(frame, dtype=np.float64)
    return annotated, {"warmup": warmup}


def _online_z(value: float, history: Sequence[float]) -> float:
    if len(history) < 4:
        return 0.0
    arr = np.asarray(history[-32:], dtype=np.float64)
    spread = float(np.std(arr))
    if spread <= 1e-9:
        return 0.0
    return max(0.0, (value - float(np.mean(arr))) / spread)


def _baseline_z(value: float, baseline: Sequence[float]) -> float:
    arr = np.asarray(baseline, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    spread = max(float(np.std(arr)), 1e-3)
    return max(0.0, (float(value) - float(np.mean(arr))) / spread)


def _status_from_surprise(z_score: float) -> str:
    if z_score >= 7.0:
        return "RED"
    if z_score >= 4.5:
        return "AMBER"
    if z_score >= 2.5:
        return "VIOLET"
    if z_score >= 1.2:
        return "BLUE"
    return "GREEN"


def _rolling_zscore_predictions(frames: np.ndarray, warmup: int = 8, threshold: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    baseline = [float(np.linalg.norm(frame) / np.sqrt(frame.size)) for frame in frames[:warmup]]
    predictions: list[bool] = []
    scores: list[float] = []
    for idx, frame in enumerate(frames):
        energy = float(np.linalg.norm(frame) / np.sqrt(frame.size))
        z_score = 0.0 if idx < warmup else _baseline_z(energy, baseline)
        scores.append(z_score)
        predictions.append(bool(z_score >= threshold))
    return np.asarray(predictions, dtype=bool), np.asarray(scores, dtype=np.float64)


def _detection_metrics(predictions: Sequence[bool], labels: Sequence[bool]) -> dict[str, Any]:
    pred = np.asarray(predictions, dtype=bool)
    truth = np.asarray(labels, dtype=bool)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = 1.0 if tp + fp == 0 and tp + fn == 0 else (tp / (tp + fp) if tp + fp else 0.0)
    recall = 1.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    delay = _detection_delay(pred, truth)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positives": fp,
        "false_negatives": fn,
        "detection_delay": delay,
    }


def _detection_delay(predictions: np.ndarray, labels: np.ndarray) -> int | None:
    anomaly_indices = np.flatnonzero(labels)
    if anomaly_indices.size == 0:
        return None
    first_anomaly = int(anomaly_indices[0])
    hits = np.flatnonzero(predictions[first_anomaly:])
    if hits.size == 0:
        return None
    return int(hits[0])


def _naive_delta_encode(frames: np.ndarray, scale: float = 1000.0) -> tuple[bytes, np.ndarray]:
    frames32 = np.asarray(frames, dtype=np.float32)
    if frames32.size == 0:
        return b"", frames32.copy()
    deltas = np.diff(frames32, axis=0)
    q = np.clip(np.rint(deltas * scale), -32768, 32767).astype(np.int16)
    encoded = b"".join([frames32[0].tobytes(), q.tobytes()])
    reconstructed = np.empty_like(frames32)
    reconstructed[0] = frames32[0]
    if frames32.shape[0] > 1:
        reconstructed[1:] = reconstructed[0] + np.cumsum(q.astype(np.float32) / scale, axis=0)
    return encoded, reconstructed.astype(np.float64)


def _zstd_compress(data: bytes) -> bytes:
    try:
        import zstandard as zstd  # type: ignore
    except ImportError as exc:
        raise OptionalCodecUnavailable("zstandard not installed") from exc
    return zstd.ZstdCompressor(level=3).compress(data)


def _frame_ratios(tokens: Sequence[Mapping[str, Any]], labels: np.ndarray, raw_bytes_per_frame: int) -> tuple[float | None, float | None]:
    normal_tokens = [token for token, label in zip(tokens, labels) if not label]
    anomaly_tokens = [token for token, label in zip(tokens, labels) if label]
    normal_ratio = _ratio(len(normal_tokens) * raw_bytes_per_frame, len(tokens_to_jsonl(normal_tokens))) if normal_tokens else None
    anomaly_ratio = _ratio(len(anomaly_tokens) * raw_bytes_per_frame, len(tokens_to_jsonl(anomaly_tokens))) if anomaly_tokens else None
    return normal_ratio, anomaly_ratio


def _surprise_separation(scores: np.ndarray, labels: np.ndarray) -> float | None:
    if not np.any(labels) or not np.any(~labels):
        return None
    return float(np.mean(scores[labels]) - np.mean(scores[~labels]))


def _compression_collapse_score(normal_ratio: float | None, anomaly_ratio: float | None) -> float | None:
    if normal_ratio is None or anomaly_ratio is None or normal_ratio <= 0.0:
        return None
    return float(max(0.0, min(1.0, 1.0 - anomaly_ratio / normal_ratio)))


def _ratio(raw_bytes: int, compressed_bytes: int) -> float:
    if compressed_bytes <= 0:
        return 0.0
    return float(raw_bytes / compressed_bytes)


def _write_csv(path: Path, results: Sequence[BenchmarkResult]) -> None:
    rows = [asdict(result) for result in results]
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["regime_status_distribution"] = json.dumps(row["regime_status_distribution"], sort_keys=True)
            writer.writerow(row)


def _markdown_report(results: Sequence[BenchmarkResult]) -> str:
    lines = [
        "# Quantum-Aware Anomaly-Preserving Compression Benchmark",
        "",
        "Synthetic, deterministic local benchmark. Eidos is evaluated as a predictive compression and anomaly-preservation layer, not as a magic quantum detector.",
        "",
        "| Scenario | Method | Compression Ratio | Reconstruction Error | Precision | Recall | F1 | Detection Delay | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.scenario,
                    result.method,
                    _fmt(result.compression_ratio),
                    _fmt(result.reconstruction_error),
                    _fmt(result.precision),
                    _fmt(result.recall),
                    _fmt(result.f1),
                    "-" if result.detection_delay is None else str(result.detection_delay),
                    result.notes.replace("|", "/"),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Plain-Language Conclusion", ""])
    lines.extend(_conclusion_lines(results))
    return "\n".join(lines) + "\n"


def _conclusion_lines(results: Sequence[BenchmarkResult]) -> list[str]:
    by_scenario: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        by_scenario.setdefault(result.scenario, []).append(result)

    eidos_wins: list[str] = []
    conventional_wins: list[str] = []
    preservation: list[float] = []
    false_positives = 0
    for scenario, rows in by_scenario.items():
        eidos = next(row for row in rows if row.method == "eidos_residual_codec_gzip")
        conventional = [row for row in rows if row.method in {"gzip_raw_frames", "lzma_raw_frames", "zstd_raw_frames"}]
        best_conventional = max(conventional, key=lambda row: row.compression_ratio, default=None)
        if best_conventional and eidos.compression_ratio >= best_conventional.compression_ratio:
            eidos_wins.append(scenario)
        elif best_conventional:
            conventional_wins.append(f"{scenario} ({best_conventional.method})")
        if eidos.anomaly_preservation_score is not None:
            preservation.append(eidos.anomaly_preservation_score)
        false_positives += int(eidos.false_positives or 0)

    avg_preservation = float(np.mean(preservation)) if preservation else 1.0
    return [
        f"- Eidos beat conventional gzip/lzma/zstd compression on: {', '.join(eidos_wins) if eidos_wins else 'none in this run'}.",
        f"- Conventional entropy codecs beat Eidos on: {', '.join(conventional_wins) if conventional_wins else 'none in this run'}.",
        f"- Eidos anomaly preservation averaged {_fmt(avg_preservation)} across labeled scenarios by emitting structured residuals or capsules.",
        f"- Eidos false positives across all scenarios: {false_positives}; tune surprise thresholds if this is too high for the deployment domain.",
        "- Next tuning target: calibrate Sentinel proxy thresholds per domain and connect the codec to the live reservoir predictor for lower normal-frame token cost.",
    ]


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.4g}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/benchmarks")
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-optional", action="store_true", help="Skip optional sklearn/River/zstd baselines.")
    args = parser.parse_args(argv)
    run_benchmark(args.output_dir, n_frames=args.frames, seed=args.seed, include_optional=not args.no_optional)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
