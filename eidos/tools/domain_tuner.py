import argparse
import csv
import importlib.util
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    name: str
    frames: np.ndarray
    labels: Optional[np.ndarray]
    meta: Dict[str, Any]

    def make_gen_factory(self, max_frames: int):
        def _gen():
            limit = min(max_frames, self.frames.shape[0])
            for idx in range(limit):
                meta = {
                    "kind": "row",
                    "dataset": self.name,
                    "row_idx": int(idx),
                }
                yield self.frames[idx], meta

        return _gen


def load_engine_module() -> Any:
    repo_root = Path(__file__).resolve().parents[1]
    engine_path = repo_root / "EIDOS_BRAIN_UNIFIED_v0_4.7.02.py"
    spec = importlib.util.spec_from_file_location("eidos_engine", engine_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load engine at {engine_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_tabular_any(engine: Any, path: str) -> pd.DataFrame:
    if hasattr(engine, "_read_tabular_any"):
        return engine._read_tabular_any(path)
    ext = Path(path).suffix.lower()
    if ext in (".csv", ".tsv", ".tab", ".txt", ".dat"):
        sep = "," if ext == ".csv" else None
        return pd.read_csv(path, sep=sep)
    if ext in (".parquet", ".feather", ".ftr"):
        return pd.read_parquet(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported tabular extension: {ext} ({path})")


def _labels_to_binary(labels: pd.Series) -> np.ndarray:
    if labels.dtype.kind in {"b", "i", "u", "f"}:
        vals = pd.to_numeric(labels, errors="coerce").fillna(0.0)
        return (vals.to_numpy(dtype=float) > 0).astype(int)
    vals = labels.astype(str).str.lower().str.strip()
    negatives = {"0", "false", "no", "normal", "benign", "none"}
    return (~vals.isin(negatives)).astype(int).to_numpy()


def _standardize_numeric(df: pd.DataFrame) -> np.ndarray:
    arr = df.to_numpy(dtype=float)
    mu = np.nanmean(arr, axis=0, keepdims=True)
    sd = np.nanstd(arr, axis=0, keepdims=True)
    sd[sd == 0.0] = 1.0
    arr = (arr - mu) / sd
    return np.nan_to_num(arr, nan=0.0)


def _project_rows(engine: Any, arr: np.ndarray, features: int, seed: int) -> np.ndarray:
    proj = engine.AutoProjector(features, seed=seed)
    projected = np.zeros((arr.shape[0], features), dtype=np.float64)
    for idx in range(arr.shape[0]):
        projected[idx] = proj.to_dim(arr[idx])
    return projected


def _resolve_kaggle_path(dataset_id: str, file_name: Optional[str]) -> str:
    try:
        import kagglehub  # type: ignore
    except ImportError as exc:
        raise ImportError("kagglehub is required for kaggle specs. Install with `pip install kagglehub`.") from exc
    root = kagglehub.dataset_download(dataset_id)
    if file_name:
        candidate = Path(root) / file_name
        if candidate.exists():
            return str(candidate)
        for path in Path(root).rglob("*"):
            if path.is_file() and path.name.lower() == file_name.lower():
                return str(path)
    for ext in (".csv", ".tsv", ".tab", ".parquet", ".feather", ".ftr", ".xlsx", ".xls", ".jsonl", ".ndjson"):
        for path in Path(root).rglob(f"*{ext}"):
            return str(path)
    raise FileNotFoundError(f"No tabular data found under kaggle dataset: {dataset_id}")


def load_dataset_stream(spec: Dict[str, Any], *, engine: Any, features: int) -> Tuple[Any, int, Dict[str, Any]]:
    kind = spec.get("kind", "local").lower()
    label_col = spec.get("label_col")
    seed = int(spec.get("seed", 123))

    if kind == "synthetic":
        steps = int(spec.get("steps", 2000))
        frames = np.stack(list(engine.synthetic_scenario(steps, features=features)), axis=0)
        labels = None
        name = spec.get("name", "synthetic")
    else:
        if kind == "kaggle":
            dataset_id = spec["id"]
            file_name = spec.get("file")
            data_path = _resolve_kaggle_path(dataset_id, file_name)
            name = spec.get("name", dataset_id)
        elif kind == "local":
            data_path = spec["path"]
            name = spec.get("name", Path(data_path).stem)
        else:
            raise ValueError(f"Unknown spec kind: {kind}")

        df = _read_tabular_any(engine, data_path)
        labels = None
        if label_col and label_col in df.columns:
            labels = _labels_to_binary(df[label_col])
            df = df.drop(columns=[label_col])

        numeric_df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        if numeric_df.shape[1] == 0:
            raise ValueError(f"No numeric columns found in {name}")

        standardized = _standardize_numeric(numeric_df)
        frames = _project_rows(engine, standardized, features, seed=seed)

    est_frames = int(frames.shape[0])
    bundle = DatasetBundle(name=name, frames=frames, labels=labels, meta={"kind": kind})
    gen_factory = bundle.make_gen_factory(est_frames)
    meta = {"labels": labels, "frames": frames, "bundle": bundle}
    return gen_factory, est_frames, meta


def _auc_roc(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    y = y_true[order]
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.arange(1, len(y) + 1)
    rank_sum = np.sum(ranks[y == 1])
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _auc_pr(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    y = y_true[order]
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    total_pos = np.sum(y_true == 1)
    if total_pos == 0:
        return 0.0
    for label in y:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / total_pos)
    area = 0.0
    prev_recall = 0.0
    for p, r in zip(precisions, recalls):
        area += p * (r - prev_recall)
        prev_recall = r
    return float(area)


def _burst_max(statuses: List[str], prefix: str) -> int:
    best = 0
    current = 0
    for status in statuses:
        if status.startswith(prefix):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _detection_delay(labels: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> Tuple[float, float]:
    delays = []
    lengths = []
    in_window = False
    start_idx = 0
    for idx, label in enumerate(labels):
        if label == 1 and not in_window:
            in_window = True
            start_idx = idx
        if label == 0 and in_window:
            end_idx = idx
            window_scores = scores[start_idx:end_idx]
            window_thresh = thresholds[start_idx:end_idx]
            detected = np.where(window_scores >= window_thresh)[0]
            delay = detected[0] if detected.size else end_idx - start_idx
            delays.append(delay)
            lengths.append(end_idx - start_idx)
            in_window = False
    if in_window:
        end_idx = len(labels)
        window_scores = scores[start_idx:end_idx]
        window_thresh = thresholds[start_idx:end_idx]
        detected = np.where(window_scores >= window_thresh)[0]
        delay = detected[0] if detected.size else end_idx - start_idx
        delays.append(delay)
        lengths.append(end_idx - start_idx)
    if not delays:
        return 0.0, 0.0
    return float(np.mean(delays)), float(np.mean(lengths))


def evaluate_run(results: Dict[str, Any], labels: Optional[np.ndarray], domain_name: str) -> Dict[str, Any]:
    summary = results.get("summary") or {}
    step_rows = results.get("step_rows") or []
    z_scores = np.array([row.get("z", 0.0) for row in step_rows], dtype=float)
    statuses = [row.get("status", "") for row in step_rows]
    thresholds = np.array([row.get("z_thresh_eff", 0.0) for row in step_rows], dtype=float)

    surprise_rate = summary.get("surprise_rate", 0.0)
    surprise_rate = surprise_rate / 100.0 if surprise_rate > 1.0 else float(surprise_rate)
    z_var = float(np.var(z_scores)) if z_scores.size else 0.0
    red_rate = sum(1 for status in statuses if status.startswith("RED")) / max(len(statuses), 1)
    max_red_burst = _burst_max(statuses, "RED")

    metrics: Dict[str, Any] = {
        "surprise_rate": surprise_rate,
        "z_var": z_var,
        "red_rate": red_rate,
        "max_red_burst": max_red_burst,
        "final_z_thresh": summary.get("final_z_thresh"),
        "final_sigma": summary.get("final_sigma"),
    }

    recall = pr_auc = roc_auc = fpr = delay = delay_norm = 0.0
    if labels is not None and len(labels) == len(z_scores) and len(z_scores) > 0:
        labels_arr = labels.astype(int)
        roc_auc = _auc_roc(labels_arr, z_scores)
        pr_auc = _auc_pr(labels_arr, z_scores)
        preds = (z_scores >= thresholds).astype(int)
        tp = int(np.sum((preds == 1) & (labels_arr == 1)))
        fp = int(np.sum((preds == 1) & (labels_arr == 0)))
        fn = int(np.sum((preds == 0) & (labels_arr == 1)))
        tn = int(np.sum((preds == 0) & (labels_arr == 0)))
        recall = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        delay, avg_len = _detection_delay(labels_arr, z_scores, thresholds)
        delay_norm = delay / max(avg_len, 1.0) if avg_len else 0.0

    metrics.update(
        {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "recall": recall,
            "fpr": fpr,
            "mean_delay": delay,
            "delay_norm": delay_norm,
        }
    )

    z_var_norm = min(1.0, z_var / 5.0)
    stable_score = 1.0 - z_var_norm
    safety_score = 1.0 - min(1.0, max_red_burst / max(len(step_rows) * 0.1, 1.0))

    domain = domain_name.lower()
    has_labels = labels is not None and len(labels) == len(z_scores) and len(z_scores) > 0
    if domain == "cyber":
        if has_labels:
            score = (
                0.45 * recall
                + 0.2 * pr_auc
                + 0.1 * roc_auc
                + 0.2 * (1.0 - delay_norm)
                + 0.05 * (1.0 - fpr)
            )
        else:
            score = 0.5 * surprise_rate + 0.3 * (1.0 - red_rate) + 0.2 * stable_score
    elif domain == "finance":
        surprise_target = 0.05
        surprise_score = 1.0 - min(1.0, abs(surprise_rate - surprise_target) / surprise_target)
        if has_labels:
            score = (
                0.35 * (1.0 - fpr)
                + 0.25 * stable_score
                + 0.2 * (1.0 - red_rate)
                + 0.2 * surprise_score
            )
        else:
            score = 0.5 * stable_score + 0.3 * (1.0 - red_rate) + 0.2 * surprise_score
    else:
        if has_labels:
            score = (
                0.35 * recall
                + 0.2 * pr_auc
                + 0.15 * (1.0 - delay_norm)
                + 0.15 * stable_score
                + 0.15 * safety_score
            )
        else:
            score = 0.4 * surprise_rate + 0.3 * stable_score + 0.3 * safety_score

    metrics["score"] = float(score)
    return metrics


def search_space(domain_name: str) -> Dict[str, Any]:
    domain = domain_name.lower()
    if domain == "cyber":
        sigma_k = (1.0, 2.5)
        target_surprise = (0.08, 0.30)
        ema_alpha = (5e-4, 5e-3)
        leak_rate = (0.10, 0.60)
        forgetting = (0.95, 0.995)
    elif domain == "finance":
        sigma_k = (1.5, 4.5)
        target_surprise = (0.01, 0.10)
        ema_alpha = (1e-4, 2e-3)
        leak_rate = (0.01, 0.20)
        forgetting = (0.985, 0.9998)
    else:
        sigma_k = (1.0, 3.5)
        target_surprise = (0.03, 0.20)
        ema_alpha = (5e-4, 5e-3)
        leak_rate = (0.01, 0.12)
        forgetting = (0.99, 0.9999)

    return {
        "sigma_k": sigma_k,
        "target_surprise": target_surprise,
        "ema_alpha": ema_alpha,
        "warmup_cap": {"min": 250, "max": 4000, "int": True},
        "spectral_radius": (0.7, 1.8),
        "leak_rate": leak_rate,
        "input_scaling": (0.1, 0.8),
        "forgetting": forgetting,
        "weight_decay": {"min": 1e-5, "max": 5e-3, "log": True},
        "reservoir": {"min": 800, "max": 3000, "int": True},
        "hippocampus_sim_theta": (0.10, 0.60),
        "hippocampus_write_z_thresh": (1.0, 3.5),
        "hippocampus_freeze_strength": (0.2, 0.95),
        "hippocampus_compute_on_surprise_only": [True, False],
    }


def _sample_value(rng: np.random.RandomState, spec: Any) -> Any:
    if isinstance(spec, list):
        return spec[int(rng.randint(0, len(spec)))]
    if isinstance(spec, dict):
        low = float(spec["min"])
        high = float(spec["max"])
        if spec.get("log"):
            value = math.exp(rng.uniform(math.log(low), math.log(high)))
        else:
            value = rng.uniform(low, high)
        if spec.get("int"):
            return int(round(value))
        return float(value)
    low, high = spec
    return float(rng.uniform(low, high))


def random_configs(space: Dict[str, Any], count: int, seed: int) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    configs = []
    for _ in range(count):
        cfg = {key: _sample_value(rng, spec) for key, spec in space.items()}
        configs.append(cfg)
    return configs


def evaluate_config(
    engine: Any,
    cfg: Dict[str, Any],
    datasets: List[DatasetBundle],
    features: int,
    fraction: float,
    seed: int,
    domain: str,
) -> Tuple[float, Dict[str, Any]]:
    scores = []
    detail_rows = []
    for bundle in datasets:
        max_frames = max(1, int(bundle.frames.shape[0] * fraction))
        gen_factory = bundle.make_gen_factory(max_frames)
        results = engine.run_stream_once(
            gen_factory,
            est_frames=max_frames,
            features=features,
            profile_label=f"{domain}_tune",
            session_label=f"tune_{bundle.name}",
            cfg_overrides=cfg,
            return_step_rows=True,
            seed=seed,
        )
        metrics = evaluate_run(results, bundle.labels[:max_frames] if bundle.labels is not None else None, domain)
        metrics["dataset"] = bundle.name
        scores.append(metrics["score"])
        detail_rows.append(metrics)
    return float(np.mean(scores)), {"per_dataset": detail_rows}


def successive_halving(
    engine: Any,
    configs: List[Dict[str, Any]],
    datasets: List[DatasetBundle],
    features: int,
    domain: str,
    trials_csv: Path,
    seeds: List[int],
) -> Dict[str, Any]:
    stages = [0.2, 0.5, 1.0]
    remaining = configs
    trial_id = 0

    with trials_csv.open("w", newline="", encoding="utf-8") as f:
        writer = None
        for stage_idx, fraction in enumerate(stages):
            stage_scores = []
            for cfg in remaining:
                seed = seeds[0]
                score, detail = evaluate_config(
                    engine, cfg, datasets, features, fraction, seed, domain
                )
                for row in detail["per_dataset"]:
                    row_out = {
                        "trial_id": trial_id,
                        "stage": stage_idx,
                        "fraction": fraction,
                        "seed": seed,
                        "score": score,
                        **row,
                        **cfg,
                    }
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row_out.keys()))
                        writer.writeheader()
                    writer.writerow(row_out)
                stage_scores.append((score, cfg))
                trial_id += 1
            stage_scores.sort(key=lambda x: x[0], reverse=True)
            keep = max(1, int(math.ceil(len(stage_scores) * 0.5)))
            remaining = [cfg for _, cfg in stage_scores[:keep]]

        top_k = min(5, len(remaining))
        final_candidates = remaining[:top_k]
        best_cfg = None
        best_median = -float("inf")
        for cfg in final_candidates:
            seed_scores = []
            for seed in seeds:
                score, detail = evaluate_config(
                    engine, cfg, datasets, features, 1.0, seed, domain
                )
                for row in detail["per_dataset"]:
                    row_out = {
                        "trial_id": trial_id,
                        "stage": len(stages),
                        "fraction": 1.0,
                        "seed": seed,
                        "score": score,
                        **row,
                        **cfg,
                    }
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row_out.keys()))
                        writer.writeheader()
                    writer.writerow(row_out)
                seed_scores.append(score)
                trial_id += 1
            median_score = float(np.median(seed_scores))
            if median_score > best_median:
                best_median = median_score
                best_cfg = cfg

    return {"best_config": best_cfg, "best_score": best_median}


def _config_deltas(base: Dict[str, Any], tuned: Dict[str, Any]) -> Dict[str, Any]:
    deltas = {}
    for key, value in tuned.items():
        if key not in base or base[key] != value:
            deltas[key] = {"base": base.get(key), "tuned": value}
    return deltas


def _ablation_importance(
    engine: Any,
    best_cfg: Dict[str, Any],
    base_cfg: Dict[str, Any],
    datasets: List[DatasetBundle],
    features: int,
    domain: str,
) -> List[Tuple[str, float]]:
    base_score, _ = evaluate_config(
        engine, best_cfg, datasets, features, 0.5, seed=0, domain=domain
    )
    drops = []
    for key in best_cfg:
        ablated = dict(best_cfg)
        if key in base_cfg:
            ablated[key] = base_cfg[key]
        score, _ = evaluate_config(
            engine, ablated, datasets, features, 0.5, seed=0, domain=domain
        )
        drops.append((key, base_score - score))
    drops.sort(key=lambda x: x[1], reverse=True)
    return drops


def _write_report(
    report_path: Path,
    domain: str,
    best_cfg: Dict[str, Any],
    base_cfg: Dict[str, Any],
    best_score: float,
    summary_metrics: Dict[str, Any],
    importance: List[Tuple[str, float]],
) -> None:
    deltas = _config_deltas(base_cfg, best_cfg)
    lines = [
        f"# Domain tuning report: {domain}",
        "",
        f"Best objective score: **{best_score:.4f}**",
        "",
        "## Summary metrics",
    ]
    for key, value in summary_metrics.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append("## Config deltas")
    for key, val in deltas.items():
        lines.append(f"- `{key}`: {val['base']} → {val['tuned']}")
    lines.append("")
    lines.append("## Parameter importance (ablation)")
    for key, drop in importance[:8]:
        lines.append(f"- `{key}`: score drop {drop:.4f}")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = parser.parse_args()

    engine = load_engine_module()
    spec_path = Path(args.spec)
    spec_obj = json.loads(spec_path.read_text(encoding="utf-8"))
    dataset_specs = spec_obj["datasets"] if isinstance(spec_obj, dict) else spec_obj

    datasets = []
    for dataset_spec in dataset_specs:
        gen_factory, est_frames, meta = load_dataset_stream(
            dataset_spec, engine=engine, features=args.features
        )
        bundle = meta["bundle"]
        datasets.append(bundle)
        _ = gen_factory, est_frames

    space = search_space(args.domain)
    configs = random_configs(space, args.trials, seed=args.seed)

    artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    trials_csv = artifacts_dir / f"trials_{args.domain}.csv"
    best_profile_path = artifacts_dir / f"best_profile_{args.domain}.json"
    report_path = artifacts_dir / f"recommendation_{args.domain}.md"

    results = successive_halving(
        engine,
        configs,
        datasets,
        args.features,
        args.domain,
        trials_csv,
        args.seeds,
    )
    best_cfg = results["best_config"]
    best_score = results["best_score"]

    best_profile_path.write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")

    final_score, detail = evaluate_config(
        engine, best_cfg, datasets, args.features, 1.0, seed=args.seeds[0], domain=args.domain
    )
    summary_metrics = detail["per_dataset"][0]

    base_cfg = dict(engine.EIDOS_BRAIN_CONFIG)
    importance = _ablation_importance(
        engine, best_cfg, base_cfg, datasets, args.features, args.domain
    )
    _write_report(report_path, args.domain, best_cfg, base_cfg, final_score, summary_metrics, importance)

    deltas = _config_deltas(base_cfg, best_cfg)
    print("Best config deltas:")
    for key, val in deltas.items():
        print(f"- {key}: {val['base']} -> {val['tuned']}")
    print(f"Objective score: {best_score:.4f}")
    if summary_metrics:
        print(
            f"Summary: final_z_thresh={summary_metrics.get('final_z_thresh')} "
            f"surprise_rate={summary_metrics.get('surprise_rate')}"
        )


if __name__ == "__main__":
    main()
