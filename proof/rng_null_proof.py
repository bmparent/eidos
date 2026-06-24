"""Proof runner internals for the RNG null-proof harness."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv, hashlib, json, math, platform, statistics, subprocess, sys, zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .rng_null_streams import StreamSpec, suite_streams


@dataclass
class BaselineFrequencyTransitionPredictor:
    """Naive online baseline: predict from local frequencies/transitions only."""
    size: int
    counts: Counter = field(default_factory=Counter)
    transitions: dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    history: list[int] = field(default_factory=list)
    call_log: list[tuple[str, int]] = field(default_factory=list)

    def predict(self) -> int:
        self.call_log.append(("predict", len(self.history)))
        if len(self.history) >= 2:
            last = self.history[-1]
            trans = self.transitions.get(last)
            if trans:
                return int(trans.most_common(1)[0][0])
        if self.counts:
            return int(self.counts.most_common(1)[0][0])
        return 0

    def learn(self, actual: int) -> None:
        self.call_log.append(("learn", len(self.history)))
        if self.history:
            self.transitions[self.history[-1]][actual] += 1
        self.history.append(actual)
        self.counts[actual] += 1


ProofPredictor = BaselineFrequencyTransitionPredictor


class EidosEngineAdapter:
    """Minimal online adapter around the repo's Sentinel confirmation machinery.

    The unified Brain module currently has no lightweight import path without
    optional torch. This adapter therefore keeps prediction discipline in this
    proof harness while feeding every revealed RNG frame through existing Eidos
    Sentinel components after scoring.
    """

    def __init__(self, size: int):
        from eidos.sentinel import EvidenceFrame, SentinelEventConfirmer

        self.size = int(size)
        self.baseline = BaselineFrequencyTransitionPredictor(size)
        self.call_log: list[tuple[str, int]] = []
        self.history: list[int] = []
        self._pending_prediction = 0
        self._compressed = 0
        self._raw = 0
        self._window: list[float] = []
        self._confirmer = SentinelEventConfirmer(mode="balanced")
        self._evidence_cls = EvidenceFrame
        self.last_metrics: dict[str, Any] = {
            "residual": None,
            "error": None,
            "surprise": 0.0,
            "is_surprise": False,
            "sentinel_status": "CALIBRATING",
            "sentinel_regime": "CALIBRATING",
            "compression_ratio": 1.0,
            "codec": "zlib_9_online_window",
            "spectral_entropy": None,
            "spectral_flatness": None,
            "eigen_dominance": None,
            "state_entropy": None,
        }

    @property
    def counts(self):
        return self.baseline.counts

    def predict(self) -> int:
        self.call_log.append(("predict", len(self.history)))
        self._pending_prediction = self.baseline.predict()
        return self._pending_prediction

    def _spectral_features(self) -> tuple[float | None, float | None]:
        if len(self._window) < 32:
            return None, None
        try:
            import numpy as np
            x = np.array(self._window[-128:], dtype=float)
            x = x - x.mean()
            s = abs(np.fft.rfft(x)) ** 2 + 1e-12
            p = s / s.sum()
            ent = -float((p * np.log(p)).sum()) / math.log(len(s))
            flat = float(np.exp(np.log(s).mean()) / (s.mean() + 1e-12))
            return ent, flat
        except Exception:
            return None, None

    def learn(self, actual: int) -> None:
        self.call_log.append(("learn", len(self.history)))
        residual = int(actual) - int(self._pending_prediction)
        prob_seen = (self.baseline.counts[actual] + 1) / (len(self.baseline.history) + self.size)
        surprise = -math.log2(prob_seen)
        is_surprise = surprise > math.log2(self.size) * 0.85
        self._window.append(float(actual))
        if len(self._window) > 128:
            self._window.pop(0)
        self._raw += 1
        payload = bytes(int(v) & 0xff for v in self._window)
        self._compressed = len(zlib.compress(payload, 9))
        compression_ratio = self._compressed / max(1, len(payload))
        spectral_entropy, spectral_flatness = self._spectral_features()
        frame = self._evidence_cls(
            frame=len(self.history),
            residual_score=float(surprise),
            surprise_rate=float(surprise),
            spectral_entropy=spectral_entropy,
            spectral_flatness=spectral_flatness,
            raw_evidence_ref="rng_null_proof",
        )
        before = len(self._confirmer._raw_events)
        self._confirmer.process(frame)
        confirmed_now = len(self._confirmer._raw_events) > before
        sentinel_status = "SURPRISE" if confirmed_now else ("high_entropy" if is_surprise else "stable")
        self.last_metrics = {
            "residual": residual,
            "error": abs(residual),
            "surprise": surprise,
            "is_surprise": is_surprise,
            "sentinel_status": sentinel_status,
            "sentinel_regime": sentinel_status,
            "compression_ratio": compression_ratio,
            "codec": "zlib_9_online_window",
            "spectral_entropy": spectral_entropy,
            "spectral_flatness": spectral_flatness,
            "eigen_dominance": None,
            "state_entropy": None,
        }
        self.baseline.learn(actual)
        self.history.append(actual)


def eidos_adapter_status() -> dict[str, Any]:
    try:
        EidosEngineAdapter(2)
        return {"sentinel_backed": True, "eidos_brain_backed": False, "prediction_backed_by": "baseline_frequency_transition", "eidos_adapter": "proof.rng_null_proof.EidosEngineAdapter"}
    except Exception as exc:
        return {"sentinel_backed": False, "eidos_brain_backed": False, "prediction_backed_by": "baseline_frequency_transition", "eidos_adapter": "proof.rng_null_proof.EidosEngineAdapter", "adapter_error": str(exc)}


def assert_predict_before_reveal(log: list[tuple[str, int]]) -> bool:
    for i in range(0, len(log), 2):
        if i + 1 >= len(log) or log[i][0] != "predict" or log[i + 1][0] != "learn":
            raise AssertionError("predict-then-reveal order violated")
        if log[i][1] != log[i + 1][1]:
            raise AssertionError("history changed before scoring/reveal")
    return True


def clamp_prediction(value: float | int, size: int) -> int:
    return max(0, min(size - 1, int(round(value))))


def binomial_tail(k: int, n: int, p: float) -> float:
    if n <= 0: return 1.0
    mean = n * p; var = n * p * (1.0 - p)
    if var <= 0: return 1.0
    z = (k - mean - 0.5) / math.sqrt(var)
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def entropy(values: list[int]) -> float:
    if not values: return 0.0
    c = Counter(values); n = len(values)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def verdict_for(category: str, acc: float, chance: float, p_value: float, corrected_p: float) -> str:
    if category == "null":
        return "SUSPICIOUS_ABOVE_CHANCE" if acc > chance and corrected_p < 0.01 else "CHANCE_LEVEL_NULL"
    if category == "structured": return "LEARNABLE" if acc > chance * 2.0 else "NOT_LEARNED"
    if category == "bias": return "BIAS_DETECTED" if acc > chance * 1.2 else "BIAS_NOT_DETECTED"
    return "UNKNOWN"


def run_source(spec: StreamSpec, frames: int, warmup: int, out_dir: Path, suite: str) -> dict:
    gen = spec.factory(); eidos = EidosEngineAdapter(spec.size); baseline = eidos.baseline
    rows=[]; actuals=[]; running_freq=Counter(); last=None; baseline_hits=Counter(); baseline_correct=0; digest=hashlib.sha256()
    for step in range(frames):
        pred = clamp_prediction(eidos.predict(), spec.size)
        baseline_pred = pred
        actual = int(next(gen))
        if not 0 <= actual < spec.size: raise ValueError(f"{spec.name} emitted {actual}, outside 0..{spec.size - 1}")
        correct = int(pred == actual); baseline_correct += int(baseline_pred == actual)
        if last is not None and last == actual: baseline_hits["last_value"] += 1
        if running_freq and running_freq.most_common(1)[0][0] == actual:
            baseline_hits["running_frequency"] += 1; baseline_hits["majority_class"] += 1
        eidos.learn(actual); m = eidos.last_metrics
        rows.append({"step":step,"source_name":spec.name,"target_space":spec.target_space,"predicted_value":pred,"actual_value":actual,"correct":correct,"error":m["error"],"residual":m["residual"],"surprise":m["surprise"],"is_surprise":m["is_surprise"],"sentinel_status":m["sentinel_status"],"sentinel_regime":m["sentinel_regime"],"compression_ratio":m["compression_ratio"],"codec":m["codec"],"spectral_entropy":m["spectral_entropy"],"spectral_flatness":m["spectral_flatness"],"eigen_dominance":m["eigen_dominance"],"state_entropy":m["state_entropy"]})
        actuals.append(actual); digest.update(bytes([actual & 0xff])); running_freq[actual]+=1; last=actual
    assert_predict_before_reveal(eidos.call_log)
    eval_rows = rows[warmup:] if warmup < len(rows) else rows; n=len(eval_rows); correct_n=sum(r["correct"] for r in eval_rows); chance=1/spec.size
    data_bytes=bytes(v & 0xff for v in actuals); comp=len(zlib.compress(data_bytes,9))/max(1,len(data_bytes))
    summary={"source_name":spec.name,"category":spec.category,"frames":frames,"warmup":warmup,"target_space":spec.target_space,"target_space_size":spec.size,"adapter_top1_accuracy":correct_n/n if n else 0.0,"baseline_frequency_transition_accuracy":baseline_correct/max(1,frames),"uniform_chance":chance,"last_value":baseline_hits["last_value"]/max(1,frames-1),"running_frequency":baseline_hits["running_frequency"]/max(1,frames-1),"majority_class":baseline_hits["majority_class"]/max(1,frames-1),"chance_top1":chance,"top1_accuracy":correct_n/n if n else 0.0,"top3_accuracy":None,"mean_absolute_error":statistics.fmean(r["error"] for r in eval_rows) if eval_rows else 0.0,"rmSE":math.sqrt(statistics.fmean(r["error"]**2 for r in eval_rows)) if eval_rows else 0.0,"RMSE":math.sqrt(statistics.fmean(r["error"]**2 for r in eval_rows)) if eval_rows else 0.0,"entropy_estimate":entropy(actuals),"surprise_rate":statistics.fmean(r["surprise"] for r in eval_rows) if eval_rows else 0.0,"sentinel_regime_distribution":dict(Counter(r["sentinel_status"] for r in rows)),"compression_ratio":comp,"external_compression_baseline":{"zlib_9_ratio":comp},"baselines":{"uniform_chance":chance,"last_value":baseline_hits["last_value"]/max(1,frames-1),"running_frequency":baseline_hits["running_frequency"]/max(1,frames-1),"majority_class":baseline_hits["majority_class"]/max(1,frames-1),"baseline_frequency_transition":baseline_correct/max(1,frames)},"binomial_p_value":binomial_tail(correct_n,n,chance),"stream_sha256_prefix":digest.hexdigest(),"reproducible":spec.reproducible}
    return {"summary":summary,"rows":rows,"manifest":{"name":spec.name,"algorithm":spec.algorithm,"seed":spec.seed,"reproducible":spec.reproducible,"category":spec.category,"target_space":spec.target_space,"frames":frames,"sha256_prefix":digest.hexdigest()}}


def write_json(path: Path, data) -> None: path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _write_plots(out_dir: Path, rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_dir/"plots"/"README.md").write_text(f"Plots skipped: matplotlib unavailable ({exc}).\n")
        return
    by=defaultdict(list)
    for r in rows: by[r["source_name"]].append(r)
    for filename, field in [("accuracy_over_time.png","correct"),("surprise_over_time.png","surprise"),("compression_over_time.png","compression_ratio")]:
        plt.figure(figsize=(8,4))
        for name, rs in by.items():
            xs=[r["step"] for r in rs]; ys=[float(r[field]) for r in rs]
            if field=="correct":
                acc=[]; c=0
                for i,y in enumerate(ys,1): c+=y; acc.append(c/i)
                ys=acc
            plt.plot(xs, ys, label=name, linewidth=1)
        plt.legend(fontsize=7); plt.title(field); plt.tight_layout(); plt.savefig(out_dir/"plots"/filename); plt.close()


def run_proof(suite: str, seed: int, frames: int, out: str | Path, warmup: int | None = None) -> dict:
    out_dir=Path(out); out_dir.mkdir(parents=True, exist_ok=True); (out_dir/"plots").mkdir(exist_ok=True); (out_dir/"controls").mkdir(exist_ok=True); (out_dir/"null").mkdir(exist_ok=True)
    warmup=min(frames//10,1000) if warmup is None else warmup; specs=suite_streams(suite, seed)
    all_rows=[]; summaries=[]; manifests=[]
    for spec in specs:
        result=run_source(spec, frames, warmup, out_dir, suite); all_rows.extend(result["rows"]); summaries.append(result["summary"]); manifests.append(result["manifest"])
    null_count=sum(1 for s in summaries if s["category"]=="null") or 1; verdicts=[]
    for s in summaries:
        s["multiple_testing_corrected_p"]=min(1.0,s["binomial_p_value"]*null_count) if s["category"]=="null" else s["binomial_p_value"]
        s["verdict"]=verdict_for(s["category"],s["adapter_top1_accuracy"],s["uniform_chance"],s["binomial_p_value"],s["multiple_testing_corrected_p"])
        verdicts.append({"source_name":s["source_name"],"category":s["category"],"verdict":s["verdict"],"adapter_top1_accuracy":s["adapter_top1_accuracy"],"baseline_frequency_transition_accuracy":s["baseline_frequency_transition_accuracy"],"uniform_chance":s["uniform_chance"],"top1_accuracy":s["adapter_top1_accuracy"],"chance_top1":s["uniform_chance"]})
    write_json(out_dir/"config.lock.json",{"suite":suite,"seed":seed,"frames":frames,"warmup":warmup})
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception: commit="unknown"
    (out_dir/"git_commit.txt").write_text(commit+"\n"); (out_dir/"environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\n")
    write_json(out_dir/"rng_manifest.json",manifests)
    fields=["step","source_name","target_space","predicted_value","actual_value","correct","error","residual","surprise","is_surprise","sentinel_status","sentinel_regime","compression_ratio","codec","spectral_entropy","spectral_flatness","eigen_dominance","state_entropy"]
    with (out_dir/"predictions.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(all_rows)
    with (out_dir/"score_summary.csv").open("w",newline="") as f: fields2=list(summaries[0].keys()); w=csv.DictWriter(f,fieldnames=fields2); w.writeheader(); w.writerows(summaries)
    tested=", ".join(s["source_name"] for s in summaries); predictable=", ".join(s["source_name"] for s in summaries if s["verdict"] in {"LEARNABLE","BIAS_DETECTED"}) or "none"; not_predictable=", ".join(s["source_name"] for s in summaries if s["category"]=="null" and s["verdict"]=="CHANCE_LEVEL_NULL") or "none"; suspicious=", ".join(s["source_name"] for s in summaries if s["verdict"]=="SUSPICIOUS_ABOVE_CHANCE") or "none"
    md=["# Eidos RNG Null Proof v1","",f"Suite `{suite}` frames={frames} seed={seed}.","","This run is Sentinel-backed; prediction currently uses the naive online baseline unless `eidos_brain_backed` is true.","","| stream | category | adapter_top1_accuracy | baseline_frequency_transition_accuracy | uniform_chance | last_value | running_frequency | majority_class | verdict |","|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for s in summaries: md.append(f"| {s['source_name']} | {s['category']} | {s['adapter_top1_accuracy']:.4f} | {s['baseline_frequency_transition_accuracy']:.4f} | {s['uniform_chance']:.4f} | {s['last_value']:.4f} | {s['running_frequency']:.4f} | {s['majority_class']:.4f} | {s['verdict']} |")
    md += ["","## Report questions",f"- What streams were tested? {tested}.",f"- Which streams were predictable? {predictable}.",f"- Which streams were not? {not_predictable}.",f"- Did Eidos beat chance on true randomness? Suspicious/null review set: {suspicious}.",f"- Did any result suggest leakage/scoring error? {suspicious}.","- Did Sentinel correctly treat strong randomness as high-entropy/incompressible? See `sentinel_summary.json` and `compression_summary.json`; chance-level null verdicts are the conservative signal.","","## Full pytest note","Full repository pytest may still be blocked by pre-existing environment/data issues: `eidos/test_report.txt` doctest decoding and optional torch-dependent hippocampus/reservoir tests. The RNG proof gate is the focused command documented in this branch."]
    (out_dir/"score_summary.md").write_text("\n".join(md)+"\n")
    write_json(out_dir/"sentinel_summary.json",{s["source_name"]:s["sentinel_regime_distribution"] for s in summaries}); write_json(out_dir/"compression_summary.json",{s["source_name"]:{"compression_ratio":s["compression_ratio"],"baseline":s["external_compression_baseline"]} for s in summaries})
    status=eidos_adapter_status(); overall={"suite":suite,"verdicts":verdicts,"prediction_backed_by":status["prediction_backed_by"],"sentinel_backed":status["sentinel_backed"],"eidos_brain_backed":status["eidos_brain_backed"],"eidos_adapter":status["eidos_adapter"],"core_engine_changed":False,"official_proof_ready":bool(status["eidos_brain_backed"]),"conservative_verdict":"null proof complete; strong randomness is chance-level or suspicious if above chance"}
    if not status["eidos_brain_backed"]: overall["verdict"]="NOT_EIDOS_BRAIN_BACKED"
    if not status["sentinel_backed"]: overall["adapter_error"]=status.get("adapter_error")
    write_json(out_dir/"null_verdict.json",overall); _write_plots(out_dir, all_rows); return overall
