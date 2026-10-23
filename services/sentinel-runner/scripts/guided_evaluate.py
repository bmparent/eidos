"""Product engineering evaluation. Never opens research/Grand Proof data.

Run plan, development, freeze, final in that order from the repository root.
Final periods are generated only after a write-once acceptance freeze exists.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.ensemble import IsolationForest
from sentinel_runner.guided.causal import temporal, baseline, initialize_engine
from sentinel_runner.guided.ingestion import parse, plain

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/sentinel-guided-20260908/evaluation"
SCENARIOS = ["benign_repetition", "harmful_repetition", "isolated_spike", "slow_drift", "regime_shift", "noisy_interval", "contaminated_calibration"]
MAPPING = {"features": ["latency", "load"], "labels": ["label"], "timestamp": "timestamp", "timezone": "UTC", "entity": "host", "session": "",
           "units": {"latency": "ms", "load": "%"}, "meaning": "One synthetic service sample per minute", "reference": "first 24 observations of this host",
           "missing": "exclude", "ordering": "sort"}

def write(name, data, exclusive=False):
    path = OUT / name; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as f: json.dump(plain(data), f, indent=2, allow_nan=False)

def digest(data): return hashlib.sha256(json.dumps(plain(data), sort_keys=True).encode()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()

def fixture(seed, scenario):
    rng = np.random.default_rng(seed); n = 180
    t = np.arange(n); latency = 40 + np.sin(t * 2 * np.pi / 12) + rng.normal(0, .35, n)
    load = 50 + np.cos(t * 2 * np.pi / 12) + rng.normal(0, .5, n)
    label = np.zeros(n, dtype=int)
    if scenario == "harmful_repetition":
        for start in [60, 100, 140]: latency[start:start+4] += 30; label[start:start+4] = 1
    if scenario == "isolated_spike": latency[100] += 40; label[100] = 1
    if scenario == "slow_drift": latency[70:120] += np.arange(50) * .3; latency[120:] += 15; label[90:] = 1
    if scenario == "regime_shift": latency[90:] += 12; label[90:] = 1
    if scenario == "noisy_interval": latency[80:120] += rng.normal(0, 6, 40)  # harmless collection noise, retained FP
    if scenario == "contaminated_calibration": latency[8:12] += 20; latency[100:104] += 30; label[100:104] = 1
    return pd.DataFrame({"timestamp": pd.date_range(f"2026-07-{1 + seed % 20:02d}", periods=n, freq="min", tz="UTC").astype(str),
                         "host": f"synthetic-{seed}", "latency": latency, "load": load, "label": label})

def groups(mask):
    groups_out=[]
    for i in np.flatnonzero(mask):
        if groups_out and i == groups_out[-1][-1] + 1: groups_out[-1].append(int(i))
        else: groups_out.append([int(i)])
    return groups_out

def event_metrics(actual, alerts):
    truth = groups(actual); incidents = groups(alerts)
    tp = sum(any(actual[i] for i in g) for g in incidents)
    fp = len(incidents) - tp
    detected = [g for g in truth if any(alerts[i] for i in g)]
    delays = [(next(i for i in g if alerts[i]) - g[0]) * 60 for g in detected]
    return {"rawAlertEvents": int(sum(alerts)), "mergedIncidents": len(incidents), "truePositiveIncidents": tp, "falsePositiveIncidents": fp,
            "truthIncidents": len(truth), "detectedTruthIncidents": len(detected), "incidentRecall": len(detected) / len(truth) if truth else None,
            "incidentPrecision": tp / len(incidents) if incidents else None, "falseAlertsPerOperatingDay": fp / (len(actual) * 60) * 86400,
            "operatingSeconds": len(actual) * 60, "meanDetectionDelaySeconds": float(np.mean(delays)) if delays else None}

def score_baselines(frame):
    x=frame[["latency", "load"]].to_numpy(); center=x[:24].mean(0); scale=np.maximum(x[:24].std(0),1e-6); z=(x-center)/scale
    forest=IsolationForest(n_estimators=100, random_state=42,n_jobs=1).fit(z[:24]); fs=-forest.score_samples(z)
    cutoff=float(np.quantile(fs[:24], .99, method="higher"))
    output={}
    for method in ["persistence", "seasonal_12", "robust_prefix", "isolation_forest_prefix"]:
        errors=[]; absolute=[]; alerts=[]; maes=[]; hits=[]; widths=[]
        # Identical first-24 reference; baseline loss is retrospective preparation.
        for i in range(1,24): errors.append(float(np.linalg.norm(z[i]-z[i-1]))); absolute.append(abs(x[i,0]-x[i-1,0]))
        for i in range(24,len(x)):
            prediction=x[i-12] if method == "seasonal_12" else x[i-1]
            med,sigma=baseline(errors); residual=float(np.linalg.norm((x[i]-prediction)/scale))
            if method == "robust_prefix":
                m=np.median(x[:24],0); s=np.maximum(1.4826*np.median(abs(x[:24]-m),0),1e-6); alert=bool(np.max(abs(x[i]-m)/s)>=5)
            elif method == "isolation_forest_prefix": alert=bool(fs[i]>cutoff)
            else: alert=(residual-med)/sigma>=5
            err=abs(x[i,0]-prediction[0]); radius=float(np.quantile(absolute[-128:],.9,method="higher"))
            alerts.append(alert); maes.append(err); hits.append(err<=radius); widths.append(2*radius)
            if not alert: errors.append(residual); absolute.append(err)
        output[method]={**event_metrics(frame.label.to_numpy()[24:],np.array(alerts)), "forecastMAE":float(np.mean(maes)) if method in ["persistence","seasonal_12"] else None,
                        "intervalCoverage":float(np.mean(hits)) if method in ["persistence","seasonal_12"] else None,
                        "intervalMeanWidth":float(np.mean(widths)) if method in ["persistence","seasonal_12"] else None,
                        "thresholdPolicy":"5 robust standardized units; Isolation Forest prefix 99th percentile", "forecastMethod":method if method in ["persistence","seasonal_12"] else None}
    return output

def targeted_ablation(frame, result, folder):
    """Canonical memory and TraceSeal in observation-only shadow: never suppress product incidents."""
    engine,_=initialize_engine(folder)
    memory=engine.HippocampusHDC(D=256,n_state=64,n_inputs=2,seed=1337,bank_by_regime=False)
    seal=engine.TraceSealProjector(2,torch.device("cpu"),torch.float32,{"trace_seal_enabled":True,"trace_seal_rank":1,"trace_seal_recalc_every":12})
    scores=[]; similarities=[]; history=[]; elapsed=time.perf_counter()
    center=np.asarray(result["groups"][0]["normalization"]["center"]); scale=np.asarray(result["groups"][0]["normalization"]["scale"])
    for event in result["observations"]:
        x=np.array([d["observed"] for d in event["drivers"]]); expected=np.array([d["expected"] for d in event["drivers"]]); residual=torch.tensor((x-expected)/scale,dtype=torch.float32)
        hx=memory.encode_content(torch.tensor((x-center)/scale,dtype=torch.float32))
        # Fixed empty context isolates content familiarity; no detection or safety inference.
        hr=memory.encode_context(torch.zeros(64)); sim,chi=memory.recall_similarity(bank="GLOBAL",h_r=hr,h_x=hx)
        similarities.append({"recordId":event["recordId"],"similarity":sim,"familiarity":chi,"rawAnomalyStillVisible":event["anomaly"]})
        memory.write(bank="GLOBAL",h_r=hr,h_x=hx)
        value=float(seal.score(residual)); med,sigma=baseline(history)
        scores.append((value-med)/sigma>=5 if len(history)>=20 else False)
        if not event["anomaly"]: seal.update(residual); history.append(value)
    write(str(folder.relative_to(OUT)/"memory-traceseal.json"),{"memory":similarities,"traceSealShadowAlerts":scores,"elapsedSeconds":time.perf_counter()-elapsed,
          "policy":"Canonical 256D memory, fixed empty context; TraceSeal rank1/recalc12. Experimental shadow only. Memory on/off preserves every raw decision by construction."})
    return {"memory": {"detectionChanged":False,"writes":memory.write_counts,"mechanism":"canonical HippocampusHDC content recall with no safety suppression"},
            "traceSeal":event_metrics(frame.label.to_numpy()[24:], np.array(scores)), "overheadSeconds":time.perf_counter()-elapsed}

def run(partition, seeds):
    rows=[]; artifacts=[]
    for seed in seeds:
        for scenario in SCENARIOS:
            frame=fixture(seed,scenario); data=frame.to_csv(index=False).encode(); dataset=parse(data,"evaluation.csv")
            folder=OUT/partition/f"{seed}-{scenario}"; folder.mkdir(parents=True,exist_ok=True); (folder/"input.csv").write_bytes(data)
            base=score_baselines(frame)
            for method,metrics in base.items(): rows.append({"partition":partition,"seed":seed,"scenario":scenario,"method":method,**metrics})
            for mechanism in ["none","multiscale","regulation","adapt_all"]:
                target=folder/mechanism; target.mkdir(exist_ok=True)
                result=temporal(dataset,MAPPING,{"horizonSeconds":60,"windowSeconds":60,"target":"latency","mechanism":mechanism},target)
                flags=np.array([o["anomaly"] for o in result["observations"]])
                metrics={**result["metrics"],**event_metrics(frame.label.to_numpy()[24:],flags)}
                rows.append({"partition":partition,"seed":seed,"scenario":scenario,"method":f"eidos_{mechanism}",**metrics})
                write(str((target/"result.json").relative_to(OUT)),result)
                if mechanism=="none": artifacts.append({"seed":seed,"scenario":scenario,**targeted_ablation(frame,result,folder)})
            print(f"{partition} seed={seed} scenario={scenario} complete",flush=True)
    # Seeds are whole-period blocks. Small sample: ranges across periods, no iid-row confidence claim.
    table=pd.DataFrame(rows); table.to_csv(OUT/f"{partition}-metrics.csv",index=False)
    aggregates=[]
    for method,group in table.groupby("method"):
        item={"method":method}
        for metric in ["forecastMAE","incidentRecall","incidentPrecision","falseAlertsPerOperatingDay","intervalCoverage","intervalMeanWidth","meanDetectionDelaySeconds"]:
            values=group.groupby("seed")[metric].mean().dropna()
            item[metric]={"mean":values.mean() if len(values) else None,"minPeriod":values.min() if len(values) else None,"maxPeriod":values.max() if len(values) else None,"independentPeriods":len(values)}
        aggregates.append(item)
    write(f"{partition}-summary.json",{"timestamp":now(),"results":aggregates,"ablations":artifacts,"peakResidentBytes":psutil.Process().memory_info().peak_wset if os.name=="nt" else None,
        "memoryLimitReason":"peak_wset available on Windows; hosted peak measured separately", "costUSD":None,"costReason":"provider billing is not attributed to these local CPU runs",
        "synthetic":True,"researchGatesAdvanced":0,"uncertainty":"ranges of seed-period means; not a population CI; scenarios correlated within a period"})

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=["plan","development","freeze","final"]); args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=="plan":
        write("partition-plan.json",{"created":now(),"development":[73],"validation":[107,109],"final":[211,223,227],"scenarios":SCENARIOS,
             "generatorSha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"sampling":"180 one-minute events per entity/period; first24 calibration; no overlapping entities across partitions",
             "labelPolicy":"Only evaluator reads synthetic event labels; never model selection, normalization, learning or thresholds",
             "prohibitedInputs":"No CICIDS, Grand Proof or sealed research data"},True)
        return
    plan=json.loads((OUT/"partition-plan.json").read_text())
    if plan["generatorSha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest(): raise ValueError("GENERATOR_CHANGED_AFTER_PARTITION_PLAN")
    if args.stage=="development": run("development",plan["development"]); run("validation",plan["validation"])
    if args.stage=="freeze":
        validation=json.loads((OUT/"validation-summary.json").read_text())
        write("acceptance-freeze.json",{"created":now(),"partitionPlanSha256":digest(plan),"validationSha256":digest(validation),
              "criteria":{"forecastMAERatioToPersistenceMax":1.0,"incidentRecallMin":.8,"incidentPrecisionMin":.8,"falseAlertsPerOperatingDayMax":1.0,"intervalCoverageMin":.85},
              "decision":"Default product remains engineering preview. Mechanisms require all criteria to make useful-detection claims; failures remain experimental. No final tuning.",
              "codeHashes":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"services/sentinel-runner/sentinel_runner/guided").glob("*.py")}},True)
    if args.stage=="final":
        freeze=json.loads((OUT/"acceptance-freeze.json").read_text())
        for name,sha in freeze["codeHashes"].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha: raise ValueError("CODE_CHANGED_AFTER_FREEZE")
        if (OUT/"final-summary.json").exists(): raise ValueError("FINAL_ALREADY_CONSUMED: retain original; do not tune and rerun")
        run("final",plan["final"])
        summary=json.loads((OUT/"final-summary.json").read_text()); methods={r["method"]:r for r in summary["results"]}; persistence=methods["persistence"]["forecastMAE"]["mean"]
        decisions=[]
        for method,result in methods.items():
            if not method.startswith("eidos_"): continue
            criteria=freeze["criteria"]; gates={"MAE":result["forecastMAE"]["mean"]<=persistence*criteria["forecastMAERatioToPersistenceMax"],
                "recall":result["incidentRecall"]["mean"]>=criteria["incidentRecallMin"],"precision":result["incidentPrecision"]["mean"]>=criteria["incidentPrecisionMin"],
                "falseAlerts":result["falseAlertsPerOperatingDay"]["mean"]<=criteria["falseAlertsPerOperatingDayMax"],"coverage":result["intervalCoverage"]["mean"]>=criteria["intervalCoverageMin"]}
            decisions.append({"method":method,"gates":gates,"qualified":all(gates.values())})
        write("qualification.json",{"freezeSha256":digest(freeze),"finalSummarySha256":digest(summary),"decisions":decisions,"researchGatesAdvanced":0})

if __name__=="__main__": main()
