from pathlib import Path

import numpy as np

from eidos_brain.proof.frame_observer import canonical_sha256
from eidos_brain.proof.grand_proof_runner import GrandProofRunner, RunnerConfig, shadow_evaluate
from eidos_brain.proof.grand_proof_scenarios import ScenarioConfig, generate_scenario


def fake_records(scenario):
    rows = []
    for frame_id in range(scenario.config.warmup_frames, scenario.config.total_frames):
        frame = scenario.frames[frame_id]
        pred = np.zeros_like(frame)
        rows.append(
            {
                "frame_id": frame_id,
                "source_id": f"{scenario.scenario_id}:seed:{scenario.seed}",
                "source_range": {"start_frame": frame_id, "end_frame": frame_id},
                "frame": frame.tolist(),
                "best_pred": pred.tolist(),
                "raw_residual": frame.tolist(),
                "normalized_error": float(np.linalg.norm(frame) / np.sqrt(frame.size)),
                "surprise_score": 2.0,
                "surprise_threshold": 1.5,
                "sentinel_metrics": {
                    "is_surprise": False,
                    "eigen_dominance": 0.2,
                    "state_entropy": 0.8,
                    "state_flatness": 0.7,
                    "spectral_entropy": 0.6,
                    "spectral_flatness": 0.5,
                },
                "hdc_metrics": {"similarity": 0.2, "familiarity": 0.2, "write": False},
                "thermodynamic_metrics": {"rho": 1.0, "temperature": 0.0, "energy": 0.0},
                "codec_decision": {"mode": "reference_or_null"},
                "codec_serialized_bytes": 10,
                "source_refs": [{"frame_id": frame_id}],
                "config_hash": "cfg",
                "code_commit": "commit",
                "replay_command": "cmd",
            }
        )
    return rows


def test_label_permutation_cannot_change_online_decisions():
    scenario = generate_scenario("S1_hidden_backdoor", seed=0, config=ScenarioConfig.smoke())
    records = fake_records(scenario)
    left = shadow_evaluate(scenario, records, thresholds={})
    poisoned = generate_scenario("S1_hidden_backdoor", seed=0, config=ScenarioConfig.smoke())
    poisoned.labels[:] = poisoned.labels[::-1]
    right = shadow_evaluate(poisoned, records, thresholds={})
    assert left["decisions"] == right["decisions"]


def test_runner_writes_scenario_system_and_ablation_receipts(tmp_path: Path):
    scenario_config = ScenarioConfig.smoke()

    def capture(scenario, **_kwargs):
        rows = fake_records(scenario)
        return rows, {"status": "COMPLETE", "records": len(rows), "runtime_seconds": 0.01}

    runner = GrandProofRunner(
        RunnerConfig(
            artifact_root=tmp_path,
            repo_root=tmp_path,
            reservoir=32,
            scenario_config=scenario_config,
            code_commit="commit",
            thresholds={},
        ),
        capture_fn=capture,
    )
    result = runner.run(stage="smoke", seeds=[0], scenarios=["S0_nominal"])
    assert result["failures"] == []
    base = tmp_path / "scenarios" / "S0_nominal" / "0"
    assert (base / "shadow_decisions.jsonl").is_file()
    assert (base / "voi_realizations.json").is_file()
    assert (base / "eidos_ms_full" / "metrics.json").is_file()
    assert (base / "A7_no_voi" / "metrics.json").is_file()
    assert (tmp_path / "ablations" / "paired_results.csv").is_file()
    assert (tmp_path / "captures" / "live_frame_observer.jsonl").is_file()
    assert (tmp_path / "captures" / "shadow_tokens.jsonl").is_file()
    assert (tmp_path / "statistics" / "paired_intervals.csv").is_file()
