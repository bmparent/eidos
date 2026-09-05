"""Data/label leakage and integrity controls; synthetic fixtures are not utility proof."""
import ast
import csv
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from proof.memory_utility_data import FEATURES, ROOT, fit_scaler, label_value, prepare, split_records, transform, verify


def fixture(path, *, timestamps=True, suffix_label="Web Attack - XSS", suffix_value=50):
    header = [*FEATURES, "Label"] + (["Timestamp"] if timestamps else [])
    data = [
        [1_000_000, 1, 1, "BENIGN", "06/07/2017 09:00:00"],
        [1_000_000, 3, 3, "BENIGN", "06/07/2017 09:00:02"],
        [1_000_000, 5, 5, "BENIGN", "06/07/2017 09:00:09"],
        [1_000_000, suffix_value, 50, suffix_label, "06/07/2017 09:00:12"],
        [1_000_000, 70, 70, "BENIGN", "06/07/2017 09:00:14"],
        [1_000_000, 90, 90, "Web Attack - Brute Force", "06/07/2017 09:00:16"],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(data if timestamps else [r[:-1] for r in data])
    return path


def prepared(path, out):
    return prepare(path, out, timestamp_format="%d/%m/%Y %H:%M:%S",
                   cutoff="2017-07-06T09:00:10", gap_seconds=2)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_timestamp_free_dataset_cannot_pass_and_no_overwrite(tmp_path):
    src = fixture(tmp_path / "source.csv", timestamps=False)
    out = tmp_path / "audit"
    result = prepare(src, out)
    assert result["status"] == "blocked"
    assert "missing_or_ambiguous_timestamp_column" in result["blockers"]
    assert not (out / "model_inputs.npz").exists()
    assert verify(out)["status"] == "passed"
    assert read_json(out / "dataset_audit.json")["rows"] == 6
    assert read_json(out / "protocol.json")["acceptable_overhead_threshold"] is None
    with pytest.raises(FileExistsError):
        prepare(src, out)


def test_suffix_values_and_labels_do_not_change_prefix_scaler(tmp_path):
    a = fixture(tmp_path / "a.csv")
    b = fixture(tmp_path / "b.csv", suffix_value=1e8, suffix_label="BENIGN")
    for src, out in ((a, tmp_path / "a"), (b, tmp_path / "b")):
        assert prepared(src, out)["status"] == "prepared"
    assert read_json(tmp_path / "a/scaler.json") == read_json(tmp_path / "b/scaler.json")
    assert read_json(tmp_path / "a/scaler.json")["mean"] == [1_000_000, 2, 2]
    with np.load(tmp_path / "a/model_inputs.npz") as data:
        assert set(data.files) == {"x", "partition", "source_rows"}
        assert data["partition"].tolist() == [0, 0, 1, 2, 2, 2]
        assert data["x"].shape == (6, 3)
    assert read_json(tmp_path / "a/run_manifest.json")["utility_status"] == "untested"


def test_label_permutation_cannot_change_model_inputs(tmp_path):
    for name, label in (("a", "Web Attack - XSS"), ("b", "BENIGN")):
        prepared(fixture(tmp_path / f"{name}.csv", suffix_label=label), tmp_path / name)
    with np.load(tmp_path / "a/model_inputs.npz") as a, np.load(tmp_path / "b/model_inputs.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    with np.load(tmp_path / "a/scoring_labels.npz") as a, np.load(tmp_path / "b/scoring_labels.npz") as b:
        assert not np.array_equal(a["labels"], b["labels"])


@pytest.mark.parametrize("label", ["", "unknown", "BENIGN1", "Web Attack - ransomware", "Web Attack - XSS2"])
def test_unknown_labels_fail_closed(label):
    assert label_value(label) is None


def test_unknown_label_blocks_preparation(tmp_path):
    result = prepared(fixture(tmp_path / "source.csv", suffix_label="unknown"), tmp_path / "out")
    assert "unrecognized_labels" in result["blockers"]


def test_explicit_timestamp_format_and_overlap_gap_required(tmp_path):
    src = fixture(tmp_path / "source.csv")
    result = prepare(src, tmp_path / "noformat", cutoff="2017-07-06T09:00:10", gap_seconds=2)
    assert "explicit_timestamp_format_required" in result["blockers"]
    result = prepare(src, tmp_path / "nogap", timestamp_format="%d/%m/%Y %H:%M:%S",
                     cutoff="2017-07-06T09:00:10", gap_seconds=0.5)
    assert result["status"] == "blocked"
    assert any("maximum observed flow duration" in reason for reason in result["blockers"])


def test_availability_order_reversals_are_sorted_not_source_order(tmp_path):
    src = fixture(tmp_path / "source.csv")
    lines = src.read_text().splitlines()
    src.write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n")
    result = prepared(src, tmp_path / "out")
    assert result["status"] == "prepared"
    assert read_json(tmp_path / "out/dataset_audit.json")["availability_order_reversals"] == 5
    with np.load(tmp_path / "out/model_inputs.npz") as data:
        assert data["source_rows"].tolist() == [5, 4, 3, 2, 1, 0]


def test_duration_changes_availability_and_boundary_ties_stay_together(tmp_path):
    src = fixture(tmp_path / "source.csv")
    text = src.read_text().replace("1000000,3,3", "9000000,3,3")
    src.write_text(text)
    result = prepare(src, tmp_path / "out", timestamp_format="%d/%m/%Y %H:%M:%S",
                     cutoff="2017-07-06T09:00:10", gap_seconds=9)
    assert result["status"] == "blocked"  # long flow is not allowed to train as if available at start


def test_completion_time_order_and_ties_are_explicit():
    def time(second):
        return datetime(2017, 7, 6, 9, 0, second)
    records = [
        (0, [9_000_000, 1, 1], 0, time(0), time(9)),
        (1, [1_000_000, 3, 3], 0, time(1), time(2)),
        (2, [1_000_000, 5, 5], 0, time(9), time(10)),
        (3, [1_000_000, 7, 7], 0, time(9), time(10)),
        (4, [1_000_000, 9, 9], 1, time(20), time(21)),
        (5, [1_000_000, 11, 11], 0, time(20), time(21)),
    ]
    ordered, _, _, partition, _ = split_records(records, time(10), 9, 9)
    assert [r[0] for r in ordered] == [1, 0, 2, 3, 4, 5]
    assert partition.tolist() == [0, 0, 1, 1, 2, 2]


@pytest.mark.parametrize("replacement, blocker", [
    ("bad,1,1", "invalid_flow_duration"),
    ("-1,1,1", "invalid_flow_duration"),
    ("1000000,1,1,extra", "row_width_mismatch"),
])
def test_malformed_data_cannot_be_silently_dropped(tmp_path, replacement, blocker):
    src = fixture(tmp_path / "source.csv")
    src.write_text(src.read_text().replace("1000000,1,1", replacement))
    result = prepared(src, tmp_path / "out")
    assert result["status"] == "blocked"
    assert blocker in result["blockers"]


def test_invalid_time_and_ambiguous_selected_feature_block(tmp_path):
    src = fixture(tmp_path / "source.csv")
    src.write_text(src.read_text().replace("06/07/2017 09:00:00", "not-a-time"))
    assert "invalid_timestamp" in prepared(src, tmp_path / "time")["blockers"]
    src = fixture(tmp_path / "duplicate.csv")
    src.write_text(src.read_text().replace("Total Fwd Packets", "Flow Duration"))
    assert "missing_or_ambiguous_required_columns" in prepared(src, tmp_path / "column")["blockers"]


def test_missing_values_and_zero_variance_are_prefix_only():
    scaler = fit_scaler([[2, 1], [2, np.nan], [2, 3]])
    result = transform([[np.inf, np.nan], [2, 1000]], scaler)
    np.testing.assert_array_equal(result, [[0, 0], [0, 1]])
    with pytest.raises(ValueError, match="no finite"):
        fit_scaler([[np.nan], [np.inf]])


def test_tamper_fails_and_source_read_failure_keeps_manifest(tmp_path):
    src = fixture(tmp_path / "source.csv")
    prepared(src, tmp_path / "out")
    (tmp_path / "out/scaler.json").write_text("{}")
    with pytest.raises(ValueError, match="frozen artifact changed"):
        verify(tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        prepare(tmp_path / "missing.csv", tmp_path / "failed")
    assert read_json(tmp_path / "failed/run_manifest.json")["status"] == "failed"


def test_existing_normalizer_changes_prefix_when_suffix_changes():
    # Execute the repository's exact function body, without importing its heavy CLI.
    path = ROOT / "eidos/tools/run_labeled_domain_proof.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "standardize_matrix")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    fn = namespace["standardize_matrix"]
    assert not np.array_equal(fn(np.array([[0.], [2.], [4.]]))[:2], fn(np.array([[0.], [2.], [100.]]))[:2])


def test_existing_calibration_decision_depends_on_ground_truth_windows():
    path = ROOT / "eidos/proof/sentinel_calibration_v1.py"
    spec = importlib.util.spec_from_file_location("memory_utility_calibration_control", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        event = {"event_id": "same-event", "start_frame": 1, "end_frame": 1, "component_count": 1}
        kwargs = dict(config=module.default_config(enabled=True), raw_labels=["BENIGN"] * 3,
                      proof_labels=["BENIGN"] * 3, kept_events=[], all_confirmed_events=[event])
        absent = module._reason_for_suppression(event, attack_windows=[], **kwargs)
        overlapping = module._reason_for_suppression(event, attack_windows=[{"start_frame": 1, "end_frame": 1}], **kwargs)
        assert absent[0] == "benign_only_pressure"
        assert overlapping == (None, None)
    finally:
        sys.modules.pop(spec.name, None)
