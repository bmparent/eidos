import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel_runner.dataset import _resolve_downloaded_file, prepare_dataframe
from sentinel_runner.spec import ExperimentSpec
from test_spec import fixture_spec


class DatasetTests(unittest.TestCase):
    def test_requested_file_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "other.csv").write_text("x\n1\n", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "locked Kaggle file"):
                _resolve_downloaded_file(root, "wanted.csv")

    def test_labels_are_absent_from_engine_frames_and_metadata(self):
        rows = 1000
        frame = pd.DataFrame({
            "feature_a": np.arange(rows, dtype=float),
            "feature_b": np.sin(np.arange(rows, dtype=float)),
            "Label": ["BENIGN"] * 700 + ["ATTACK"] * 300,
        })
        spec = ExperimentSpec.from_dict(fixture_spec())
        prepared = prepare_dataframe(frame, spec, file_sha256="a" * 64)
        self.assertNotIn("Label", prepared.feature_columns)
        first_frame, metadata = next(iter(prepared.make_gen_factory()()))
        self.assertEqual(first_frame.shape, (64,))
        self.assertFalse(any(key.lower() in {"label", "attack", "class", "target"} for key in metadata))
        self.assertFalse(prepared.receipt["label_isolation"]["heldout_sent_to_engine"])
        self.assertEqual(prepared.frames.shape[0], 800)
        self.assertEqual(prepared.label_vault.holdout_rows, 200)

    def test_normalization_is_fit_only_on_calibration(self):
        rows = 1000
        feature = np.concatenate((np.arange(200, dtype=float), np.full(800, 10_000.0)))
        frame = pd.DataFrame({"feature": feature, "Label": ["BENIGN"] * rows})
        spec = ExperimentSpec.from_dict(fixture_spec())
        prepared = prepare_dataframe(frame, spec, file_sha256="b" * 64)
        calibration = prepared.frames[:200, 0]
        evaluation = prepared.frames[200:, 0]
        self.assertAlmostEqual(float(np.mean(calibration)), 0.0, places=10)
        self.assertGreater(float(np.mean(evaluation)), 100.0)

    def test_label_like_feature_is_rejected_even_when_explicit(self):
        value = fixture_spec()
        value["dataContract"]["featureColumns"] = ["feature", "attack_target"]
        spec = ExperimentSpec.from_dict(value)
        frame = pd.DataFrame({"feature": range(1000), "attack_target": range(1000), "Label": ["BENIGN"] * 1000})
        with self.assertRaisesRegex(ValueError, "label-like"):
            prepare_dataframe(frame, spec, file_sha256="c" * 64)


if __name__ == "__main__":
    unittest.main()
