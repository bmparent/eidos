import unittest

import numpy as np
import pandas as pd

from sentinel_runner.dataset import prepare_dataframe
from sentinel_runner.metrics import evaluate_frozen_predictions
from sentinel_runner.spec import ExperimentSpec
from test_spec import fixture_spec


class MetricsTests(unittest.TestCase):
    def test_only_evaluation_rows_are_scored_after_freeze(self):
        rows = 1000
        labels = ["BENIGN"] * 500 + ["ATTACK"] * 300 + ["BENIGN"] * 200
        frame = pd.DataFrame({"feature": np.arange(rows, dtype=float), "Label": labels})
        prepared = prepare_dataframe(frame, ExperimentSpec.from_dict(fixture_spec()), file_sha256="d" * 64)
        step_rows = [
            {"step": index, "z": 4.0 if index >= 500 else 0.0, "z_thresh_eff": 2.0, "status": "GREEN"}
            for index in range(800)
        ]
        metrics, trace = evaluate_frozen_predictions(step_rows, prepared)
        self.assertEqual(metrics["evaluation_rows_scored"], 600)
        self.assertTrue(metrics["labels_unsealed_after_prediction_freeze"])
        self.assertFalse(metrics["heldout_evaluated"])
        self.assertEqual(min(row["step"] for row in trace), 200)
        self.assertEqual(max(row["step"] for row in trace), 799)
        self.assertTrue(all("label" not in row for row in trace))


if __name__ == "__main__":
    unittest.main()
