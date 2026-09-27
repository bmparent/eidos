import unittest

import numpy as np
import pandas as pd

from sentinel_runner.dataset import prepare_dataframe
from sentinel_runner.metrics import average_precision, evaluate_frozen_predictions
from sentinel_runner.spec import ExperimentSpec
from test_spec import fixture_spec


class MetricsTests(unittest.TestCase):
    def prepared(self):
        frame = pd.DataFrame({"feature": np.arange(1000, dtype=float), "Label": ["BENIGN"] * 500 + ["ATTACK"] * 300 + ["BENIGN"] * 200})
        return prepare_dataframe(frame, ExperimentSpec.from_dict(fixture_spec()), file_sha256="d" * 64)

    def test_ties_cannot_invent_ranking_information(self):
        for labels in ([1, 0], [0, 1]):
            self.assertEqual(average_precision(np.array(labels), np.ones(2)), 0.5)

    def test_evaluation_requires_complete_unique_finite_frozen_predictions(self):
        prepared = self.prepared()
        rows = [{"step": i, "z": 1.0, "z_thresh_eff": 2.0} for i in range(200, 800)]
        cases = [rows[:-1], rows + [rows[0]], rows + [{"step": 800, "z": 1, "z_thresh_eff": 2}],
                 [{**rows[0], "z": float("nan")}, *rows[1:]], [{**rows[0], "step": 200.5}, *rows[1:]],
                 [{**rows[0], "is_surprise": True}, *rows[1:]]]
        for candidate in cases:
            with self.subTest(candidate=candidate[0]):
                with self.assertRaises(RuntimeError):
                    evaluate_frozen_predictions(candidate, prepared)
        forward, trace = evaluate_frozen_predictions(rows, prepared)
        backward, reordered = evaluate_frozen_predictions(list(reversed(rows)), prepared)
        self.assertEqual(forward, backward)
        self.assertEqual(trace, reordered)
        self.assertIsNone(forward["precision"])
        self.assertEqual(forward["average_precision"], 0.5)

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
