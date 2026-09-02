import copy
import unittest

from sentinel_runner.spec import ExperimentEnvelope, ExperimentSpec, lock_digest


def fixture_spec():
    return {
        "schema": "eidos.sentinel-lab.experiment.v0.2",
        "evidenceClass": "REAL_DATA_ENGINEERING",
        "dataset": {
            "provider": "kaggle",
            "ref": "dhoogla/cicids2017",
            "version": 3,
            "file": "WebAttacks-Thursday-no-metadata.parquet",
            "expectedSha256": "7db47b2bf97ad58c3556ee25e8e1eb1e697cd391670733833865d0e84d8ed82a",
        },
        "dataContract": {
            "labelColumn": "Label",
            "negativeLabels": ["BENIGN"],
            "orderMode": "source",
            "excludedColumns": [],
            "featureColumns": [],
            "maxRows": 25000,
        },
        "split": {"calibration": 0.2, "evaluation": 0.6, "sealedHoldout": 0.2},
        "engine": {"version": "0.4.7.02", "features": 64, "seed": 0, "configProfile": "cicids_webattacks"},
        "protocol": {
            "labelPolicy": "sealed_until_prediction_freeze",
            "normalization": "calibration_only_zscore",
            "projection": "seeded_gaussian_or_pad",
            "heldoutPolicy": "exclude_from_engineering_run",
            "proofVerdict": "BLOCKED_RESOURCE_BEFORE_HELDOUT",
        },
    }


class SpecTests(unittest.TestCase):
    def test_lock_round_trip_and_tamper_rejection(self):
        spec = ExperimentSpec.from_dict(fixture_spec())
        digest = lock_digest(spec)
        envelope = ExperimentEnvelope.from_dict({"schema": "eidos.sentinel-runner.request.v0.2", "lockDigest": digest, "spec": fixture_spec()})
        self.assertEqual(envelope.lock_digest, digest)

        tampered = fixture_spec()
        tampered["dataContract"]["maxRows"] = 6000
        with self.assertRaisesRegex(ValueError, "RUN_LOCK_MISMATCH"):
            ExperimentEnvelope.from_dict({"schema": "eidos.sentinel-runner.request.v0.2", "lockDigest": digest, "spec": tampered})

    def test_version_and_path_are_fail_closed(self):
        missing_version = fixture_spec()
        missing_version["dataset"].pop("version")
        with self.assertRaises(ValueError):
            ExperimentSpec.from_dict(missing_version)

        traversal = fixture_spec()
        traversal["dataset"]["file"] = "../other.csv"
        with self.assertRaisesRegex(ValueError, "without traversal"):
            ExperimentSpec.from_dict(traversal)

    def test_protocol_locks_cannot_be_relaxed(self):
        value = copy.deepcopy(fixture_spec())
        value["protocol"]["heldoutPolicy"] = "include"
        with self.assertRaisesRegex(ValueError, "safety locks"):
            ExperimentSpec.from_dict(value)


if __name__ == "__main__":
    unittest.main()
