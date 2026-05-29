import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from engine import EidosLifeEngine  # noqa: E402


class EidosLifeEngineSmokeTest(unittest.TestCase):
    def test_snapshot_contains_required_metrics(self):
        engine = EidosLifeEngine(seed=42)
        snapshot = engine.snapshot()
        for key in [
            "generation",
            "alive",
            "density",
            "components",
            "largestComponent",
            "birthCandidates",
            "activeGenomes",
            "activeLineages",
            "aliveEnergyMean",
            "aliveMemoryMean",
            "aliveSignalMean",
            "aliveNutrientMean",
            "aliveWasteMean",
            "aliveStressMean",
            "sentinelRegime",
        ]:
            self.assertIn(key, snapshot["metrics"])
        self.assertEqual(snapshot["width"], 72)
        self.assertEqual(snapshot["height"], 72)
        self.assertEqual(len(snapshot["alive"]), 72 * 72)

    def test_commands_mutate_world_without_core_repo_dependencies(self):
        engine = EidosLifeEngine(seed=42)
        before = engine.snapshot()["alive"][10 * engine.width + 10]
        engine.apply_command({"command": "toggle_cell", "x": 10, "y": 10})
        after = engine.snapshot()["alive"][10 * engine.width + 10]
        self.assertNotEqual(before, after)
        engine.apply_command({"command": "paint_disk", "x": 12, "y": 12, "radius": 2, "mode": "birth"})
        self.assertGreater(engine.snapshot()["metrics"]["alive"], 0)
        engine.apply_command({"command": "inject_pattern", "x": 20, "y": 20, "pattern": "acorn"})
        self.assertTrue(any(event["kind"] == "manual_pattern" for event in engine.snapshot()["events"]))


if __name__ == "__main__":
    unittest.main()
