import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sentinel_runner.engine_bridge import discover_engine_path, load_engine
from sentinel_runner.profiles import EXECUTION_PROFILES, require_profile_capacity


class ProfileTests(unittest.TestCase):
    def test_client_and_runner_profiles_match(self):
        root = Path(__file__).resolve().parents[3]
        uri = (root / "apps/sentinel-lab/lib/experiments/profiles.js").as_uri()
        result = subprocess.check_output(["node", "--input-type=module", "-e", f"import {{ ENGINE_PROFILES }} from {json.dumps(uri)}; console.log(JSON.stringify(ENGINE_PROFILES));"], text=True)
        client = json.loads(result)
        self.assertEqual(set(client), set(EXECUTION_PROFILES))
        for name, profile in EXECUTION_PROFILES.items():
            self.assertEqual(profile, {key: client[name][key] for key in profile})

    def test_full_capacity_is_denied_without_explicit_runner_budget(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "FULL_CAPACITY_NOT_ENABLED"):
                require_profile_capacity("full_capacity")
            require_profile_capacity("cpu_mechanisms")

    def test_actual_multiband_reservoir_preserves_numeric_type(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch unavailable; run the full-engine verification before release")
        with tempfile.TemporaryDirectory() as directory:
            engine = load_engine(discover_engine_path(), Path(directory))
            original = engine.EIDOS_BRAIN_CONFIG.copy()
            dtype = torch.get_default_dtype()
            try:
                engine.EIDOS_BRAIN_CONFIG.update({"fractal_bands": 4, "thermo_enabled": False})
                for precision in (torch.float32, torch.float64):
                    torch.set_default_dtype(precision)
                    reservoir = engine.RLS_Reservoir(4, n_reservoir=16)
                    for _ in range(3):
                        reservoir.listen(torch.ones(4, device=engine.device, dtype=precision))
                        self.assertEqual(reservoir.state.dtype, reservoir.W_out.dtype)
                        self.assertTrue(torch.isfinite(reservoir.W_out @ reservoir.state).all())
            finally:
                engine.EIDOS_BRAIN_CONFIG.clear()
                engine.EIDOS_BRAIN_CONFIG.update(original)
                torch.set_default_dtype(dtype)
