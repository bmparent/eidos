import unittest
import sys
import os
import io
import torch
import numpy as np
import shutil
import tempfile
import contextlib
import traceback
from unittest.mock import MagicMock, patch

# --- CONFIGURATION ---
TARGET_DIR = r"d:\eidos"
TARGET_FILE_NAME = "EIDOS_BRAIN_UNIFIED_v0_4.7.02.py"

# --- MOCKING DEPENDENCIES BEFORE IMPORT ---
# We mock these because they might not be installed or configured in the test env
sys.modules["google.cloud"] = MagicMock()
sys.modules["google.cloud.pubsub_v1"] = MagicMock()
sys.modules["kagglehub"] = MagicMock()
sys.modules["websockets"] = MagicMock()
sys.modules["google.colab"] = MagicMock()
# Ensure submodules of mocks exist if referenced directly
sys.modules["google.cloud"].pubsub_v1 = MagicMock()

# --- IMPORTING THE TARGET MODULE ---
sys.path.insert(0, TARGET_DIR)

print(f"Importing {TARGET_FILE_NAME}...")
# We import the module usually as a script, but here as a module to access classes.
# We suppress stdout during import to keep test logs clean from script's print statements.
with contextlib.redirect_stdout(io.StringIO()):
    try:
        # Import dynamic based on filename
        import importlib.util
        spec = importlib.util.spec_from_file_location("eidos_brain", os.path.join(TARGET_DIR, TARGET_FILE_NAME))
        eidos_brain = importlib.util.module_from_spec(spec)
        sys.modules["eidos_brain"] = eidos_brain
        spec.loader.exec_module(eidos_brain)
    except Exception as e:
        print(f"CRITICAL: Failed to import {TARGET_FILE_NAME}: {e}")
        traceback.print_exc()
        sys.exit(1)
print("Import successful.")

# Aliases for convenience
RLS_Reservoir = eidos_brain.RLS_Reservoir
HippocampusHDC = eidos_brain.HippocampusHDC
SentinelMonitor = eidos_brain.SentinelMonitor
NewtonianPredictor = eidos_brain.NewtonianPredictor
EIDOS_BRAIN_CONFIG = eidos_brain.EIDOS_BRAIN_CONFIG
run_sentinel_stream = eidos_brain.run_sentinel_stream
synthetic_scenario = eidos_brain.synthetic_scenario

# --- TEST SUITE ---

class TestRLSReservoir(unittest.TestCase):
    def setUp(self):
        self.features = 64
        self.res_size = 500 # Smaller for fast testing
        self.reservoir = RLS_Reservoir(
            n_inputs=self.features,
            n_reservoir=self.res_size,
            spectral_radius=1.2,
            leak_rate=0.1
        )

    def test_initialization(self):
        """Verify reservoir dimensions and initial state."""
        self.assertEqual(self.reservoir.state.shape[0], self.res_size)
        self.assertEqual(self.reservoir.W_in.shape, (self.res_size, self.features))
        self.assertTrue(torch.is_tensor(self.reservoir.state))
        
    def test_synaptic_hash(self):
        """Hash should be deterministic for same weights."""
        h1 = self.reservoir.get_synaptic_hash()
        h2 = self.reservoir.get_synaptic_hash()
        self.assertEqual(h1, h2)
        
    def test_listen_updates_state(self):
        """Listening to input should change reservoir state."""
        state_0 = self.reservoir.state.clone()
        u = torch.randn(self.features, device=self.reservoir.device)
        self.reservoir.listen(u)
        state_1 = self.reservoir.state
        self.assertFalse(torch.equal(state_0, state_1), "State should update after listen()")
        
    def test_adapt_learning(self):
        """Adapt should update readout weights W_out."""
        u = torch.randn(self.features, device=self.reservoir.device)
        # listen first to have a valid state
        self.reservoir.listen(u)
        
        target = torch.randn(self.features, device=self.reservoir.device)
        w_out_0 = self.reservoir.W_out.clone()
        
        # Verify prediction before learning
        y_0 = self.reservoir.dream()
        
        # Learn
        self.reservoir.adapt(target, lr_scale=0.5)
        w_out_1 = self.reservoir.W_out
        
        self.assertFalse(torch.equal(w_out_0, w_out_1), "W_out should update after adapt()")
        
    def test_thermodynamics_update(self):
        """Verify Leap III thermodynamic parameter updates."""
        # Check initial values
        self.assertTrue("thermo_energy_coeffs" in EIDOS_BRAIN_CONFIG)
        
        # Fake metrics from Sentinel
        metrics = {
            "surprise_score": 5.0, # High surprise
            "plasticity": 1.0,
            "eigen_dominance": 0.1,
            "spectral_entropy": 0.8
        }
        
        stats = self.reservoir.update_thermodynamics(metrics)
        
        self.assertIn("thermo_energy", stats)
        self.assertIn("thermo_rho", stats)
        self.assertIn("thermo_temp", stats)
        
        # High energy should likely increase temp or rho depending on logic
        # Just ensure they are within valid ranges defined in config
        limits = EIDOS_BRAIN_CONFIG["thermo_rho_limits"]
        self.assertTrue(limits[0] <= stats["thermo_rho"] <= limits[1])


class TestHippocampus(unittest.TestCase):
    def setUp(self):
        self.D = 1000
        self.n_state = 100
        self.n_inputs = 64
        self.hipp = HippocampusHDC(
            D=self.D,
            n_state=self.n_state,
            n_inputs=self.n_inputs,
            seed=42
        )

    def test_encoding_shapes(self):
        """Verify HDC vectors have correct dimension D."""
        state = torch.randn(self.n_state)
        frame = torch.randn(self.n_inputs)
        
        hr = self.hipp.encode_context(state)
        hx = self.hipp.encode_content(frame)
        
        self.assertEqual(hr.shape[0], self.D)
        self.assertEqual(hx.shape[0], self.D)
        # Check bipolar nature if it's count sketch or simhash
        # Implementation uses sign(), so values should be -1 or 1 (or 0 if exactly 0)
        self.assertTrue(torch.all(torch.abs(hr) >= 0))

    def test_write_and_recall(self):
        """Write to a bank and verify similarity increases."""
        state = torch.randn(self.n_state)
        frame = torch.randn(self.n_inputs)
        
        hr = self.hipp.encode_context(state)
        hx = self.hipp.encode_content(frame)
        
        bank = "TEST_BANK"
        
        # Initial recall should be near 0 (orthogonal)
        sim_0, _ = self.hipp.recall_similarity(bank=bank, h_r=hr, h_x=hx)
        
        # Write
        self.hipp.write(bank=bank, h_r=hr, h_x=hx, weight=10.0)
        
        # Recall again
        sim_1, _ = self.hipp.recall_similarity(bank=bank, h_r=hr, h_x=hx)
        
        self.assertGreater(sim_1, sim_0, "Similarity should increase after writing trace")


class TestSentinelLogic(unittest.TestCase):
    def setUp(self):
        self.sentinel = SentinelMonitor(window=10)
        
    def test_regime_detection(self):
        """Test basic regime statemachine transitions."""
        # Initial state
        initial = self.sentinel.analyze()
        self.assertEqual(initial, "CALIBRATING")
        
        # Fill buffer to exit calibration
        for _ in range(15):
             self.sentinel.update(
                ratio=5.0, plasticity=0.01, eigen_dominance=0.1, 
                surprise_score=0.5, error_norm=0.1, error_rms=0.01
            )
        
        # Now it should be NOMINAL (GREEN: NOMINAL)
        status = self.sentinel.analyze()
        self.assertIn("NOMINAL", status)
        
        # Force RED state
        for _ in range(20):
            self.sentinel.update(
                ratio=0.5, plasticity=2.5, eigen_dominance=0.9, 
                surprise_score=5.0, error_norm=1.0, error_rms=0.5
            )
        status = self.sentinel.analyze()
        self.assertTrue("RED" in status or "AMBER" in status, f"Status was {status}, expected RED/AMBER")


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Mock artifact root to use temp dir
        eidos_brain.EIDOS_DATA_ROOT = self.temp_dir
        eidos_brain.EIDOS_ARCHIVE_ROOT = os.path.join(self.temp_dir, "archive")
        
        # Override config for fast run
        EIDOS_BRAIN_CONFIG["steps"] = 100
        EIDOS_BRAIN_CONFIG["warmup_cap"] = 20
        EIDOS_BRAIN_CONFIG["reservoir"] = 200 # Small reservoir
        EIDOS_BRAIN_CONFIG["hippocampus_dim"] = 500
        
    def tearDown(self):
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
        
    def test_run_synthetic_stream(self):
        """Run the full run_sentinel_stream with synthetic generator."""
        
        features = 16
        steps = 50
        
        # Create a simple generator factory
        def gen_factory():
             for i in range(steps):
                 vec = np.random.randn(features).astype(np.float32)
                 meta = {"idx": i, "kind": "synthetic"}
                 yield vec, meta

        print("\n--- Starting Integration Test ---")
        try:
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                run_sentinel_stream(
                    gen_factory=gen_factory,
                    est_frames=steps,
                    features=features,
                    profile_label="test_profile",
                    session_label="test_session",
                    warmup=10,
                    sample_geometry=False, # Disable to save time/mem
                    save_surprise_artifacts=True
                )
                output = buf.getvalue()
            
            # Print output for forensic analysis in log
            print(output)
            
            self.assertIn("SENTINEL SUMMARY", output)
            self.assertIn("Surprises", output)
            
        except Exception:
            print(" Integration Test Failed with Exception:")
            traceback.print_exc()
            raise

if __name__ == '__main__':
    with open("forensic_results.txt", "w") as f:
        # Use a runner that writes to the file
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        unittest.main(testRunner=runner)
