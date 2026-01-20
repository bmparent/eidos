
import sys
import os
import importlib.util

# Override environment variables
os.environ["EIDOS_DATA_SOURCE_TYPE"] = "LOCAL"
os.environ["EIDOS_ARTIFACT_ROOT"] = r"C:\Users\roland\eidos_artifacts_test"

# Path to the module
module_path = r"d:\eidos\EIDOS_BRAIN_UNIFIED_v0_4.7.02.py"
module_name = "eidos_brain"

try:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    brain = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = brain
    spec.loader.exec_module(brain)
except Exception as e:
    print(f"Failed to load module: {e}")
    sys.exit(1)

# Configure for Test 2: Trace Seal Enabled
brain.DATA_SOURCE_TYPE = "LOCAL"
brain.LOCAL_MODE = "SYNTHETIC"
brain.EIDOS_BRAIN_CONFIG["steps"] = 1000  # Short run
brain.EIDOS_BRAIN_CONFIG["trace_seal_enabled"] = True # Enabled
brain.EIDOS_BRAIN_CONFIG["trace_seal_rank"] = 4
brain.EIDOS_BRAIN_CONFIG["trace_seal_mix"] = 1.0 # 100% Trace Seal
brain.EIDOS_BRAIN_CONFIG["trace_seal_recalc_every"] = 100
brain.EIDOS_BRAIN_CONFIG["demo_enable"] = True
brain.EIDOS_BRAIN_CONFIG["demo_every"] = 100

# Run
print("Starting Verification Test 2: Synthetic Mode (Trace Seal Enabled)")
try:
    brain.run_eidos_sentinel()
    print("Test 2 Complete")
except Exception as e:
    print(f"Test 2 Failed: {e}")
    import traceback
    traceback.print_exc()
