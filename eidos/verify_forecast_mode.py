import sys
import os
import importlib.util
import time
import json
import shutil

# Setup
os.environ["EIDOS_DATA_SOURCE_TYPE"] = "LOCAL"
os.environ["EIDOS_ARTIFACT_ROOT"] = r"C:\Users\roland\eidos_artifacts_forecast"

# Clean prior run
if os.path.exists(os.environ["EIDOS_ARTIFACT_ROOT"]):
    shutil.rmtree(os.environ["EIDOS_ARTIFACT_ROOT"])
os.makedirs(os.environ["EIDOS_ARTIFACT_ROOT"])

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

# Config for Test 2: Hybrid Mode + Forecast + Incidents
# Generate Synthetic CSV Data (Longer)
data_dir = r"d:\eidos\verify_data"
if os.path.exists(data_dir): shutil.rmtree(data_dir)
os.makedirs(data_dir)
csv_path = os.path.join(data_dir, "incident_test_data.csv")

with open(csv_path, "w") as f:
    f.write("x,y,z\n")
    # 1000 lines of quiet
    import random
    random.seed(42)
    for _ in range(1000):
        v = 0.1 + random.random() * 0.05
        f.write(f"{v},{v},{v}\n")
    # 100 lines of massive spike
    for _ in range(100):
        v = 50.0 + random.random() * 10
        f.write(f"{v},{v},{v}\n")

brain.DATA_SOURCE_TYPE = "LOCAL"
brain.LOCAL_MODE = "ARCHIVE"
brain.LOCAL_TARGET = data_dir

brain.EIDOS_BRAIN_CONFIG["steps"] = 3000
brain.EIDOS_BRAIN_CONFIG["mode"] = "hybrid"
brain.EIDOS_BRAIN_CONFIG["domain"] = "dataset" # Use dataset adapter
brain.EIDOS_BRAIN_CONFIG["forecast_enabled"] = True
brain.EIDOS_BRAIN_CONFIG["incident_cards_enabled"] = True
brain.EIDOS_BRAIN_CONFIG["incident_min_gap_steps"] = 1 # Allow quick trigger
brain.EIDOS_BRAIN_CONFIG["procedural_enabled"] = True
brain.EIDOS_BRAIN_CONFIG["procedural_policy"] = "recommend"
brain.EIDOS_BRAIN_CONFIG["demo_enable"] = True
brain.EIDOS_BRAIN_CONFIG["procedural_policy"] = "recommend"
brain.EIDOS_BRAIN_CONFIG["demo_enable"] = True
brain.EIDOS_BRAIN_CONFIG["demo_every"] = 1
brain.EIDOS_BRAIN_CONFIG["warmup_cap"] = 50 # Short warmup for test

# Ensure demo plots off
brain.EIDOS_BRAIN_CONFIG["demo_plot_matplotlib"] = False

print("Starting Verification Test 2: Forecast & Incident Mode")
# Redirect stdout/stderr to file to capture debugs
log_file = r"d:\eidos\verify_run.log"
import sys

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure flush
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)
sys.stderr = sys.stdout

try:
    brain.run_eidos_sentinel()
    print("Test 2 Complete")
except Exception as e:
    print(f"Test 2 Failed: {e}")
    import traceback
    traceback.print_exc()


# Verify Artifacts
print("Verifying artifacts...")
root = os.environ["EIDOS_ARTIFACT_ROOT"]
cards_path = os.path.join(root, "incident_cards.jsonl")
forecast_path = os.path.join(root, "forecast.jsonl")

if os.path.exists(cards_path):
    print(f"PASS: {cards_path} exists.")
    with open(cards_path, "r") as f:
        print(f"  Incident Cards Generated: {sum(1 for _ in f)}")
else:
    print(f"FAIL: {cards_path} not found. (Maybe no incidents were triggered?)")

if os.path.exists(forecast_path):
    print(f"PASS: {forecast_path} exists.")
    with open(forecast_path, "r") as f:
        print(f"  Forecasts Generated: {sum(1 for _ in f)}")

else:
    print(f"FAIL: {forecast_path} not found.")

# Minimal check of content
if os.path.exists(cards_path):
    with open(cards_path, "r") as f:
        line = f.readline()
        if line:
            data = json.loads(line)
            print(f"  Sample Card Keys: {list(data.keys())}")
            print(f"  Sample Card Evidence: {data.get('evidence', {}).keys()}")
