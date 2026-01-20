"""
demo_runner.py

CLI entrypoint to run the end-to-end demo:
1. Validates environment
2. Launches engine (background or foreground)
3. Launches dashboard
"""

import argparse
import subprocess
import os
import sys
import time
import threading
import signal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "stream", "kaggle"], default="local")
    parser.add_argument("--config", help="Path to config file", default=None)
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    config_path = args.config
    
    env = os.environ.copy()
    
    if args.mode == "local":
        if not config_path:
            config_path = os.path.join(base_dir, "configs", "demo_local.yaml")
        print(f"Running LOCAL demo with config: {config_path}")
        
    elif args.mode == "stream":
        if not config_path:
            config_path = os.path.join(base_dir, "configs", "demo_stream.yaml")
        print(f"Running STREAM demo with config: {config_path}")
        
        # Launch Streamer
        print("Launching Demo Streamer...")
        streamer_proc = subprocess.Popen(
            [sys.executable, "-m", "eidos_brain.demo.demo_streamer"],
            env=env
        )
        
    # Launch Engine
    print("Launching Engine Daemon...")
    # We use the 'eidos' command (daemon main)
    # Check if installed, or run via python -m
    engine_cmd = [sys.executable, "-m", "eidos_brain.service.daemon", "--once", "--config", config_path]
    engine_proc = subprocess.Popen(engine_cmd, env=env)
    
    # Launch Dashboard
    print("Launching Dashboard...")
    dashboard_path = os.path.join(base_dir, "src", "eidos_brain", "demo", "dashboard.py")
    dashboard_cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path]
    dashboard_proc = subprocess.Popen(dashboard_cmd, env=env)
    
    try:
        engine_proc.wait()
        print("Engine finished. Dashboard still running. Press Ctrl+C to exit.")
        dashboard_proc.wait()
    except KeyboardInterrupt:
        print("Stopping...")
        if 'streamer_proc' in locals(): streamer_proc.terminate()
        engine_proc.terminate()
        dashboard_proc.terminate()

if __name__ == "__main__":
    main()
