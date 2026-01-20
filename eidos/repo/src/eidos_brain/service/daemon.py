"""
daemon.py

Main entrypoint for the Eidos Brain service (Daemon mode).
Runs the engine in a loop or once.
"""

import argparse
import sys
import logging
import time
import json
import asyncio
from eidos_brain.utils.config import load_config
from eidos_brain.engine.adapters import run_session
from eidos_brain.io.hive_event import HiveEventV1

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("eidos-daemon")

def main():
    parser = argparse.ArgumentParser(description="Eidos Brain Service Daemon")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--once", action="store_true", help="Run a single session and exit")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--source", help="Override data source type (LOCAL, STREAM, etc.)")
    
    args = parser.parse_args()
    
    logger.info("Starting Eidos Daemon...")
    
    # Load Config
    config = load_config(args.config)
    
    # CLI Overrides
    if args.source:
        config["source_type"] = args.source
        
    running = True
    while running:
        logger.info(f"Launching session. Source: {config.get('source_type')}")
        
        # Run Engine
        try:
            result = run_session(config)
            
            # Emit Session End Event (mocked here, engine emits its own usually)
            # Standardize output
            logger.info(f"Session finished. Status: {result.get('status')}")
            
            # If we want to emit a HiveEvent for the session summary:
            if result.get("status") == "SUCCESS":
                 evt = HiveEventV1.create(
                     event_type="eidos.session.end.v1",
                     session_id=result.get("session_id", "unknown"),
                     payload=result,
                     source=config.get("source_type", "UNKNOWN")
                 )
                 print(evt.to_jsonl_line()) # Emit to stdout for capture
            
        except KeyboardInterrupt:
            logger.info("Stopping daemon...")
            break
        except Exception as e:
            logger.error(f"Session failed: {e}")
            
        if args.once:
            break
            
        if args.loop:
            time.sleep(5) # Cooldown between sessions
        else:
            break

if __name__ == "__main__":
    main()
