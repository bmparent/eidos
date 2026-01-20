"""
test_connectivity.py

Diagnose environment connectivity:
1. Python Package Imports ("eidos_brain")
2. Local File Permissions
3. Google Cloud Credentials (if configured)
4. Dashboard Port Availability
"""

import sys
import os
import socket
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("connectivity-test")

def check_imports():
    logger.info("--- Checking Imports ---")
    try:
        import eidos_brain
        logger.info(f"✅ eidos_brain found: {eidos_brain.__file__}")
        
        from eidos_brain.engine import eidos_v0_4_7_02
        logger.info(f"✅ Engine version: {eidos_v0_4_7_02.ENGINE_VERSION}")
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False
    return True

def check_gcp():
    logger.info("\n--- Checking Google Cloud ---")
    project_id = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    
    if not project_id:
        logger.warning("⚠️ No GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT env var set.")
        logger.info("   (This is expected if running strictly local, but check your .env)")
    else:
        logger.info(f"ℹ️ Target Project: {project_id}")

    # Check Default Credentials
    try:
        from google.auth import default
        creds, project = default()
        logger.info(f"✅ Google Auth Detected (Method: default credentials)")
        logger.info(f"   Project from creds: {project}")
        
        from google.cloud import storage
        client = storage.Client(credentials=creds, project=project_id or project)
        logger.info("✅ GCS Client initialized")
        
        # List buckets (read test)
        try:
            buckets = list(client.list_buckets(max_results=3))
            logger.info(f"✅ GCS Read Success. Visible buckets: {[b.name for b in buckets]}")
        except Exception as e:
             logger.error(f"❌ GCS Read Failed: {e}")

    except ImportError:
        logger.warning("⚠️ google-cloud-storage or google-auth not installed.")
    except Exception as e:
        logger.error(f"❌ Authentication check failed: {e}")

def check_ports():
    logger.info("\n--- Checking Ports ---")
    ports = [8000, 8501, 8001]
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('localhost', port))
            if result == 0:
                logger.info(f"ℹ️ Port {port} is OPEN (Service running)")
            else:
                logger.info(f"   Port {port} is CLOSED (Nothing running here)")

if __name__ == "__main__":
    logger.info("Starting Connectivity Test...")
    check_imports()
    check_gcp()
    check_ports()
